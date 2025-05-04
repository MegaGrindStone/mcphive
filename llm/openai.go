package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"slices"

	"github.com/MegaGrindStone/go-mcp"
	"github.com/MegaGrindStone/mcphive"
	goopenai "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

// OpenAI represents a client for the OpenAI API that implements the LLM interface.
// It handles communication with OpenAI's models and supports streaming responses,
// tool calling, and structured JSON outputs.
type OpenAI struct {
	model  string
	client *goopenai.Client

	endpoint        string
	developerPrompt string

	responseSchema *goopenai.ChatCompletionResponseFormatJSONSchema
}

// OpenAIOption is a functional option that configures an OpenAI client.
type OpenAIOption func(o *OpenAI) error

// NewOpenAI creates a new OpenAI client with the provided API key and model name.
// Additional options can be provided to further configure the client.
// The apiKey parameter is used for authentication with the OpenAI API.
// The model parameter specifies which model ID to use (e.g., "gpt-4").
// The options parameter allows for additional configuration of the client.
// It returns an initialized OpenAI client or an error if initialization fails.
func NewOpenAI(apiKey, model string, options ...OpenAIOption) (OpenAI, error) {
	o := OpenAI{
		model: model,
	}
	for _, opt := range options {
		if err := opt(&o); err != nil {
			return OpenAI{}, fmt.Errorf("failed to apply option: %w", err)
		}
	}

	// Create either default client or one with custom endpoint
	if o.endpoint == "" {
		o.client = goopenai.NewClient(apiKey)
	} else {
		cfg := goopenai.DefaultConfig(apiKey)
		cfg.BaseURL = o.endpoint
		o.client = goopenai.NewClientWithConfig(cfg)
	}

	return o, nil
}

// OpenAIWithEndpoint returns an option to set a custom API endpoint for the OpenAI client.
// This is useful for using OpenAI-compatible APIs or proxies.
// The endpoint parameter contains the base URL for the API.
// It returns an OpenAIOption that can be passed to NewOpenAI.
func OpenAIWithEndpoint(endpoint string) OpenAIOption {
	return func(o *OpenAI) error {
		o.endpoint = endpoint
		return nil
	}
}

// OpenAIWithDeveloperPrompt returns an option to set a developer prompt for the OpenAI client.
// The developer prompt provides context and instructions that will be included at the beginning
// of conversations if no developer role message is present.
// The devPrompt parameter contains the developer prompt text to be used.
// It returns an OpenAIOption that can be passed to NewOpenAI.
func OpenAIWithDeveloperPrompt(devPrompt string) OpenAIOption {
	return func(o *OpenAI) error {
		o.developerPrompt = devPrompt
		return nil
	}
}

// OpenAIWithJSONResponse returns an option that configures the OpenAI client
// to generate responses in a specific JSON format based on the provided type.
// The name parameter specifies the name of the schema.
// The response parameter should be a struct or type that represents the expected response format.
// The strict parameter determines whether the model must strictly adhere to the schema.
// It returns an OpenAIOption that can be passed to NewOpenAI.
func OpenAIWithJSONResponse(name string, response any, strict bool) OpenAIOption {
	return func(o *OpenAI) error {
		schema, err := jsonschema.GenerateSchemaForType(response)
		if err != nil {
			return fmt.Errorf("failed to generate schema for response: %w", err)
		}
		o.responseSchema = &goopenai.ChatCompletionResponseFormatJSONSchema{
			Name:   name,
			Schema: schema,
			Strict: strict,
		}
		return nil
	}
}

func openAIMessages(messages []mcphive.Message, devPrompt string) []goopenai.ChatCompletionMessage {
	msgs := make([]goopenai.ChatCompletionMessage, 0, len(messages))
	devPromptFound := false
	for _, msg := range messages {
		if msg.Role == "developer" {
			devPromptFound = true
		}
		for _, ct := range msg.Contents {
			switch ct.Type {
			case mcphive.ContentTypeText:
				if ct.Text != "" {
					msgs = append(msgs, goopenai.ChatCompletionMessage{
						Role:    string(msg.Role),
						Content: ct.Text,
					})
				}
			case mcphive.ContentTypeCallTool:
				msgs = append(msgs, goopenai.ChatCompletionMessage{
					Role: string(msg.Role),
					ToolCalls: []goopenai.ToolCall{
						{
							Type: "function",
							ID:   ct.CallToolID,
							Function: goopenai.FunctionCall{
								Name:      ct.ToolName,
								Arguments: string(ct.ToolInput),
							},
						},
					},
				})
			case mcphive.ContentTypeToolResult:
				msgs = append(msgs, goopenai.ChatCompletionMessage{
					Role:       "tool",
					Content:    string(ct.ToolResult),
					ToolCallID: ct.CallToolID,
				})
			}
		}
	}
	// Prepend developer prompt if not already present in messages
	if !devPromptFound {
		if devPrompt != "" {
			msgs = slices.Insert(msgs, 0, goopenai.ChatCompletionMessage{
				Role:    "developer",
				Content: devPrompt,
			})
		}
	}
	return msgs
}

// Call sends a request to the OpenAI API with the provided messages and tools,
// and returns an iterator of Content and error values.
// The function streams the response from the API and yields content elements as they become available.
// The ctx parameter provides a context for controlling the request lifecycle.
// The messages parameter contains the conversation history to send to the model.
// The tools parameter specifies tools that the model can use during the conversation.
// It returns an iterator that produces Content elements and potential errors.
func (o OpenAI) Call(
	ctx context.Context,
	messages []mcphive.Message,
	tools []mcp.Tool,
) iter.Seq2[mcphive.Content, error] {
	return func(yield func(mcphive.Content, error) bool) {
		msgs := openAIMessages(messages, o.developerPrompt)

		// Convert our tool interface to OpenAI's tool format
		oTools := make([]goopenai.Tool, len(tools))
		for i, tool := range tools {
			oTools[i] = goopenai.Tool{
				Type: goopenai.ToolTypeFunction,
				Function: &goopenai.FunctionDefinition{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  tool.InputSchema,
				},
			}
		}

		req := o.chatRequest(msgs, oTools, true)

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		stream, err := o.client.CreateChatCompletionStream(ctx, req)
		if err != nil {
			yield(mcphive.Content{}, fmt.Errorf("error sending request: %w", err))
			return
		}

		// Track tool calling state across stream chunks
		toolUse := false
		toolArgs := ""
		callToolContent := mcphive.Content{
			Type: mcphive.ContentTypeCallTool,
		}
		for {
			response, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				if errors.Is(err, context.Canceled) {
					return
				}
				yield(mcphive.Content{}, fmt.Errorf("error receiving response: %w", err))
				return
			}

			if len(response.Choices) == 0 {
				continue
			}

			res := response.Choices[0].Delta
			if res.Content != "" {
				if !yield(mcphive.Content{
					Type: mcphive.ContentTypeText,
					Text: res.Content,
				}, nil) {
					return
				}
			}
			if len(res.ToolCalls) > 0 {
				// Accumulate tool arguments across stream chunks
				toolArgs += res.ToolCalls[0].Function.Arguments
				if !toolUse {
					// Initialize tool call data on first chunk with tool call
					toolUse = true
					callToolContent.ToolName = res.ToolCalls[0].Function.Name
					callToolContent.CallToolID = res.ToolCalls[0].ID
				}
			}
		}
		// After stream completes, yield the tool call if one was made
		if toolUse {
			if toolArgs == "" {
				toolArgs = "{}"
			}
			callToolContent.ToolInput = json.RawMessage(toolArgs)
			yield(callToolContent, nil)
		}
	}
}

func (o OpenAI) chatRequest(
	messages []goopenai.ChatCompletionMessage,
	tools []goopenai.Tool,
	stream bool,
) goopenai.ChatCompletionRequest {
	req := goopenai.ChatCompletionRequest{
		Model:    o.model,
		Messages: messages,
		Stream:   stream,
		Tools:    tools,
	}

	// Add JSON schema response format if configured
	if o.responseSchema != nil {
		req.ResponseFormat = &goopenai.ChatCompletionResponseFormat{
			Type:       goopenai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: o.responseSchema,
		}
	}

	return req
}
