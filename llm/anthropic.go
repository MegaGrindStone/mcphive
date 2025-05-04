package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"

	"github.com/MegaGrindStone/go-mcp"
	"github.com/MegaGrindStone/mcphive"
	"github.com/sashabaranov/go-openai/jsonschema"
	"github.com/tmaxmax/go-sse"
)

// Anthropic represents a client for the Anthropic API that implements the LLM interface.
// It handles communication with Anthropic's Claude models and supports streaming responses,
// tool calling, and structured JSON outputs.
type Anthropic struct {
	apiKey         string
	model          string
	maxTokens      int
	systemPrompt   string
	responseSchema []byte

	client *http.Client
}

// AnthropicOption is a functional option that configures an Anthropic client.
type AnthropicOption func(a *Anthropic) error

type anthropicChatRequest struct {
	Model     string             `json:"model"`
	Messages  []anthropicMessage `json:"messages"`
	System    string             `json:"system"`
	MaxTokens int                `json:"max_tokens"`
	Tools     []anthropicTool    `json:"tools"`
	Stream    bool               `json:"stream"`
}

type anthropicMessage struct {
	Role    string                    `json:"role"`
	Content []anthropicMessageContent `json:"content"`
}

type anthropicMessageContent struct {
	Type string `json:"type"`

	// For text type.
	Text string `json:"text,omitempty"`

	// For tool_use type.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// For tool_result type.
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type anthropicContentBlockStart struct {
	Type         string
	ContentBlock struct {
		Type  string          `json:"type"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content_block"`
}

type anthropicContentBlockDelta struct {
	Type  string `json:"type"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
}

type anthropicError struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

const (
	anthropicAPIEndpoint = "https://api.anthropic.com/v1"
)

var responsePrompt = `
Your response must strictly adhere to the following JSON schema:
%s

IMPORTANT INSTRUCTIONS:
1. Return ONLY valid JSON that matches this schema exactly
2. Do not include any explanatory text before or after the JSON
3. Do not use markdown code blocks or backticks
4. Ensure all required fields are included
5. Make sure the JSON is properly formatted and valid`

// NewAnthropic creates a new Anthropic client with the provided API key, model name, and token limit.
// Additional options can be provided to further configure the client.
// The apiKey parameter is used for authentication with the Anthropic API.
// The model parameter specifies which model ID to use (e.g., "claude-3-opus-20240229").
// The maxTokens parameter sets the maximum number of tokens to generate in the response.
// The options parameter allows for additional configuration of the client.
// It returns an initialized Anthropic client or an error if initialization fails.
func NewAnthropic(apiKey, model string, maxTokens int, options ...AnthropicOption) (Anthropic, error) {
	a := Anthropic{
		apiKey:    apiKey,
		model:     model,
		maxTokens: maxTokens,
		client:    &http.Client{},
	}
	for _, opt := range options {
		if err := opt(&a); err != nil {
			return Anthropic{}, fmt.Errorf("failed to apply option: %w", err)
		}
	}

	return a, nil
}

// AnthropicWithSystemPrompt returns an option to set a system prompt for the Anthropic client.
// The system prompt provides context and instructions to the model that persist across messages.
// The prompt parameter contains the system prompt text to be used.
// It returns an AnthropicOption that can be passed to NewAnthropic.
func AnthropicWithSystemPrompt(prompt string) AnthropicOption {
	return func(a *Anthropic) error {
		a.systemPrompt = prompt
		return nil
	}
}

// AnthropicWithJSONResponse returns an option that configures the Anthropic client
// to generate responses in a specific JSON format based on the provided type.
// The type is used to generate a JSON schema that the model will follow.
// The response parameter should be a struct or type that represents the expected response format.
// It returns an AnthropicOption that can be passed to NewAnthropic.
func AnthropicWithJSONResponse(response any) AnthropicOption {
	return func(a *Anthropic) error {
		// Generate JSON schema from the provided type
		schema, err := jsonschema.GenerateSchemaForType(response)
		if err != nil {
			return fmt.Errorf("failed to generate schema for response: %w", err)
		}
		schemaJSON, err := json.Marshal(schema)
		if err != nil {
			return fmt.Errorf("failed to marshal schema: %w", err)
		}
		a.responseSchema = schemaJSON
		return nil
	}
}

// Call sends a request to the Anthropic API with the provided messages and tools,
// and returns an iterator of Content and error values.
// The function streams the response from the API and yields content elements as they become available.
// The ctx parameter provides a context for controlling the request lifecycle.
// The messages parameter contains the conversation history to send to the model.
// The tools parameter specifies tools that the model can use during the conversation.
// It returns an iterator that produces Content elements and potential errors.
func (a Anthropic) Call(
	ctx context.Context,
	messages []mcphive.Message,
	tools []mcp.Tool,
) iter.Seq2[mcphive.Content, error] {
	return func(yield func(mcphive.Content, error) bool) {
		resp, err := a.doRequest(ctx, messages, tools, true)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return
			}
			yield(mcphive.Content{}, fmt.Errorf("error sending request: %w", err))
			return
		}
		defer resp.Body.Close()

		// Track state for tool usage across SSE events
		isToolUse := false
		inputJSON := ""
		toolContent := mcphive.Content{
			Type: mcphive.ContentTypeCallTool,
		}
		for ev, err := range sse.Read(resp.Body, nil) {
			if err != nil {
				yield(mcphive.Content{}, fmt.Errorf("error reading response: %w", err))
				return
			}
			switch ev.Type {
			case "error":
				var e anthropicError
				if err := json.Unmarshal([]byte(ev.Data), &e); err != nil {
					yield(mcphive.Content{}, fmt.Errorf("error unmarshaling error: %w", err))
					return
				}
				yield(mcphive.Content{}, fmt.Errorf("anthropic error %s: %s", e.Error.Type, e.Error.Message))
				return
			case "message_stop":
				// End of the message stream
				return
			case "content_block_start":
				// Start of a new content block (might be text or tool use)
				var res anthropicContentBlockStart
				if err := json.Unmarshal([]byte(ev.Data), &res); err != nil {
					yield(mcphive.Content{}, fmt.Errorf("error unmarshaling block start: %w", err))
					return
				}
				if res.ContentBlock.Type != "tool_use" {
					continue
				}
				// Begin capturing tool use information
				isToolUse = true
				toolContent.ToolName = res.ContentBlock.Name
				toolContent.CallToolID = res.ContentBlock.ID
			case "content_block_delta":
				var res anthropicContentBlockDelta
				if err := json.Unmarshal([]byte(ev.Data), &res); err != nil {
					yield(mcphive.Content{}, fmt.Errorf("error unmarshaling block delta: %w", err))
					return
				}
				if isToolUse {
					// Accumulate JSON for tool input
					inputJSON += res.Delta.PartialJSON
					continue
				}
				// Yield text content as it arrives
				if !yield(mcphive.Content{
					Type: mcphive.ContentTypeText,
					Text: res.Delta.Text,
				}, nil) {
					return
				}
			case "content_block_stop":
				if !isToolUse {
					continue
				}

				// Handle empty tool inputs with a default empty object
				if inputJSON == "" {
					inputJSON = "{}"
				}
				toolContent.ToolInput = json.RawMessage(inputJSON)
				if !yield(toolContent, nil) {
					return
				}
				// Reset tool state for next potential tool call
				isToolUse = false
				inputJSON = ""
			default:
			}
		}
	}
}

func (a Anthropic) doRequest(
	ctx context.Context,
	messages []mcphive.Message,
	tools []mcp.Tool,
	stream bool,
) (*http.Response, error) {
	msgs, systemPrompt, err := a.convertMessages(messages)
	if err != nil {
		return nil, err
	}

	// Convert our internal tool representation to Anthropic's format
	aTools := make([]anthropicTool, len(tools))
	for i, tool := range tools {
		aTools[i] = anthropicTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
	}

	reqBody := anthropicChatRequest{
		Model:     a.model,
		Messages:  msgs,
		System:    systemPrompt,
		MaxTokens: a.maxTokens,
		Tools:     aTools,
		Stream:    stream,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		anthropicAPIEndpoint+"/messages", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	// Set required headers for Anthropic API
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d, body: %s, request: %s", resp.StatusCode, string(body), jsonBody)
	}

	return resp, nil
}

func (a Anthropic) convertMessages(messages []mcphive.Message) ([]anthropicMessage, string, error) {
	var msgs []anthropicMessage
	systemPrompt := a.systemPrompt

	for _, msg := range messages {
		if msg.Role == "system" {
			// Extract system message content rather than adding it as a regular message
			systemPrompt = msg.Contents[0].Text
			continue
		}

		if msg.Role == "user" {
			userMsg, err := a.processUserMessage(msg)
			if err != nil {
				return nil, "", err
			}
			msgs = append(msgs, userMsg)
			continue
		}

		// Handle assistant messages which may contain tool calls or results
		otherMsgs := a.processOtherRoleMessage(msg)
		msgs = append(msgs, otherMsgs...)
	}

	// Add JSON schema instruction if responseSchema is set
	if a.responseSchema != nil {
		prompt := fmt.Sprintf(responsePrompt, string(a.responseSchema))
		msgs = append(msgs, anthropicMessage{
			Role: "user",
			Content: []anthropicMessageContent{
				{
					Type: "text",
					Text: prompt,
				},
			},
		})
	}

	return msgs, systemPrompt, nil
}

func (a Anthropic) processUserMessage(msg mcphive.Message) (anthropicMessage, error) {
	contents := make([]anthropicMessageContent, 0, len(msg.Contents))

	for _, ct := range msg.Contents {
		switch ct.Type {
		case mcphive.ContentTypeText:
			if ct.Text != "" {
				contents = append(contents, anthropicMessageContent{
					Type: "text",
					Text: ct.Text,
				})
			}
		case mcphive.ContentTypeCallTool, mcphive.ContentTypeToolResult:
			// Users can't call tools or provide tool results directly
			return anthropicMessage{}, fmt.Errorf("content type %s is not supported for user messages", ct.Type)
		}
	}

	return anthropicMessage{
		Role:    string(msg.Role),
		Content: contents,
	}, nil
}

func (a Anthropic) processOtherRoleMessage(msg mcphive.Message) []anthropicMessage {
	var msgs []anthropicMessage
	contents := make([]anthropicMessageContent, 0, len(msg.Contents))

	for _, ct := range msg.Contents {
		switch ct.Type {
		case mcphive.ContentTypeText:
			if ct.Text != "" {
				contents = append(contents, anthropicMessageContent{
					Type: "text",
					Text: ct.Text,
				})
			}
		case mcphive.ContentTypeCallTool:
			// Tool calls require special handling in Anthropic's API
			contents = append(contents, anthropicMessageContent{
				Type:  "tool_use",
				ID:    ct.CallToolID,
				Name:  ct.ToolName,
				Input: ct.ToolInput,
			})
			msgs = append(msgs, anthropicMessage{
				Role:    string(msg.Role),
				Content: contents,
			})
			// Reset contents after adding a tool call
			contents = make([]anthropicMessageContent, 0, len(msg.Contents))
		case mcphive.ContentTypeToolResult:
			// Tool results are sent as user messages in Anthropic's API
			msgs = append(msgs, anthropicMessage{
				Role: "user",
				Content: []anthropicMessageContent{
					{
						Type:      "tool_result",
						ToolUseID: ct.CallToolID,
						IsError:   ct.CallToolFailed,
						Content:   ct.ToolResult,
					},
				},
			})
		}
	}

	// Add any remaining contents as a message
	if len(contents) > 0 {
		msgs = append(msgs, anthropicMessage{
			Role:    string(msg.Role),
			Content: contents,
		})
	}

	return msgs
}
