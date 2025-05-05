package mcphive

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"

	"github.com/MegaGrindStone/go-mcp"
)

// Hive represents a central coordination structure for managing both internal and external tools.
// It serves as a registry and execution environment for tools, handling tool discovery,
// invocation, and interactions between tools and language models. Hive manages tool permissions
// and provides a unified interface for accessing both local and remote tool capabilities.
type Hive struct {
	tools           []Tool
	mcpClients      []*mcp.Client
	exposedMCPTools []mcp.Tool
	mcpTools        []mcp.Tool

	internalToolsMap map[string]Tool // Map of tool names to tools
	externalToolsMap map[string]int  // Map of tool names to mcpClient indices

	logger *slog.Logger
}

// Options is a function type used for configuring a Hive instance through the builder pattern.
// It allows for modular addition of features and dependencies to the Hive.
type Options func(hive *Hive) error

var errToolNotFound = errors.New("tool not found")

// New creates a new Hive instance with the specified tools and configuration options.
// The tools parameter is a slice of Tool instances to be registered with the Hive.
// Additional options can be provided to further configure the Hive's behavior.
// It returns an initialized Hive or an error if configuration fails.
func New(tools []Tool, options ...Options) (Hive, error) {
	h := Hive{
		tools:            tools,
		internalToolsMap: make(map[string]Tool),
		externalToolsMap: make(map[string]int),
	}

	for _, opt := range options {
		if err := opt(&h); err != nil {
			return Hive{}, fmt.Errorf("error applying option: %w", err)
		}
	}

	if h.logger == nil {
		h.logger = slog.Default()
	}

	for _, tool := range tools {
		h.internalToolsMap[tool.mcpTool.Name] = tool
		h.mcpTools = append(h.mcpTools, tool.mcpTool)
		if !tool.exposed {
			continue
		}
		h.exposedMCPTools = append(h.exposedMCPTools, tool.mcpTool)
	}

	return h, nil
}

// WithExternalMCPClients returns an Options function that adds external MCP clients to the Hive.
// The clients parameter is a slice of MCP clients whose tools will be made available through the Hive.
// When this option is applied, the Hive will discover and register tools from these external clients,
// allowing them to be invoked alongside internal tools.
func WithExternalMCPClients(clients []*mcp.Client) Options {
	return func(hive *Hive) error {
		for _, cli := range clients {
			serverInfo := cli.ServerInfo()

			if !cli.ToolServerSupported() {
				continue
			}

			listTools, err := cli.ListTools(context.Background(), mcp.ListToolsParams{})
			if err != nil {
				return fmt.Errorf("failed to list tools from server %s: %w", serverInfo.Name, err)
			}
			for _, tool := range listTools.Tools {
				hive.externalToolsMap[tool.Name] = len(hive.mcpClients)
				hive.mcpTools = append(hive.mcpTools, tool)
			}

			hive.mcpClients = append(hive.mcpClients, cli)
		}
		return nil
	}
}

// WithLogger returns an Options function that sets the logger to be used by the Hive.
// The logger parameter is the slog.Logger instance that will be used for logging operations within the Hive.
// If not provided, the Hive will default to using slog.Default().
func WithLogger(logger *slog.Logger) Options {
	return func(hive *Hive) error {
		hive.logger = logger
		return nil
	}
}

func processLLMIter(llmIter iter.Seq2[Content, error]) (Content, Content, bool, bool, error) {
	var textContent Content
	textContent.Type = ContentTypeText
	badToolInputFlag := false

	for content, err := range llmIter {
		if err != nil {
			return Content{}, Content{}, false, false, fmt.Errorf("failed to call llm: %w", err)
		}

		// Process different content types from the LLM's response
		switch content.Type {
		case ContentTypeText:
			textContent.Text += content.Text
		case ContentTypeCallTool:
			// Validate the tool input is valid JSON
			_, err := json.Marshal(content.ToolInput)
			if err != nil {
				badToolInputFlag = true
			}
			return textContent, content, true, badToolInputFlag, nil
		case ContentTypeToolResult:
			return Content{}, Content{}, false, false, fmt.Errorf("content type tool result not allowed")
		default:
			return Content{}, Content{}, false, false, fmt.Errorf("unknown content type: %s", content.Type)
		}
	}

	return textContent, Content{}, false, false, nil
}

func callToolError(err error) json.RawMessage {
	contents := []mcp.Content{
		{
			Type: mcp.ContentTypeText,
			Text: err.Error(),
		},
	}

	res, _ := json.Marshal(contents)
	return res
}

// ListTools returns a list of exposed tools available in the Hive.
// This method implements the ToolServer interface from the go-mcp package.
// It returns only the tools marked as exposed during Hive initialization.
// The context, parameters, progress reporter, and request client function parameters
// are part of the interface but unused in this implementation.
// The method always returns a ListToolsResult containing the list of available exposed tools
// with no error.
func (h Hive) ListTools(
	context.Context,
	mcp.ListToolsParams,
	mcp.ProgressReporter,
	mcp.RequestClientFunc,
) (mcp.ListToolsResult, error) {
	return mcp.ListToolsResult{
		Tools: h.exposedMCPTools,
	}, nil
}

// CallTool executes a tool with the given parameters.
// This method implements the ToolServer interface from the go-mcp package.
// It delegates to CallInternalTool to execute the tool and handles error cases.
// The context is used for cancellation, and params contains the tool name and arguments.
// The progress reporter and request client function are part of the interface but unused.
// It returns a CallToolResult containing the tool execution result or an error if the tool
// is not found or if execution fails.
func (h Hive) CallTool(
	ctx context.Context,
	params mcp.CallToolParams,
	_ mcp.ProgressReporter,
	_ mcp.RequestClientFunc,
) (mcp.CallToolResult, error) {
	res, err := h.CallInternalTool(ctx, params)
	if err != nil {
		if errors.Is(err, errToolNotFound) {
			return mcp.CallToolResult{}, fmt.Errorf("tool %s not found", params.Name)
		}
		return mcp.CallToolResult{}, err
	}
	return res, nil
}

// CallInternalTool executes an internal tool registered with the Hive.
// It finds the tool by name, prepares the available tools that this tool can access,
// and then calls the tool's handler with the prepared arguments. The context is used
// for cancellation, and params contains the tool name and arguments to pass to the tool.
// It returns a CallToolResult containing the tool execution result or errToolNotFound if
// the tool is not found. Other errors may be returned if the execution fails.
func (h Hive) CallInternalTool(ctx context.Context, params mcp.CallToolParams) (mcp.CallToolResult, error) {
	internalTool, ok := h.internalToolsMap[params.Name]
	if !ok {
		return mcp.CallToolResult{}, errToolNotFound
	}
	tools := make([]mcp.Tool, 0, len(h.mcpTools))
	if !internalTool.excludeAll {
		for _, tool := range h.mcpTools {
			if internalTool.isToolIncluded(tool.Name) {
				tools = append(tools, tool)
			}
		}
	}
	args := HandleArgs{
		Params:   params.Arguments,
		CallTool: h.callAllTool,
		CallLLM:  h.llmCaller(internalTool.llm, tools),
		Logger:   h.logger.With(slog.String("tool", params.Name)),
	}
	return internalTool.handle(ctx, args)
}

func (h Hive) llmCaller(
	llm LLM,
	tools []mcp.Tool,
) func(ctx context.Context, messages []Message) iter.Seq2[Content, error] {
	return func(ctx context.Context, messages []Message) iter.Seq2[Content, error] {
		return h.callLLM(ctx, llm, messages, tools)
	}
}

func (h Hive) callLLM(
	ctx context.Context,
	llm LLM,
	messages []Message,
	tools []mcp.Tool,
) iter.Seq2[Content, error] {
	return func(yield func(Content, error) bool) {
		if llm == nil {
			yield(Content{}, errors.New("llm is not set"))
			return
		}

		// Create a copy of messages and add an assistant message to build the response
		currentMessages := make([]Message, len(messages)+1)
		copy(currentMessages, messages)
		currentMessages[len(messages)] = Message{
			Role: RoleAssistant,
		}
		lastMsgIndex := len(currentMessages) - 1
		lastMsg := currentMessages[lastMsgIndex]

		for {
			// Call the LLM with the current conversation state and available tools
			llmIter := llm.Call(ctx, currentMessages, tools)

			textContent, callToolContent, callTool, badToolInputFlag, err := processLLMIter(llmIter)
			if err != nil {
				yield(Content{}, err)
				return
			}

			if textContent.Text != "" {
				if !yield(textContent, nil) {
					return
				}
			}

			if !callTool {
				break
			}

			// Record the tool call in the conversation history
			lastMsg.Contents = append(lastMsg.Contents, callToolContent)
			if !yield(callToolContent, nil) {
				return
			}

			// Set up the tool result content to be added to the conversation
			toolResContent := Content{
				Type:       ContentTypeToolResult,
				CallToolID: callToolContent.CallToolID,
			}

			// Handle case where tool input was invalid JSON
			if badToolInputFlag {
				toolResContent.ToolResult = callToolError(
					fmt.Errorf("tool input %s is not valid json", string(callToolContent.ToolInput)))
				toolResContent.CallToolFailed = true
				lastMsg.Contents = append(lastMsg.Contents, toolResContent)
				currentMessages[lastMsgIndex] = lastMsg
				if !yield(toolResContent, nil) {
					return
				}
				continue
			}

			// Actually call the requested tool and handle any errors
			toolRes, err := h.callAllTool(ctx, mcp.CallToolParams{
				Name:      callToolContent.ToolName,
				Arguments: callToolContent.ToolInput,
			})
			if err != nil {
				toolResContent.ToolResult = callToolError(err)
				toolResContent.CallToolFailed = true
				lastMsg.Contents = append(lastMsg.Contents, toolResContent)
				currentMessages[lastMsgIndex] = lastMsg
				if !yield(toolResContent, nil) {
					return
				}
				continue
			}

			// Convert the successful tool result to JSON
			resContent, err := json.Marshal(toolRes.Content)
			if err != nil {
				toolResContent.ToolResult = callToolError(err)
				toolResContent.CallToolFailed = true
				lastMsg.Contents = append(lastMsg.Contents, toolResContent)
				currentMessages[lastMsgIndex] = lastMsg
				if !yield(toolResContent, nil) {
					return
				}
				continue
			}

			// Add the successful tool result to the conversation and continue the loop
			toolResContent.ToolResult = resContent
			toolResContent.CallToolFailed = false
			lastMsg.Contents = append(lastMsg.Contents, toolResContent)
			currentMessages[lastMsgIndex] = lastMsg
			if !yield(toolResContent, nil) {
				return
			}
		}
	}
}

func (h Hive) callAllTool(ctx context.Context, params mcp.CallToolParams) (mcp.CallToolResult, error) {
	res, err := h.CallInternalTool(ctx, params)
	if err != nil {
		if !errors.Is(err, errToolNotFound) {
			return mcp.CallToolResult{}, err
		}
	} else {
		return res, nil
	}

	clientIdx, ok := h.externalToolsMap[params.Name]
	if !ok {
		return mcp.CallToolResult{}, fmt.Errorf("tool %s not found in internal and external tools map", params.Name)
	}

	mcpClient := h.mcpClients[clientIdx]
	return mcpClient.CallTool(ctx, params)
}
