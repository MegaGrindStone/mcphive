package mcphive

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"time"

	"github.com/MegaGrindStone/go-mcp"
)

// Hive represents a central coordination structure for managing both internal and external tools.
// It serves as a registry and execution environment for tools, handling tool discovery,
// invocation, and interactions between tools and language models. Hive manages tool permissions
// and provides a unified interface for accessing both local and remote tool capabilities.
type Hive struct {
	info             mcp.Info
	tools            []Tool
	mcpClients       []*mcp.Client
	exposedMCPTools  []mcp.Tool
	mcpTools         []mcp.Tool
	mcpClientConfigs []MCPClientConfig

	internalToolsMap map[string]Tool // Map of tool names to tools
	externalToolsMap map[string]int  // Map of tool names to mcpClient indices

	mcpServer  mcp.Server
	httpServer *http.Server

	logger *slog.Logger
}

// Options is a function type used for configuring a Hive instance through the builder pattern.
// It allows for modular addition of features and dependencies to the Hive.
type Options func(hive *Hive)

var errToolNotFound = errors.New("tool not found")

// New creates a new Hive instance with the specified tools and configuration options.
// The info parameter provides identification and capability information for the Hive.
// The tools parameter is a slice of Tool instances to be registered with the Hive.
// Additional options can be provided to further configure the Hive's behavior.
// It returns an initialized Hive or an error if configuration fails.
func New(info mcp.Info, tools []Tool, options ...Options) (*Hive, error) {
	h := &Hive{
		info:             info,
		tools:            tools,
		internalToolsMap: make(map[string]Tool),
		externalToolsMap: make(map[string]int),
	}

	for _, opt := range options {
		opt(h)
	}

	if h.logger == nil {
		h.logger = slog.Default()
	}

	for _, cliConfig := range h.mcpClientConfigs {
		cli, err := cliConfig.MCPClient(info, h.logger)
		if err != nil {
			return nil, fmt.Errorf("failed to create MCP client: %w", err)
		}
		h.mcpClients = append(h.mcpClients, cli)
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

// WithExternalMCPClients returns an Options function that adds external MCP client configurations to the Hive.
// The configs parameter is a slice of MCPClientConfig implementations that specify how to create
// and manage connections to external MCP-compatible services.
// When this option is applied, the Hive will establish connections to these external services,
// discover their available tools, and register them for use alongside internal tools.
func WithExternalMCPClients(configs []MCPClientConfig) Options {
	return func(hive *Hive) {
		hive.mcpClientConfigs = append(hive.mcpClientConfigs, configs...)
	}
}

// WithLogger returns an Options function that sets the logger to be used by the Hive.
// The logger parameter is the slog.Logger instance that will be used for logging operations within the Hive.
// If not provided, the Hive will default to using slog.Default().
func WithLogger(logger *slog.Logger) Options {
	return func(hive *Hive) {
		hive.logger = logger
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
func (h *Hive) ListTools(
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
func (h *Hive) CallTool(
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
func (h *Hive) CallInternalTool(ctx context.Context, params mcp.CallToolParams) (mcp.CallToolResult, error) {
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

// ServeSSE starts the Hive as an SSE (Server-Sent Events) server on the specified port.
// It initializes an MCP server using the Hive's info and configures HTTP handlers for
// SSE communication. The server runs until an error occurs or until explicitly shut down.
// The port parameter specifies which TCP port to listen on.
// It returns an error if the HTTP server fails to start or encounters an error while running.
func (h *Hive) ServeSSE(port int) error {
	sse := mcp.NewSSEServer("/message", mcp.WithSSEServerLogger(h.logger))

	go h.serve(sse)

	h.httpServer = &http.Server{
		Addr:              fmt.Sprintf(":%d", port),
		ReadHeaderTimeout: 5 * time.Second,
	}

	http.Handle("/sse", sse.HandleSSE())
	http.Handle("/message", sse.HandleMessage())

	h.logger.Info("Hive is serving on port", slog.Int("port", port))

	return h.httpServer.ListenAndServe()
}

// ServeStdIO starts the Hive as a standard I/O server using the provided reader and writer.
// It initializes an MCP server using the Hive's info and configures it to communicate
// through the given standard I/O streams. This method blocks until the server is shut down.
// The reader parameter is the input stream for receiving messages.
// The writer parameter is the output stream for sending responses.
func (h *Hive) ServeStdIO(reader io.Reader, writer io.Writer) {
	srvIO := mcp.NewStdIO(reader, writer, mcp.WithStdIOLogger(h.logger))

	h.serve(srvIO)
}

// ShutdownSSE gracefully stops the SSE server and associated MCP server.
// It attempts to complete all in-flight requests and connections before shutting down,
// with a timeout of 10 seconds. Both the MCP server and HTTP server are shut down in sequence.
// It returns an error if either server fails to shut down properly within the timeout period.
func (h *Hive) ShutdownSSE() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := h.mcpServer.Shutdown(ctx); err != nil {
		h.logger.Error("failed to shutdown mcp server", slog.String("err", err.Error()))
		return err
	}

	if err := h.httpServer.Shutdown(ctx); err != nil {
		h.logger.Error("failed to shutdown http server", slog.String("err", err.Error()))
		return err
	}

	if err := h.disconnectMCPClients(ctx); err != nil {
		h.logger.Error("failed to disconnect MCP clients", slog.String("err", err.Error()))
		return err
	}

	return nil
}

// ShutdownStdIO gracefully stops the standard I/O server.
// It attempts to complete all in-flight requests before shutting down,
// with a timeout of 10 seconds. Only the MCP server needs to be shut down
// for StdIO mode since there is no separate HTTP server.
// It returns an error if the server fails to shut down properly within the timeout period.
func (h *Hive) ShutdownStdIO() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := h.mcpServer.Shutdown(ctx); err != nil {
		h.logger.Error("failed to shutdown mcp server", slog.String("err", err.Error()))
		return err
	}

	if err := h.disconnectMCPClients(ctx); err != nil {
		h.logger.Error("failed to disconnect MCP clients", slog.String("err", err.Error()))
		return err
	}

	return nil
}

func (h *Hive) serve(transport mcp.ServerTransport) {
	h.mcpServer = mcp.NewServer(h.info, transport,
		mcp.WithToolServer(h), mcp.WithServerLogger(h.logger))
	h.mcpServer.Serve()
}

func (h *Hive) disconnectMCPClients(ctx context.Context) error {
	for _, cliConfig := range h.mcpClientConfigs {
		if err := cliConfig.Shutdown(); err != nil {
			return fmt.Errorf("failed to shutdown MCP client: %w", err)
		}
	}
	for _, cli := range h.mcpClients {
		if err := cli.Disconnect(ctx); err != nil {
			return fmt.Errorf("failed to disconnect MCP client: %w", err)
		}
	}
	return nil
}

func (h *Hive) llmCaller(
	llm LLM,
	tools []mcp.Tool,
) func(ctx context.Context, messages []Message) iter.Seq2[Content, error] {
	return func(ctx context.Context, messages []Message) iter.Seq2[Content, error] {
		return h.callLLM(ctx, llm, messages, tools)
	}
}

func (h *Hive) callLLM(
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

func (h *Hive) callAllTool(ctx context.Context, params mcp.CallToolParams) (mcp.CallToolResult, error) {
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
