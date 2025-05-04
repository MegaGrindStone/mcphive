package mcphive

import (
	"context"
	"encoding/json"
	"iter"
	"log/slog"
	"slices"

	"github.com/MegaGrindStone/go-mcp"
)

// Tool represents a tool within the MCPHive framework that can be exposed to an LLM.
// It wraps an MCP tool with additional functionality for controlling access to other tools,
// handling execution, and managing LLM interactions.
type Tool struct {
	mcpTool mcp.Tool
	handle  Handle
	exposed bool

	llm LLM

	includedTools []string
	excludedTools []string
	excludeAll    bool
}

// HandleArgs contains the arguments passed to a tool's handle function when the tool is invoked.
// It provides access to the tool parameters, functions for calling other tools or the LLM,
// and a logger for recording information during tool execution.
type HandleArgs struct {
	Params   json.RawMessage
	CallTool func(ctx context.Context, params mcp.CallToolParams) (mcp.CallToolResult, error)
	CallLLM  func(ctx context.Context, messages []Message) iter.Seq2[Content, error]
	Logger   *slog.Logger
}

// Handle is a function type that defines how a tool processes its input and produces a result.
// It receives a context and HandleArgs, and returns a CallToolResult or an error.
type Handle func(ctx context.Context, args HandleArgs) (mcp.CallToolResult, error)

// ToolOptions is a function type used for configuring a Tool with additional options
// through the builder pattern.
type ToolOptions func(tool *Tool)

// NewTool creates a new Tool with the specified MCP tool, handle function, and exposure setting.
// The mcpTool parameter is the underlying MCP tool being wrapped.
// The handle parameter is the function that will be called when the tool is invoked.
// The expose parameter determines if the tool should be exposed to the LLM.
// Additional options can be provided to further configure the tool.
// It returns an initialized Tool configured according to the provided options.
func NewTool(mcpTool mcp.Tool, handle Handle, expose bool, options ...ToolOptions) Tool {
	t := Tool{
		mcpTool:       mcpTool,
		handle:        handle,
		exposed:       expose,
		includedTools: make([]string, 0),
		excludedTools: make([]string, 0),
	}
	for _, opt := range options {
		opt(&t)
	}

	return t
}

// WithLLM returns a ToolOptions function that sets the LLM to be used by the tool.
// The llm parameter is the LLM instance to be associated with the tool.
// It returns a ToolOptions function that can be passed to NewTool.
func WithLLM(llm LLM) ToolOptions {
	return func(tool *Tool) {
		tool.llm = llm
	}
}

// WithIncludedTools returns a ToolOptions function that sets the list of tool names
// that are visible to the LLM when called through this tool.
// If provided, only tools in this list will be visible to the LLM (unless explicitly
// excluded). If the includedTools list is empty, all tools are visible by default
// unless otherwise excluded. Note that these restrictions only affect what tools are
// visible to the LLM; the tool handler itself always has direct access to all tools
// via the CallTool function regardless of these settings.
func WithIncludedTools(tools []string) ToolOptions {
	return func(tool *Tool) {
		tool.includedTools = tools
	}
}

// WithExcludedTools returns a ToolOptions function that sets the list of tool names
// that should be hidden from the LLM when called through this tool.
// Tools in this list will never be visible to the LLM, regardless of whether they appear
// in the includedTools list. Note that these restrictions only affect what tools are
// visible to the LLM; the tool handler itself always has direct access to all tools
// via the CallTool function regardless of these settings.
func WithExcludedTools(tools []string) ToolOptions {
	return func(tool *Tool) {
		tool.excludedTools = tools
	}
}

// WithExcludeAllTools returns a ToolOptions function that hides all other tools
// from the LLM when called through this tool.
// When this option is set, the LLM will not see any other tools, overriding both
// includedTools and excludedTools settings. Note that this restriction only affects
// what tools are visible to the LLM; the tool handler itself always has direct access
// to all tools via the CallTool function regardless of this setting.
func WithExcludeAllTools() ToolOptions {
	return func(tool *Tool) {
		tool.excludeAll = true
	}
}

func (t Tool) isToolIncluded(name string) bool {
	if t.mcpTool.Name == name {
		return false
	}
	if t.excludeAll {
		return false
	}
	if slices.Contains(t.excludedTools, name) {
		return false
	}
	if len(t.includedTools) == 0 {
		return true
	}
	if slices.Contains(t.includedTools, name) {
		return true
	}
	return false
}
