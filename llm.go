package mcphive

import (
	"context"
	"encoding/json"
	"iter"
	"time"

	"github.com/MegaGrindStone/go-mcp"
)

// LLM is a language model.
type LLM interface {
	Call(ctx context.Context, messages []Message, tools []mcp.Tool) iter.Seq2[Content, error]
}

// Message is a message in a chat.
type Message struct {
	Role      Role
	Contents  []Content
	Timestamp time.Time
}

// Content is a message content with its type.
type Content struct {
	Type ContentType

	// Text would be filled if Type is ContentTypeText.
	Text string

	// ToolName would be filled if Type is ContentTypeCallTool.
	ToolName string
	// ToolInput would be filled if Type is ContentTypeCallTool.
	ToolInput json.RawMessage

	// ToolResult would be filled if Type is ContentTypeToolResult. The value would be either tool result or error.
	ToolResult json.RawMessage

	// CallToolID would be filled if Type is ContentTypeCallTool or ContentTypeToolResult.
	CallToolID string
	// CallToolFailed is a flag indicating if the call tool failed.
	// This flag would be set to true if the call tool failed and Type is ContentTypeToolResult.
	CallToolFailed bool
}

// Role represents the role of a message participant.
type Role string

// ContentType represents the type of content in messages.
type ContentType string

const (
	// RoleUser represents a user role.
	RoleUser Role = "user"
	// RoleAssistant represents an assistant role.
	RoleAssistant Role = "assistant"

	// ContentTypeText represents text content.
	ContentTypeText ContentType = "text"
	// ContentTypeCallTool represents a call to a tool.
	ContentTypeCallTool ContentType = "call_tool"
	// ContentTypeToolResult represents the result of a tool call.
	ContentTypeToolResult ContentType = "tool_result"
)
