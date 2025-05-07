package mcphive_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"log/slog"
	"testing"

	"github.com/MegaGrindStone/go-mcp"
	"github.com/MegaGrindStone/mcphive"
)

type mockLLM struct {
	responses []mcphive.Content
	err       error

	// Tool verification fields
	t             *testing.T // For test assertions
	verifyTools   bool       // Flag to enable tool verification
	expectedTools []string   // Tools that should be visible
	excludedTools []string   // Tools that should not be visible
}

func TestHiveNew(t *testing.T) {
	// Create tools
	echoTool := mcphive.NewTool(mcp.Tool{
		Name:        "echo",
		Description: "Echoes back the input",
	}, echoToolHandler, true)

	// Test with single tool
	_, err := mcphive.New(mcp.Info{}, []mcphive.Tool{echoTool})
	if err != nil {
		t.Fatalf("Failed to create Hive: %v", err)
	}

	// Test with logger option
	logger := slog.Default()
	_, err = mcphive.New(mcp.Info{}, []mcphive.Tool{echoTool}, mcphive.WithLogger(logger))
	if err != nil {
		t.Fatalf("Failed to create Hive with logger: %v", err)
	}
}

func TestHiveCallInternalTool(t *testing.T) {
	// Create tools
	echoTool := mcphive.NewTool(mcp.Tool{
		Name:        "echo",
		Description: "Echoes back the input",
	}, echoToolHandler, true)

	errorTool := mcphive.NewTool(mcp.Tool{
		Name:        "error",
		Description: "Always returns an error",
	}, errorToolHandler, true)

	// Create hive
	hive, err := mcphive.New(mcp.Info{}, []mcphive.Tool{echoTool, errorTool})
	if err != nil {
		t.Fatalf("Failed to create Hive: %v", err)
	}

	// Test successful tool call
	message := "Hello, world!"
	paramsJSON, _ := json.Marshal(map[string]string{"message": message})
	result, err := hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "echo",
		Arguments: paramsJSON,
	})
	if err != nil {
		t.Fatalf("CallInternalTool failed: %v", err)
	}

	if len(result.Content) != 1 || result.Content[0].Type != mcp.ContentTypeText {
		t.Fatalf("Unexpected content format: %v", result.Content)
	}

	expected := "Echo: " + message
	if result.Content[0].Text != expected {
		t.Errorf("Expected content '%s', got '%s'", expected, result.Content[0].Text)
	}

	// Test error from tool
	_, err = hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "error",
		Arguments: json.RawMessage("{}"),
	})

	if err == nil {
		t.Error("Expected error from tool, got nil")
	}

	// Test non-existent tool
	_, err = hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "nonexistent",
		Arguments: json.RawMessage("{}"),
	})

	if err == nil || err.Error() != "tool not found" {
		t.Errorf("Expected 'tool not found' error, got %v", err)
	}
}

func TestToolChaining(t *testing.T) {
	// Create tools
	echoTool := mcphive.NewTool(mcp.Tool{
		Name:        "echo",
		Description: "Echoes back the input",
	}, echoToolHandler, true)

	chainTool := mcphive.NewTool(mcp.Tool{
		Name:        "chain",
		Description: "Chains to another tool",
	}, chainToolHandler, true)

	// Create hive
	hive, err := mcphive.New(mcp.Info{}, []mcphive.Tool{echoTool, chainTool})
	if err != nil {
		t.Fatalf("Failed to create Hive: %v", err)
	}

	// Test chaining tools
	paramsJSON, _ := json.Marshal(map[string]string{
		"tool":    "echo",
		"message": "Chained message",
	})

	result, err := hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "chain",
		Arguments: paramsJSON,
	})
	if err != nil {
		t.Fatalf("CallInternalTool failed: %v", err)
	}

	expected := "Echo: Chained message"
	if result.Content[0].Text != expected {
		t.Errorf("Expected content '%s', got '%s'", expected, result.Content[0].Text)
	}
}

func TestLLMIntegration(t *testing.T) {
	// Create a mock LLM
	mockLLM := &mockLLM{
		responses: []mcphive.Content{
			{
				Type: mcphive.ContentTypeText,
				Text: "LLM response",
			},
		},
	}

	// Create tools with LLM
	llmTool := mcphive.NewTool(mcp.Tool{
		Name:        "llm",
		Description: "Calls the LLM",
	}, llmToolHandler, true, mcphive.WithLLM(mockLLM))

	// Create hive
	hive, err := mcphive.New(mcp.Info{}, []mcphive.Tool{llmTool})
	if err != nil {
		t.Fatalf("Failed to create Hive: %v", err)
	}

	// Test LLM integration
	paramsJSON, _ := json.Marshal(map[string]string{
		"message": "Hello, LLM!",
	})

	result, err := hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "llm",
		Arguments: paramsJSON,
	})
	if err != nil {
		t.Fatalf("CallInternalTool failed: %v", err)
	}

	expected := "LLM response"
	if result.Content[0].Text != expected {
		t.Errorf("Expected content '%s', got '%s'", expected, result.Content[0].Text)
	}

	// Test LLM error
	mockLLM.err = errors.New("LLM error")
	_, err = hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "llm",
		Arguments: paramsJSON,
	})

	if err == nil {
		t.Error("Expected error from LLM, got nil")
	}
}

func TestLLMTextFollowedByToolCall(t *testing.T) {
	// Create a mock LLM that sends text followed by a tool call
	mockLLM := &mockLLM{
		responses: []mcphive.Content{
			{
				Type: mcphive.ContentTypeText,
				Text: "I'm thinking about this question...",
			},
			{
				Type:       mcphive.ContentTypeCallTool,
				ToolName:   "echo",
				ToolInput:  json.RawMessage(`{"message":"Let me echo this message"}`),
				CallToolID: "call1",
			},
		},
	}

	// Create tools
	echoTool := mcphive.NewTool(mcp.Tool{
		Name:        "echo",
		Description: "Echoes back the input",
	}, echoToolHandler, true)

	llmTool := mcphive.NewTool(mcp.Tool{
		Name:        "llm_with_tool_call",
		Description: "Uses LLM that sends text followed by a tool call",
	}, llmToolHandler, true, mcphive.WithLLM(mockLLM))

	// Create hive
	hive, err := mcphive.New(mcp.Info{}, []mcphive.Tool{echoTool, llmTool})
	if err != nil {
		t.Fatalf("Failed to create Hive: %v", err)
	}

	// Call the tool
	paramsJSON, _ := json.Marshal(map[string]string{
		"message": "Please process with echo",
	})

	result, err := hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "llm_with_tool_call",
		Arguments: paramsJSON,
	})
	if err != nil {
		t.Fatalf("CallInternalTool failed: %v", err)
	}

	// Verify the result
	// The result should contain just the text content from the LLM
	if len(result.Content) != 1 || result.Content[0].Type != mcp.ContentTypeText {
		t.Fatalf("Expected 1 text content, got: %v", result.Content)
	}

	// The text should be the text from the LLM's response
	expectedText := "I'm thinking about this question..."
	if result.Content[0].Text != expectedText {
		t.Errorf("Expected text '%s', got '%s'", expectedText, result.Content[0].Text)
	}
}

func TestToolWithToolFiltering(t *testing.T) {
	// Create tools
	echoTool := mcphive.NewTool(mcp.Tool{
		Name:        "echo",
		Description: "Echoes back the input",
	}, echoToolHandler, true)

	pingTool := mcphive.NewTool(mcp.Tool{
		Name:        "ping",
		Description: "Returns pong",
	}, echoToolHandler, true) // Reusing echo handler for simplicity

	// This tool has echo excluded from LLM visibility,
	// but its handler can still call echo directly
	restrictedTool := mcphive.NewTool(mcp.Tool{
		Name:        "restricted",
		Description: "Has restricted LLM tool visibility but handler has direct access",
	}, chainToolHandler, true, mcphive.WithExcludedTools([]string{"echo"}))

	// Create an enhanced mockLLM that will verify tool visibility
	verifierLLM := &mockLLM{
		t:             t,
		verifyTools:   true,
		excludedTools: []string{"echo"},
		expectedTools: []string{"ping", "restricted"}, // Tools that should be visible
		responses: []mcphive.Content{
			{
				Type: mcphive.ContentTypeText,
				Text: "LLM visibility test complete",
			},
		},
	}

	// Tool with LLM that will verify tool visibility
	llmTool := mcphive.NewTool(mcp.Tool{
		Name:        "llm_restricted",
		Description: "Uses LLM with restricted tool visibility",
	}, llmToolHandler, true,
		mcphive.WithLLM(verifierLLM),
		mcphive.WithExcludedTools([]string{"echo"}))

	// Create hive with all tools
	hive, err := mcphive.New(mcp.Info{}, []mcphive.Tool{echoTool, pingTool, restrictedTool, llmTool})
	if err != nil {
		t.Fatalf("Failed to create Hive: %v", err)
	}

	// PART 1: Test that tool handler can call excluded tools directly
	paramsJSON, _ := json.Marshal(map[string]string{
		"tool":    "echo",
		"message": "Should succeed because handler has direct access",
	})

	result, err := hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "restricted",
		Arguments: paramsJSON,
	})
	// This should succeed because restrictions only apply to LLM visibility
	if err != nil {
		t.Errorf("Expected success because handler has direct access, got error: %v", err)
	}

	expected := "Echo: Should succeed because handler has direct access"
	if result.Content[0].Text != expected {
		t.Errorf("Expected content '%s', got '%s'", expected, result.Content[0].Text)
	}

	// PART 2: Test that LLM only sees the allowed tools
	paramsJSON, _ = json.Marshal(map[string]string{
		"message": "Check tool visibility",
	})

	// This call triggers the LLM which will verify tool visibility
	_, err = hive.CallInternalTool(context.Background(), mcp.CallToolParams{
		Name:      "llm_restricted",
		Arguments: paramsJSON,
	})
	if err != nil {
		t.Errorf("LLM tool call failed: %v", err)
	}
}

func echoToolHandler(_ context.Context, args mcphive.HandleArgs) (mcp.CallToolResult, error) {
	var inputParams map[string]string
	if err := json.Unmarshal(args.Params, &inputParams); err != nil {
		return mcp.CallToolResult{}, err
	}

	content := []mcp.Content{
		{
			Type: mcp.ContentTypeText,
			Text: "Echo: " + inputParams["message"],
		},
	}

	return mcp.CallToolResult{
		Content: content,
	}, nil
}

func errorToolHandler(context.Context, mcphive.HandleArgs) (mcp.CallToolResult, error) {
	return mcp.CallToolResult{}, errors.New("tool error")
}

func chainToolHandler(ctx context.Context, args mcphive.HandleArgs) (mcp.CallToolResult, error) {
	var inputParams map[string]string
	if err := json.Unmarshal(args.Params, &inputParams); err != nil {
		return mcp.CallToolResult{}, err
	}

	toolName := inputParams["tool"]
	message := inputParams["message"]

	msgJSON, _ := json.Marshal(map[string]string{"message": message})
	result, err := args.CallTool(ctx, mcp.CallToolParams{
		Name:      toolName,
		Arguments: msgJSON,
	})
	if err != nil {
		return mcp.CallToolResult{}, err
	}

	return result, nil
}

func llmToolHandler(ctx context.Context, args mcphive.HandleArgs) (mcp.CallToolResult, error) {
	var inputParams map[string]string
	if err := json.Unmarshal(args.Params, &inputParams); err != nil {
		return mcp.CallToolResult{}, err
	}

	messages := []mcphive.Message{
		{
			Role: mcphive.RoleUser,
			Contents: []mcphive.Content{
				{
					Type: mcphive.ContentTypeText,
					Text: inputParams["message"],
				},
			},
		},
	}

	var response string
	for content, err := range args.CallLLM(ctx, messages) {
		if err != nil {
			return mcp.CallToolResult{}, err
		}
		if content.Type == mcphive.ContentTypeText {
			response += content.Text
		}
	}

	mcpContent := []mcp.Content{
		{
			Type: mcp.ContentTypeText,
			Text: response,
		},
	}

	return mcp.CallToolResult{
		Content: mcpContent,
	}, nil
}

func (m *mockLLM) Call(
	_ context.Context,
	messages []mcphive.Message,
	tools []mcp.Tool,
) iter.Seq2[mcphive.Content, error] {
	return func(yield func(mcphive.Content, error) bool) {
		if m.err != nil {
			yield(mcphive.Content{}, m.err)
			return
		}

		// Verify tools visibility if requested
		if m.verifyTools && m.t != nil {
			// Check that excluded tools are not present
			for _, excluded := range m.excludedTools {
				for _, tool := range tools {
					if tool.Name == excluded {
						m.t.Errorf("LLM received excluded tool '%s' when it should have been filtered out", excluded)
					}
				}
			}

			// Check that expected tools are present
			for _, expected := range m.expectedTools {
				found := false
				for _, tool := range tools {
					if tool.Name == expected {
						found = true
						break
					}
				}
				if !found {
					m.t.Errorf("LLM did not receive expected tool '%s'", expected)
				}
			}
		}

		// Check if we already processed a tool call by examining the last message
		if len(messages) > 0 {
			lastMessage := messages[len(messages)-1]
			// If the last message has contents and the last content is a tool result,
			// don't issue the same tool call again
			if len(lastMessage.Contents) > 0 {
				lastContent := lastMessage.Contents[len(lastMessage.Contents)-1]
				if lastContent.Type == mcphive.ContentTypeToolResult {
					// Immediately return
					return
				}
			}
		}

		for _, resp := range m.responses {
			if !yield(resp, nil) {
				return
			}
		}
	}
}
