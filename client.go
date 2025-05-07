package mcphive

import (
	"bufio"
	"fmt"
	"log/slog"
	"os/exec"

	"github.com/MegaGrindStone/go-mcp"
)

// MCPClientConfig defines the interface for configuring MCP clients.
// Implementations must provide methods to create an MCP client and shutdown
// any resources associated with the client.
type MCPClientConfig interface {
	MCPClient(info mcp.Info, logger *slog.Logger) (*mcp.Client, error)
	Shutdown() error
}

// MCPSSEClient implements MCPClientConfig for creating MCP clients that
// communicate via Server-Sent Events (SSE).
// It configures an SSE client with the specified URL and payload size limits.
type MCPSSEClient struct {
	URL            string
	MaxPayloadSize int
}

// MCPStdIOClient implements MCPClientConfig for creating MCP clients that
// communicate with external processes via standard I/O streams.
// It executes the specified command and establishes stdin/stdout pipes for communication.
type MCPStdIOClient struct {
	Command string
	Args    []string

	cmd *exec.Cmd
}

// MCPClient creates a new MCP client that communicates via SSE.
// It initializes an SSE client with the configured URL and payload size,
// then wraps it in an MCP client with the provided info and logger.
// Returns the configured client or an error if initialization fails.
func (m MCPSSEClient) MCPClient(info mcp.Info, logger *slog.Logger) (*mcp.Client, error) {
	sseClient := mcp.NewSSEClient(m.URL, nil,
		mcp.WithSSEClientMaxPayloadSize(m.MaxPayloadSize),
		mcp.WithSSEClientLogger(logger))
	return mcp.NewClient(info, sseClient, mcp.WithClientLogger(logger)), nil
}

// Shutdown performs necessary cleanup for the SSE client.
// This implementation is a no-op as SSE clients don't require special shutdown.
// It's included to satisfy the MCPClientConfig interface.
func (m MCPSSEClient) Shutdown() error {
	return nil
}

// MCPClient creates a new MCP client that communicates via standard I/O.
// It executes the configured command, establishes pipes for stdin/stdout/stderr,
// and initializes an MCP client to communicate through these pipes.
// The stderr output from the command is logged as errors.
// Returns the configured client or an error if command execution fails.
func (m *MCPStdIOClient) MCPClient(info mcp.Info, logger *slog.Logger) (*mcp.Client, error) {
	m.cmd = exec.Command(m.Command, m.Args...)

	in, err := m.cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	out, err := m.cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	stderr, err := m.cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	if err := m.cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start command: %w", err)
	}

	// Listen for stderr output and log it
	go func() {
		errScanner := bufio.NewScanner(stderr)
		for errScanner.Scan() {
			logger.Error("StdIO error", slog.String("err", errScanner.Text()))
		}
	}()

	cliStdIO := mcp.NewStdIO(out, in, mcp.WithStdIOLogger(logger))

	return mcp.NewClient(info, cliStdIO, mcp.WithClientLogger(logger)), nil
}

// Shutdown terminates the external process and cleans up associated resources.
// It sends a kill signal to the command process and waits for it to exit.
// Returns an error if the process cannot be terminated properly.
func (m *MCPStdIOClient) Shutdown() error {
	if m.cmd != nil {
		if err := m.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill command: %w", err)
		}
		if err := m.cmd.Wait(); err != nil {
			return fmt.Errorf("failed to wait for command: %w", err)
		}
	}

	return nil
}
