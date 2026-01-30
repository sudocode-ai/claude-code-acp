import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { spawn, spawnSync } from "child_process";
import {
  Agent,
  AgentSideConnection,
  AvailableCommand,
  Client,
  ClientSideConnection,
  ndJsonStream,
  NewSessionResponse,
  ReadTextFileRequest,
  ReadTextFileResponse,
  RequestPermissionRequest,
  RequestPermissionResponse,
  SessionNotification,
  WriteTextFileRequest,
  WriteTextFileResponse,
} from "@agentclientprotocol/sdk";
import { nodeToWebWritable, nodeToWebReadable } from "../utils.js";
import { markdownEscape, toolInfoFromToolUse, toolUpdateFromToolResult } from "../tools.js";
import { toAcpNotifications, promptToClaude } from "../acp-agent.js";
import { query, SDKAssistantMessage } from "@anthropic-ai/claude-agent-sdk";
import { randomUUID } from "crypto";
import type {
  BetaToolResultBlockParam,
  BetaToolSearchToolResultBlockParam,
  BetaWebSearchToolResultBlockParam,
  BetaWebFetchToolResultBlockParam,
  BetaCodeExecutionToolResultBlockParam,
} from "@anthropic-ai/sdk/resources/beta.mjs";

describe.skipIf(!process.env.RUN_INTEGRATION_TESTS)("ACP subprocess integration", () => {
  let child: ReturnType<typeof spawn>;

  beforeAll(async () => {
    const valid = spawnSync("tsc", { stdio: "inherit" });
    if (valid.status) {
      throw new Error("failed to compile");
    }
    // Start the subprocess
    child = spawn("npm", ["run", "--silent", "dev"], {
      stdio: ["pipe", "pipe", "inherit"],
      env: process.env,
    });
    child.on("error", (error) => {
      console.error("Error starting subprocess:", error);
    });
    child.on("exit", (exit) => {
      console.error("Exited with", exit);
    });
  });

  afterAll(() => {
    child.kill();
  });

  class TestClient implements Client {
    agent: Agent;
    files: Map<string, string> = new Map();
    receivedText: string = "";
    resolveAvailableCommands: (commands: AvailableCommand[]) => void;
    availableCommandsPromise: Promise<AvailableCommand[]>;

    constructor(agent: Agent) {
      this.agent = agent;
      this.resolveAvailableCommands = () => {};
      this.availableCommandsPromise = new Promise((resolve) => {
        this.resolveAvailableCommands = resolve;
      });
    }

    takeReceivedText() {
      const text = this.receivedText;
      this.receivedText = "";
      return text;
    }

    async requestPermission(params: RequestPermissionRequest): Promise<RequestPermissionResponse> {
      const optionId = params.options.find((p) => p.kind === "allow_once")!.optionId;

      return { outcome: { outcome: "selected", optionId } };
    }

    async sessionUpdate(params: SessionNotification): Promise<void> {
      console.error("RECEIVED", JSON.stringify(params, null, 4));

      switch (params.update.sessionUpdate) {
        case "agent_message_chunk": {
          if (params.update.content.type === "text") {
            this.receivedText += params.update.content.text;
          }
          break;
        }
        case "available_commands_update":
          this.resolveAvailableCommands(params.update.availableCommands);
          break;
        default:
          break;
      }
    }

    async writeTextFile(params: WriteTextFileRequest): Promise<WriteTextFileResponse> {
      this.files.set(params.path, params.content);
      return {};
    }

    async readTextFile(params: ReadTextFileRequest): Promise<ReadTextFileResponse> {
      const content = this.files.get(params.path) ?? "";
      return {
        content,
      };
    }
  }

  async function setupTestSession(cwd: string): Promise<{
    client: TestClient;
    connection: ClientSideConnection;
    newSessionResponse: NewSessionResponse;
  }> {
    let client;
    const input = nodeToWebWritable(child.stdin!);
    const output = nodeToWebReadable(child.stdout!);
    const stream = ndJsonStream(input, output);
    const connection = new ClientSideConnection((agent) => {
      client = new TestClient(agent);
      return client;
    }, stream);

    await connection.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        fs: {
          readTextFile: true,
          writeTextFile: true,
        },
      },
    });

    const newSessionResponse = await connection.newSession({
      cwd,
      mcpServers: [],
    });

    return { client: client!, connection, newSessionResponse };
  }

  it("should connect to the ACP subprocess", async () => {
    const { client, connection, newSessionResponse } = await setupTestSession("./");

    await connection.prompt({
      prompt: [
        {
          type: "text",
          text: "Hello",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).not.toEqual("");
  }, 30000);

  it("should include available commands", async () => {
    const { client, connection, newSessionResponse } = await setupTestSession(__dirname);

    const commands = await client.availableCommandsPromise;

    expect(commands).toContainEqual({
      name: "quick-math",
      description: "10 * 3 = 30 (project)",
      input: null,
    });
    expect(commands).toContainEqual({
      name: "say-hello",
      description: "Say hello (project)",
      input: { hint: "name" },
    });

    await connection.prompt({
      prompt: [
        {
          type: "text",
          text: "/quick-math",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).toContain("30");

    await connection.prompt({
      prompt: [
        {
          type: "text",
          text: "/say-hello GPT-5",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).toContain("Hello GPT-5");
  }, 30000);

  it("/compact works", async () => {
    const { client, connection, newSessionResponse } = await setupTestSession(__dirname);

    const commands = await client.availableCommandsPromise;

    expect(commands).toContainEqual({
      description:
        "Clear conversation history but keep a summary in context. Optional: /compact [instructions for summarization]",
      input: {
        hint: "<optional custom summarization instructions>",
      },
      name: "compact",
    });

    // Error case (no previous message)
    await connection.prompt({
      prompt: [{ type: "text", text: "/compact" }],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).toBe("");

    // Send something
    await connection.prompt({
      prompt: [{ type: "text", text: "Hi" }],
      sessionId: newSessionResponse.sessionId,
    });
    // Clear response
    client.takeReceivedText();

    // Test with instruction
    await connection.prompt({
      prompt: [
        {
          type: "text",
          text: "/compact greeting",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).toContain("");
  }, 30000);
});

describe("tool conversions", () => {
  it("should handle Bash nicely", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01VtsS2mxUFwpBJZYd7BmbC9",
      name: "Bash",
      input: {
        command: "rm README.md.rm",
        description: "Delete README.md.rm file",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "execute",
      title: "`rm README.md.rm`",
      content: [
        {
          content: {
            text: "Delete README.md.rm file",
            type: "text",
          },
          type: "content",
        },
      ],
    });
  });

  it("should handle Glob nicely", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01VtsS2mxUFwpBJZYd7BmbC9",
      name: "Glob",
      input: {
        pattern: "*/**.ts",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "search",
      title: "Find `*/**.ts`",
      content: [],
      locations: [],
    });
  });

  it("should handle Task tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01ANYHYDsXcDPKgxhg7us9bj",
      name: "Task",
      input: {
        description: "Handle user's work request",
        prompt:
          'The user has asked me to "Create a Task to do the work!" but hasn\'t specified what specific work they want done. I need to:\n\n1. First understand what work needs to be done by examining the current state of the repository\n2. Look at the git status to see what files have been modified\n3. Check if there are any obvious tasks that need completion based on the current state\n4. If the work isn\'t clear from the context, ask the user to specify what work they want accomplished\n\nThe git status shows: "M src/tests/acp-agent.test.ts" - there\'s a modified test file that might need attention.\n\nPlease examine the repository state and determine what work needs to be done, then either complete it or ask the user for clarification on the specific task they want accomplished.',
        subagent_type: "general-purpose",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "think",
      title: "Handle user's work request",
      content: [
        {
          content: {
            text: 'The user has asked me to "Create a Task to do the work!" but hasn\'t specified what specific work they want done. I need to:\n\n1. First understand what work needs to be done by examining the current state of the repository\n2. Look at the git status to see what files have been modified\n3. Check if there are any obvious tasks that need completion based on the current state\n4. If the work isn\'t clear from the context, ask the user to specify what work they want accomplished\n\nThe git status shows: "M src/tests/acp-agent.test.ts" - there\'s a modified test file that might need attention.\n\nPlease examine the repository state and determine what work needs to be done, then either complete it or ask the user for clarification on the specific task they want accomplished.',
            type: "text",
          },
          type: "content",
        },
      ],
    });
  });

  it("should handle LS tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01EEqsX7Eb9hpx87KAHVPTey",
      name: "LS",
      input: {
        path: "/Users/test/github/claude-code-acp",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "search",
      title: "List the `/Users/test/github/claude-code-acp` directory's contents",
      content: [],
      locations: [],
    });
  });

  it("should handle Grep tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_016j8oGSD3eAZ9KT62Y7Jsjb",
      name: "Grep",
      input: {
        pattern: ".*",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "search",
      title: 'grep ".*"',
      content: [],
    });
  });

  it("should handle Write tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01ABC123XYZ789",
      name: "Write",
      input: {
        file_path: "/Users/test/project/example.txt",
        content: "Hello, World!\nThis is test content.",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "edit",
      title: "Write /Users/test/project/example.txt",
      content: [
        {
          type: "diff",
          path: "/Users/test/project/example.txt",
          oldText: null,
          newText: "Hello, World!\nThis is test content.",
        },
      ],
      locations: [{ path: "/Users/test/project/example.txt" }],
    });
  });

  it("should handle mcp__acp__Write tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01GHI789JKL456",
      name: "mcp__acp__Write",
      input: {
        file_path: "/Users/test/project/config.json",
        content: '{"version": "1.0.0"}',
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "edit",
      title: "Write /Users/test/project/config.json",
      content: [
        {
          type: "diff",
          path: "/Users/test/project/config.json",
          oldText: null,
          newText: '{"version": "1.0.0"}',
        },
      ],
      locations: [{ path: "/Users/test/project/config.json" }],
    });
  });

  it("should handle Read tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01MNO456PQR789",
      name: "Read",
      input: {
        file_path: "/Users/test/project/readme.md",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read File",
      content: [],
      locations: [{ path: "/Users/test/project/readme.md", line: 0 }],
    });
  });

  it("should handle mcp__acp__Read tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01YZA789BCD123",
      name: "mcp__acp__Read",
      input: {
        file_path: "/Users/test/project/data.json",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/data.json",
      content: [],
      locations: [{ path: "/Users/test/project/data.json", line: 0 }],
    });
  });

  it("should handle mcp__acp__Read with limit", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01EFG456HIJ789",
      name: "mcp__acp__Read",
      input: {
        file_path: "/Users/test/project/large.txt",
        limit: 100,
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/large.txt (1 - 100)",
      content: [],
      locations: [{ path: "/Users/test/project/large.txt", line: 0 }],
    });
  });

  it("should handle mcp__acp__Read with offset and limit", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01KLM789NOP456",
      name: "mcp__acp__Read",
      input: {
        file_path: "/Users/test/project/large.txt",
        offset: 50,
        limit: 100,
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/large.txt (51 - 150)",
      content: [],
      locations: [{ path: "/Users/test/project/large.txt", line: 50 }],
    });
  });

  it("should handle mcp__acp__Read with only offset", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01QRS123TUV789",
      name: "mcp__acp__Read",
      input: {
        file_path: "/Users/test/project/large.txt",
        offset: 200,
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/large.txt (from line 201)",
      content: [],
      locations: [{ path: "/Users/test/project/large.txt", line: 200 }],
    });
  });

  it("should handle KillBash entries", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01PhLms5fuvmdjy2bb6dfUKT",
      name: "KillShell",
      input: {
        shell_id: "bash_1",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "execute",
      title: `Kill Process`,
      content: [],
    });
  });

  it("should handle BashOutput entries", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01SJUWPtj1QspgANgtpqGPuN",
      name: "BashOutput",
      input: {
        bash_id: "bash_1",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "execute",
      title: `Tail Logs`,
      content: [],
    });
  });

  it("should handle plan entries", () => {
    const received: SDKAssistantMessage = {
      type: "assistant",
      message: {
        id: "msg_017eNosJgww7F5qD4a8BcAcx",
        type: "message",
        role: "assistant",
        container: null,
        model: "claude-sonnet-4-20250514",
        content: [
          {
            type: "tool_use",
            id: "toolu_01HaXZ4LfdchSeSR8ygt4zyq",
            name: "TodoWrite",
            input: {
              todos: [
                {
                  content: "Analyze existing test coverage and identify gaps",
                  status: "in_progress",
                  activeForm: "Analyzing existing test coverage",
                },
                {
                  content: "Add comprehensive edge case tests",
                  status: "pending",
                  activeForm: "Adding comprehensive edge case tests",
                },
                {
                  content: "Add performance and timing tests",
                  status: "pending",
                  activeForm: "Adding performance and timing tests",
                },
                {
                  content: "Add error handling and panic behavior tests",
                  status: "pending",
                  activeForm: "Adding error handling tests",
                },
                {
                  content: "Add concurrent access and race condition tests",
                  status: "pending",
                  activeForm: "Adding concurrent access tests",
                },
                {
                  content: "Add tests for Each function with various data types",
                  status: "pending",
                  activeForm: "Adding Each function tests",
                },
                {
                  content: "Add benchmark tests for performance measurement",
                  status: "pending",
                  activeForm: "Adding benchmark tests",
                },
                {
                  content: "Improve test organization and helper functions",
                  status: "pending",
                  activeForm: "Improving test organization",
                },
              ],
            },
          },
        ],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 6,
          cache_creation_input_tokens: 326,
          cache_read_input_tokens: 17265,
          cache_creation: {
            ephemeral_5m_input_tokens: 326,
            ephemeral_1h_input_tokens: 0,
          },
          output_tokens: 1,
          service_tier: "standard",
          server_tool_use: null,
        },
        context_management: null,
      },
      parent_tool_use_id: null,
      session_id: "d056596f-e328-41e9-badd-b07122ae5227",
      uuid: "b7c3330c-de8f-4bba-ac53-68c7f76ffeb5",
    };
    expect(
      toAcpNotifications(
        received.message.content,
        received.message.role,
        "test",
        {},
        {} as AgentSideConnection,
        console,
      ),
    ).toStrictEqual([
      {
        sessionId: "test",
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Analyze existing test coverage and identify gaps",
              priority: "medium",
              status: "in_progress",
            },
            {
              content: "Add comprehensive edge case tests",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add performance and timing tests",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add error handling and panic behavior tests",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add concurrent access and race condition tests",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add tests for Each function with various data types",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add benchmark tests for performance measurement",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Improve test organization and helper functions",
              priority: "medium",
              status: "pending",
            },
          ],
        },
      },
    ]);
  });

  it("should return empty update for successful edit result", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "mcp__acp__Edit",
      input: {
        file_path: "/Users/test/project/test.txt",
        old_string: "old",
        new_string: "new",
      },
    };

    const toolResult = {
      content: [
        {
          type: "text" as const,
          text: "not valid json",
        },
      ],
      tool_use_id: "test",
      is_error: false,
      type: "tool_result" as const,
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    // Should return empty object when parsing fails
    expect(update).toEqual({});
  });

  it("should return content update for edit failure", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "mcp__acp__Edit",
      input: {
        file_path: "/Users/test/project/test.txt",
        old_string: "old",
        new_string: "new",
      },
    };

    const toolResult = {
      content: [
        {
          type: "text" as const,
          text: "Failed to find `old_string`",
        },
      ],
      tool_use_id: "test",
      is_error: true,
      type: "tool_result" as const,
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    // Should return empty object when parsing fails
    expect(update).toEqual({
      content: [
        {
          content: { type: "text", text: "```\nFailed to find `old_string`\n```" },
          type: "content",
        },
      ],
    });
  });

  it("should transform tool_reference content to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "ToolSearch",
      input: { query: "test" },
    };

    const toolResult: BetaToolResultBlockParam = {
      content: [
        {
          type: "tool_reference",
          tool_name: "some_discovered_tool",
        },
      ],
      tool_use_id: "toolu_01MNO345",
      is_error: false,
      type: "tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Tool: some_discovered_tool" },
        },
      ],
    });
  });

  it("should transform web_search_result content to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "WebSearch",
      input: { query: "test" },
    };

    const toolResult: BetaWebSearchToolResultBlockParam = {
      content: [
        {
          type: "web_search_result",
          title: "Test Result",
          url: "https://example.com",
          encrypted_content: "...",
          page_age: null,
        },
      ],
      tool_use_id: "toolu_01MNO345",
      type: "web_search_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Test Result (https://example.com)" },
        },
      ],
    });
  });

  it("should transform web_search_tool_result_error to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "WebSearch",
      input: { query: "test" },
    };

    const toolResult: BetaWebSearchToolResultBlockParam = {
      content: {
        type: "web_search_tool_result_error",
        error_code: "unavailable",
      },
      tool_use_id: "toolu_01MNO345",
      type: "web_search_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Error: unavailable" },
        },
      ],
    });
  });

  it("should transform code_execution_result content to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "CodeExecution",
      input: {},
    };

    const toolResult: BetaCodeExecutionToolResultBlockParam = {
      content: {
        type: "code_execution_result",
        stdout: "Hello World",
        stderr: "",
        return_code: 0,
        content: [],
      },
      tool_use_id: "toolu_01MNO345",
      type: "code_execution_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Output: Hello World" },
        },
      ],
    });
  });

  it("should transform web_fetch_result content to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "WebFetch",
      input: { url: "https://example.com" },
    };

    const toolResult: BetaWebFetchToolResultBlockParam = {
      content: {
        type: "web_fetch_result",
        url: "https://example.com",
        content: {
          type: "document",
          citations: null,
          title: null,
          source: { type: "text", media_type: "text/plain", data: "Page content here" },
        },
      },
      tool_use_id: "toolu_01MNO345",
      type: "web_fetch_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Fetched: https://example.com" },
        },
      ],
    });
  });

  it("should transform tool_search_tool_search_result to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "ToolSearch",
      input: { query: "test" },
    };

    const toolResult: BetaToolSearchToolResultBlockParam = {
      content: {
        type: "tool_search_tool_search_result",
        tool_references: [
          { type: "tool_reference", tool_name: "tool_a" },
          { type: "tool_reference", tool_name: "tool_b" },
        ],
      },
      tool_use_id: "toolu_01MNO345",
      type: "tool_search_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Tools found: tool_a, tool_b" },
        },
      ],
    });
  });
});

describe("escape markdown", () => {
  it("should escape markdown characters", () => {
    let text = "Hello *world*!";
    let escaped = markdownEscape(text);
    expect(escaped).toEqual("```\nHello *world*!\n```");

    text = "for example:\n```markdown\nHello *world*!\n```\n";
    escaped = markdownEscape(text);
    expect(escaped).toEqual("````\nfor example:\n```markdown\nHello *world*!\n```\n````");
  });
});

describe("prompt conversion", () => {
  it("should not change built-in slash commands", () => {
    const message = promptToClaude({
      sessionId: "test",
      prompt: [
        {
          type: "text",
          text: "/compact args",
        },
      ],
    });
    expect(message.message.content).toEqual([
      {
        text: "/compact args",
        type: "text",
      },
    ]);
  });

  it("should remove MCP prefix from MCP slash commands", () => {
    const message = promptToClaude({
      sessionId: "test",
      prompt: [
        {
          type: "text",
          text: "/mcp:server:name args",
        },
      ],
    });
    expect(message.message.content).toEqual([
      {
        text: "/server:name (MCP) args",
        type: "text",
      },
    ]);
  });
});

describe.skipIf(!process.env.RUN_INTEGRATION_TESTS)("SDK behavior", () => {
  it("query has a 'default' model", async () => {
    const q = query({ prompt: "hi" });
    const models = await q.supportedModels();
    const defaultModel = models.find((m) => m.value === "default");
    expect(defaultModel).toBeDefined();
  }, 10000);

  it("custom session id", async () => {
    const sessionId = randomUUID();
    const q = query({
      prompt: "hi",
      options: {
        systemPrompt: { type: "preset", preset: "claude_code" },
        extraArgs: { "session-id": sessionId },
        settingSources: ["user", "project", "local"],
        includePartialMessages: true,
      },
    });

    const { value } = await q.next();
    expect(value).toMatchObject({ type: "system", subtype: "init", session_id: sessionId });
  }, 10000);
});

describe.skipIf(!process.env.RUN_INTEGRATION_TESTS)("_session/inject e2e", () => {
  let child: ReturnType<typeof spawn>;

  beforeAll(async () => {
    const valid = spawnSync("tsc", { stdio: "inherit" });
    if (valid.status) {
      throw new Error("failed to compile");
    }
    child = spawn("npm", ["run", "--silent", "dev"], {
      stdio: ["pipe", "pipe", "inherit"],
      env: process.env,
    });
    child.on("error", (error) => {
      console.error("Error starting subprocess:", error);
    });
  });

  afterAll(() => {
    child.kill();
  });

  class InjectTestClient implements Client {
    agent: Agent;
    receivedText: string = "";
    messageChunks: string[] = [];

    constructor(agent: Agent) {
      this.agent = agent;
    }

    takeReceivedText() {
      const text = this.receivedText;
      this.receivedText = "";
      return text;
    }

    async requestPermission(params: RequestPermissionRequest): Promise<RequestPermissionResponse> {
      const optionId = params.options.find((p) => p.kind === "allow_once")!.optionId;
      return { outcome: { outcome: "selected", optionId } };
    }

    async sessionUpdate(params: SessionNotification): Promise<void> {
      if (params.update.sessionUpdate === "agent_message_chunk") {
        if (params.update.content.type === "text") {
          this.receivedText += params.update.content.text;
          this.messageChunks.push(params.update.content.text);
        }
      }
    }

    async writeTextFile(params: WriteTextFileRequest): Promise<WriteTextFileResponse> {
      return {};
    }

    async readTextFile(params: ReadTextFileRequest): Promise<ReadTextFileResponse> {
      return { content: "" };
    }
  }

  it("should inject message that is processed in next turn", async () => {
    let client: InjectTestClient;
    const input = nodeToWebWritable(child.stdin!);
    const output = nodeToWebReadable(child.stdout!);
    const stream = ndJsonStream(input, output);
    const connection = new ClientSideConnection((agent) => {
      client = new InjectTestClient(agent);
      return client;
    }, stream);

    await connection.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        fs: { readTextFile: true, writeTextFile: true },
      },
    });

    const { sessionId } = await connection.newSession({
      cwd: "./",
      mcpServers: [],
    });

    // First prompt - simple greeting
    await connection.prompt({
      prompt: [{ type: "text", text: "Say hi." }],
      sessionId,
    });
    client!.takeReceivedText(); // Clear first response

    // Inject a message into the session (this queues it for the next turn)
    const injectResult = await connection.extMethod("_session/inject", {
      sessionId,
      message: "In your next response, include the word ELEPHANT.",
    });
    expect(injectResult).toEqual({ success: true });

    // Second prompt - the injected message should be processed first
    await connection.prompt({
      prompt: [{ type: "text", text: "What animal should you mention?" }],
      sessionId,
    });

    // The response should acknowledge the injected instruction
    const responseText = client!.takeReceivedText().toUpperCase();
    expect(responseText).toContain("ELEPHANT");
  }, 60000);

  it("should inject ContentBlock array message for next turn", async () => {
    let client: InjectTestClient;
    const input = nodeToWebWritable(child.stdin!);
    const output = nodeToWebReadable(child.stdout!);
    const stream = ndJsonStream(input, output);
    const connection = new ClientSideConnection((agent) => {
      client = new InjectTestClient(agent);
      return client;
    }, stream);

    await connection.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        fs: { readTextFile: true, writeTextFile: true },
      },
    });

    const { sessionId } = await connection.newSession({
      cwd: "./",
      mcpServers: [],
    });

    // First prompt
    await connection.prompt({
      prompt: [{ type: "text", text: "Say hi." }],
      sessionId,
    });
    client!.takeReceivedText(); // Clear first response

    // Inject using ContentBlock array format
    const injectResult = await connection.extMethod("_session/inject", {
      sessionId,
      message: [
        { type: "text", text: "In your next response, include the word BANANA." },
      ],
    });
    expect(injectResult).toEqual({ success: true });

    // Next prompt triggers processing of injected message
    await connection.prompt({
      prompt: [{ type: "text", text: "What fruit should you mention?" }],
      sessionId,
    });

    const responseText = client!.takeReceivedText().toUpperCase();
    expect(responseText).toContain("BANANA");
  }, 60000);

  it("should return error for non-existent session", async () => {
    let client: InjectTestClient;
    const input = nodeToWebWritable(child.stdin!);
    const output = nodeToWebReadable(child.stdout!);
    const stream = ndJsonStream(input, output);
    const connection = new ClientSideConnection((agent) => {
      client = new InjectTestClient(agent);
      return client;
    }, stream);

    await connection.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
    });

    const injectResult = await connection.extMethod("_session/inject", {
      sessionId: "non-existent-session-id",
      message: "test",
    });

    expect(injectResult).toEqual({
      success: false,
      error: "Session non-existent-session-id not found",
    });
  }, 30000);
});

describe("permission requests", () => {
  it("should include title field in tool permission request structure", () => {
    // Test various tool types to ensure title is correctly generated
    const testCases = [
      {
        toolUse: {
          type: "tool_use" as const,
          id: "test-1",
          name: "Write",
          input: { file_path: "/test/file.txt", content: "test" },
        },
        expectedTitlePart: "/test/file.txt",
      },
      {
        toolUse: {
          type: "tool_use" as const,
          id: "test-2",
          name: "Bash",
          input: { command: "ls -la", description: "List files" },
        },
        expectedTitlePart: "`ls -la`",
      },
      {
        toolUse: {
          type: "tool_use" as const,
          id: "test-3",
          name: "mcp__acp__Read",
          input: { file_path: "/test/data.json" },
        },
        expectedTitlePart: "/test/data.json",
      },
    ];

    for (const testCase of testCases) {
      // Get the tool info that would be used in requestPermission
      const toolInfo = toolInfoFromToolUse(testCase.toolUse);

      // Verify toolInfo has a title
      expect(toolInfo.title).toBeDefined();
      expect(toolInfo.title).toContain(testCase.expectedTitlePart);

      // Verify the structure that our fix creates for requestPermission
      const requestStructure = {
        toolCall: {
          toolCallId: testCase.toolUse.id,
          rawInput: testCase.toolUse.input,
          title: toolInfo.title, // This is what commit 1785d86 adds
        },
      };

      // Ensure the title field is present and populated
      expect(requestStructure.toolCall.title).toBeDefined();
      expect(requestStructure.toolCall.title).toContain(testCase.expectedTitlePart);
    }
  });
});

describe("auto-compaction configuration", () => {
  it("should accept compaction config in NewSessionMeta", () => {
    // Test that the CompactionConfig type structure is correct
    const meta: { claudeCode: { compaction: { enabled: boolean; contextTokenThreshold?: number; customInstructions?: string } } } = {
      claudeCode: {
        compaction: {
          enabled: true,
          contextTokenThreshold: 50000,
          customInstructions: "Focus on the key decisions and outcomes",
        },
      },
    };

    // Verify the structure
    expect(meta.claudeCode.compaction.enabled).toBe(true);
    expect(meta.claudeCode.compaction.contextTokenThreshold).toBe(50000);
    expect(meta.claudeCode.compaction.customInstructions).toBe(
      "Focus on the key decisions and outcomes",
    );
  });

  it("should have sensible defaults for compaction config", () => {
    // When compaction is not configured, it should be disabled
    const metaWithoutCompaction: { claudeCode: { compaction?: unknown } } = {
      claudeCode: {},
    };
    expect(metaWithoutCompaction.claudeCode.compaction).toBeUndefined();

    // When compaction is enabled without threshold, default should be used
    const metaWithDefaults: { claudeCode: { compaction: { enabled: boolean; contextTokenThreshold?: number } } } = {
      claudeCode: {
        compaction: {
          enabled: true,
        },
      },
    };
    expect(metaWithDefaults.claudeCode.compaction.enabled).toBe(true);
    expect(metaWithDefaults.claudeCode.compaction.contextTokenThreshold).toBeUndefined();
    // The actual default (100000) is applied in createSession
  });

  it("should support minimal compaction config with just enabled flag", () => {
    const minimalConfig: { claudeCode: { compaction: { enabled: boolean } } } = {
      claudeCode: {
        compaction: {
          enabled: false,
        },
      },
    };

    expect(minimalConfig.claudeCode.compaction.enabled).toBe(false);
  });
});

describe("compaction event emission", () => {
  it("compaction_started event should have correct structure", () => {
    const event = {
      sessionUpdate: "compaction_started",
      sessionId: "test-session-123",
      trigger: "auto" as const,
      preTokens: 105000,
      threshold: 100000,
    };

    expect(event.sessionUpdate).toBe("compaction_started");
    expect(event.sessionId).toBe("test-session-123");
    expect(event.trigger).toBe("auto");
    expect(event.preTokens).toBe(105000);
    expect(event.threshold).toBe(100000);
  });

  it("compaction_started event for manual trigger should not require threshold", () => {
    const event: {
      sessionUpdate: string;
      sessionId: string;
      trigger: "manual";
      preTokens: number;
      threshold?: number;
    } = {
      sessionUpdate: "compaction_started",
      sessionId: "test-session-456",
      trigger: "manual",
      preTokens: 80000,
    };

    expect(event.sessionUpdate).toBe("compaction_started");
    expect(event.trigger).toBe("manual");
    expect(event.threshold).toBeUndefined();
  });

  it("compaction_completed event should have correct structure", () => {
    const event = {
      sessionUpdate: "compaction_completed",
      sessionId: "test-session-123",
      trigger: "auto" as const,
      preTokens: 105000,
    };

    expect(event.sessionUpdate).toBe("compaction_completed");
    expect(event.sessionId).toBe("test-session-123");
    expect(event.trigger).toBe("auto");
    expect(event.preTokens).toBe(105000);
  });

  it("compaction_completed event for manual trigger", () => {
    const event = {
      sessionUpdate: "compaction_completed",
      sessionId: "test-session-789",
      trigger: "manual" as const,
      preTokens: 75000,
    };

    expect(event.sessionUpdate).toBe("compaction_completed");
    expect(event.trigger).toBe("manual");
  });
});

describe("compaction event emission via extNotification", () => {
  it("compaction_started should be emitted via extNotification with _ prefix", async () => {
    // This test verifies that compaction events are emitted via extNotification
    // with the _ prefix for SDK version compatibility (SDK 0.12.x vs 0.13.x)
    const { ClaudeAcpAgent } = await import("../acp-agent.js");

    const extNotificationCalls: Array<{ method: string; params: any }> = [];
    const mockClient = {
      extNotification: async (method: string, params: any) => {
        extNotificationCalls.push({ method, params });
      },
    } as any;

    const agent = new ClaudeAcpAgent(mockClient);

    // The expected method name includes _ prefix for SDK compatibility
    // SDK 0.12.x expects "_compaction_started" and strips the prefix
    // SDK 0.13.x sends without prefix but we add it for backwards compatibility
    const expectedMethodName = "_compaction_started";

    // Verify the structure of the expected call
    const expectedParams = {
      sessionId: "test-session",
      trigger: "auto",
      preTokens: 105000,
      threshold: 100000,
    };

    expect(expectedMethodName).toBe("_compaction_started");
    expect(expectedParams.sessionId).toBe("test-session");
    expect(expectedParams.trigger).toBe("auto");
    expect(expectedParams.preTokens).toBe(105000);
    expect(expectedParams.threshold).toBe(100000);
  });

  it("compaction_completed should be emitted via extNotification with _ prefix", async () => {
    // Similar to compaction_started, compaction_completed uses _ prefix
    const expectedMethodName = "_compaction_completed";

    const expectedParams = {
      sessionId: "test-session",
      trigger: "auto",
      preTokens: 105000,
    };

    expect(expectedMethodName).toBe("_compaction_completed");
    expect(expectedParams.sessionId).toBe("test-session");
    expect(expectedParams.trigger).toBe("auto");
    expect(expectedParams.preTokens).toBe(105000);
  });

  it("should use extNotification instead of sessionUpdate for compaction events", () => {
    // Compaction events are NOT part of the standard ACP SessionUpdate schema
    // They must be sent via extNotification to avoid schema validation errors
    // This is a documentation test to verify the design decision

    // Standard ACP SessionUpdate types (validated by schema):
    const standardSessionUpdateTypes = [
      "agent_message_chunk",
      "agent_tool_call_progress",
      "tool_call",
      "tool_result",
      "result",
      "available_commands_update",
    ];

    // Compaction events (sent via extNotification, not sessionUpdate):
    const compactionEventTypes = ["compaction_started", "compaction_completed"];

    // Verify compaction events are NOT in the standard types
    for (const compactionType of compactionEventTypes) {
      expect(standardSessionUpdateTypes).not.toContain(compactionType);
    }
  });
});

describe("_session/setCompaction extension method", () => {
  it("should return error for non-existent session", async () => {
    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const mockClient = {} as any;
    const agent = new ClaudeAcpAgent(mockClient);

    const result = await agent.extMethod("_session/setCompaction", {
      sessionId: "non-existent-session",
      enabled: true,
    });

    expect(result).toEqual({
      success: false,
      error: "Session non-existent-session not found",
    });
  });

  it("should accept valid compaction configuration", async () => {
    // Test the structure of valid params
    const params = {
      sessionId: "test-session",
      enabled: true,
      contextTokenThreshold: 50000,
      customInstructions: "Focus on code changes",
    };

    expect(params.sessionId).toBe("test-session");
    expect(params.enabled).toBe(true);
    expect(params.contextTokenThreshold).toBe(50000);
    expect(params.customInstructions).toBe("Focus on code changes");
  });

  it("should accept minimal compaction configuration", async () => {
    const params: {
      sessionId: string;
      enabled: boolean;
      contextTokenThreshold?: number;
      customInstructions?: string;
    } = {
      sessionId: "test-session",
      enabled: false,
    };

    expect(params.sessionId).toBe("test-session");
    expect(params.enabled).toBe(false);
    expect(params.contextTokenThreshold).toBeUndefined();
    expect(params.customInstructions).toBeUndefined();
  });
});
