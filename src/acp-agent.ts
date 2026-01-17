import {
  Agent,
  AgentSideConnection,
  AuthenticateRequest,
  AvailableCommand,
  CancelNotification,
  ClientCapabilities,
  ForkSessionRequest,
  ForkSessionResponse,
  InitializeRequest,
  InitializeResponse,
  LoadSessionRequest,
  LoadSessionResponse,
  ndJsonStream,
  NewSessionRequest,
  NewSessionResponse,
  PromptRequest,
  PromptResponse,
  ReadTextFileRequest,
  ReadTextFileResponse,
  RequestError,
  SessionModelState,
  SessionNotification,
  SetSessionModelRequest,
  SetSessionModelResponse,
  SetSessionModeRequest,
  SetSessionModeResponse,
  TerminalHandle,
  TerminalOutputResponse,
  WriteTextFileRequest,
  WriteTextFileResponse,
} from "@agentclientprotocol/sdk";
import { SettingsManager } from "./settings.js";
import {
  CanUseTool,
  McpServerConfig,
  Options,
  PermissionMode,
  Query,
  query,
  SDKPartialAssistantMessage,
  SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { nodeToWebReadable, nodeToWebWritable, Pushable, unreachable } from "./utils.js";
import { createMcpServer } from "./mcp-server.js";
import { EDIT_TOOL_NAMES, acpToolNames } from "./tools.js";
import {
  toolInfoFromToolUse,
  planEntries,
  toolUpdateFromToolResult,
  ClaudePlanEntry,
  registerHookCallback,
  createPostToolUseHook,
  createPreToolUseHook,
} from "./tools.js";
import { ContentBlockParam } from "@anthropic-ai/sdk/resources";
import { BetaContentBlock, BetaRawContentBlockDelta } from "@anthropic-ai/sdk/resources/beta.mjs";
import packageJson from "../package.json" with { type: "json" };
import { randomUUID } from "node:crypto";

export const CLAUDE_CONFIG_DIR = process.env.CLAUDE ?? path.join(os.homedir(), ".claude");

/**
 * Logger interface for customizing logging output
 */
export interface Logger {
  log: (...args: any[]) => void;
  error: (...args: any[]) => void;
}

type Session = {
  query: Query;
  input: Pushable<SDKUserMessage>;
  cancelled: boolean;
  permissionMode: PermissionMode;
  settingsManager: SettingsManager;
  abortController: AbortController;
  cwd: string;
  /** Optional: the actual session file path (for forked sessions where filename differs from sessionId) */
  sessionFilePath?: string;
};

type BackgroundTerminal =
  | {
      handle: TerminalHandle;
      status: "started";
      lastOutput: TerminalOutputResponse | null;
    }
  | {
      status: "aborted" | "exited" | "killed" | "timedOut";
      pendingOutput: TerminalOutputResponse;
    };

/**
 * Extra metadata that can be given to Claude Code when creating a new session.
 */
export type NewSessionMeta = {
  claudeCode?: {
    /**
     * Options forwarded to Claude Code when starting a new session.
     * Those parameters will be ignored and managed by ACP:
     *   - cwd
     *   - includePartialMessages
     *   - allowDangerouslySkipPermissions
     *   - permissionMode
     *   - canUseTool
     *   - executable
     * Those parameters will be used and updated to work with ACP:
     *   - hooks (merged with ACP's hooks)
     *   - mcpServers (merged with ACP's mcpServers)
     */
    options?: Options;
  };
};

/**
 * Extra metadata that the agent provides for each tool_call / tool_update update.
 */
export type ToolUpdateMeta = {
  claudeCode?: {
    /* The name of the tool that was used in Claude Code. */
    toolName: string;
    /* The structured output provided by Claude Code. */
    toolResponse?: unknown;
  };
};

type ToolUseCache = {
  [key: string]: {
    type: "tool_use" | "server_tool_use" | "mcp_tool_use";
    id: string;
    name: string;
    input: any;
  };
};

// Bypass Permissions doesn't work if we are a root/sudo user
const IS_ROOT = (process.geteuid?.() ?? process.getuid?.()) === 0;

// Implement the ACP Agent interface
export class ClaudeAcpAgent implements Agent {
  sessions: {
    [key: string]: Session;
  };
  client: AgentSideConnection;
  toolUseCache: ToolUseCache;
  backgroundTerminals: { [key: string]: BackgroundTerminal } = {};
  clientCapabilities?: ClientCapabilities;
  logger: Logger;

  constructor(client: AgentSideConnection, logger?: Logger) {
    this.sessions = {};
    this.client = client;
    this.toolUseCache = {};
    this.logger = logger ?? console;
  }

  async initialize(request: InitializeRequest): Promise<InitializeResponse> {
    this.clientCapabilities = request.clientCapabilities;

    // Default authMethod
    const authMethod: any = {
      description: "Run `claude /login` in the terminal",
      name: "Log in with Claude Code",
      id: "claude-login",
    };

    // If client supports terminal-auth capability, use that instead.
    // if (request.clientCapabilities?._meta?.["terminal-auth"] === true) {
    //   const cliPath = fileURLToPath(import.meta.resolve("@anthropic-ai/claude-agent-sdk/cli.js"));

    //   authMethod._meta = {
    //     "terminal-auth": {
    //       command: "node",
    //       args: [cliPath, "/login"],
    //       label: "Claude Code Login",
    //     },
    //   };
    // }

    return {
      protocolVersion: 1,
      agentCapabilities: {
        promptCapabilities: {
          image: true,
          embeddedContext: true,
        },
        mcpCapabilities: {
          http: true,
          sse: true,
        },
        sessionCapabilities: {
          fork: {},
          resume: {},
        },
        loadSession: true,
      },
      agentInfo: {
        name: packageJson.name,
        title: "Claude Code",
        version: packageJson.version,
      },
      authMethods: [authMethod],
    };
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    if (
      fs.existsSync(path.resolve(os.homedir(), ".claude.json.backup")) &&
      !fs.existsSync(path.resolve(os.homedir(), ".claude.json"))
    ) {
      throw RequestError.authRequired();
    }

    return await this.createSession(params, {
      // Revisit these meta values once we support resume
      resume: (params._meta as NewSessionMeta | undefined)?.claudeCode?.options?.resume,
    });
  }

  /**
   * Fork an existing session to create a new independent session.
   * This is the ACP protocol method handler for session/fork.
   * Named unstable_forkSession to match SDK expectations (session/fork routes to this method).
   */
  async unstable_forkSession(params: ForkSessionRequest): Promise<ForkSessionResponse> {
    // Get the session directory to track new files
    const sessionDir = this.getSessionDirPath(params.cwd);
    const beforeFiles = new Set(
      fs.existsSync(sessionDir)
        ? fs.readdirSync(sessionDir).filter((f) => f.endsWith(".jsonl"))
        : [],
    );

    const result = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
        forkSession: true,
      },
    );

    // Wait briefly for CLI to create the session file
    await new Promise((resolve) => setTimeout(resolve, 200));

    // Find the CLI-assigned session ID by looking for new session files
    const cliSessionId = await this.discoverCliSessionId(sessionDir, beforeFiles, result.sessionId);

    if (cliSessionId && cliSessionId !== result.sessionId) {
      // Check if the CLI assigned a non-UUID session ID (e.g., "agent-xxx")
      // If so, we need to extract the internal sessionId from the file
      const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
        cliSessionId,
      );

      if (!isUuid) {
        // Read the session file to extract the internal sessionId
        const oldFilePath = path.join(sessionDir, `${cliSessionId}.jsonl`);
        const internalSessionId = this.extractInternalSessionId(oldFilePath);

        if (internalSessionId) {
          this.logger.log(
            `[claude-code-acp] Fork: extracted internal sessionId ${internalSessionId} from ${cliSessionId}`,
          );

          // Check if target file already exists (CLI reuses session IDs for forks from same parent)
          // If so, generate a new unique session ID to avoid collisions
          let finalSessionId = internalSessionId;
          let newFilePath = path.join(sessionDir, `${finalSessionId}.jsonl`);

          if (fs.existsSync(newFilePath)) {
            // Session ID collision - CLI created a fork with the same internal ID
            // Generate a new UUID and update the file's internal session ID
            finalSessionId = randomUUID();
            newFilePath = path.join(sessionDir, `${finalSessionId}.jsonl`);
            this.logger.log(
              `[claude-code-acp] Fork: session ID collision detected, using new ID: ${finalSessionId}`,
            );

            // Update the internal session ID in the file before renaming
            this.updateSessionIdInFile(oldFilePath, finalSessionId);
          }

          // Rename the file to match the session ID so CLI can find it
          try {
            fs.renameSync(oldFilePath, newFilePath);
            this.logger.log(
              `[claude-code-acp] Fork: renamed ${cliSessionId}.jsonl -> ${finalSessionId}.jsonl`,
            );

            // Promote sidechain to full session so it can be resumed/forked again
            this.promoteToFullSession(newFilePath);
          } catch (err) {
            this.logger.error(`[claude-code-acp] Failed to rename session file: ${err}`);
            // Continue anyway - the session might still work
          }

          // Re-register session with the final session ID
          const session = this.sessions[result.sessionId];
          this.sessions[finalSessionId] = session;
          delete this.sessions[result.sessionId];
          return { ...result, sessionId: finalSessionId };
        }

        // Fall through if we couldn't extract the internal ID
        this.logger.error(
          `[claude-code-acp] Could not extract internal sessionId from ${oldFilePath}`,
        );
      }

      // Re-register session with the CLI's session ID (if it's already a UUID or extraction failed)
      this.logger.log(
        `[claude-code-acp] Fork: remapping session ${result.sessionId} -> ${cliSessionId}`,
      );
      this.sessions[cliSessionId] = this.sessions[result.sessionId];
      delete this.sessions[result.sessionId];
      return { ...result, sessionId: cliSessionId };
    }

    return result;
  }

  /**
   * Get the directory where session files are stored for a given cwd.
   */
  private getSessionDirPath(cwd: string): string {
    const realCwd = fs.realpathSync(cwd);
    const cwdHash = realCwd.replace(/[/_]/g, "-");
    return path.join(os.homedir(), ".claude", "projects", cwdHash);
  }

  /**
   * Extract the internal sessionId from a session JSONL file.
   * The CLI stores the actual session ID inside the file, which may differ from the filename.
   * For forked sessions, the filename is "agent-xxx" but the internal sessionId is a UUID.
   */
  private extractInternalSessionId(filePath: string): string | null {
    try {
      if (!fs.existsSync(filePath)) {
        return null;
      }

      const content = fs.readFileSync(filePath, "utf-8");
      const firstLine = content.split("\n").find((line) => line.trim().length > 0);

      if (!firstLine) {
        return null;
      }

      const parsed = JSON.parse(firstLine);
      if (parsed.sessionId && typeof parsed.sessionId === "string") {
        // Verify it's a UUID format
        const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
          parsed.sessionId,
        );
        if (isUuid) {
          return parsed.sessionId;
        }
      }

      return null;
    } catch (err) {
      this.logger.error(`[claude-code-acp] Failed to extract sessionId from ${filePath}: ${err}`);
      return null;
    }
  }

  /**
   * Promote a sidechain session to a regular session by modifying the session file.
   * Forked sessions have "isSidechain": true which prevents them from being resumed.
   * This method changes it to false so the session can be resumed/forked again.
   */
  private promoteToFullSession(filePath: string): boolean {
    try {
      if (!fs.existsSync(filePath)) {
        return false;
      }

      const content = fs.readFileSync(filePath, "utf-8");
      const lines = content.split("\n");
      const modifiedLines: string[] = [];

      for (const line of lines) {
        if (!line.trim()) {
          modifiedLines.push(line);
          continue;
        }

        try {
          const parsed = JSON.parse(line);
          // Change isSidechain from true to false
          if (parsed.isSidechain === true) {
            parsed.isSidechain = false;
          }
          modifiedLines.push(JSON.stringify(parsed));
        } catch {
          // Keep line as-is if it can't be parsed
          modifiedLines.push(line);
        }
      }

      fs.writeFileSync(filePath, modifiedLines.join("\n"), "utf-8");
      this.logger.log(`[claude-code-acp] Promoted sidechain to full session: ${filePath}`);
      return true;
    } catch (err) {
      this.logger.error(`[claude-code-acp] Failed to promote session: ${err}`);
      return false;
    }
  }

  /**
   * Update the sessionId in all lines of a session JSONL file.
   * This is used when we need to assign a new unique session ID to avoid collisions.
   */
  private updateSessionIdInFile(filePath: string, newSessionId: string): boolean {
    try {
      if (!fs.existsSync(filePath)) {
        return false;
      }

      const content = fs.readFileSync(filePath, "utf-8");
      const lines = content.split("\n");
      const modifiedLines: string[] = [];

      for (const line of lines) {
        if (!line.trim()) {
          modifiedLines.push(line);
          continue;
        }

        try {
          const parsed = JSON.parse(line);
          // Update the sessionId in each line
          if (parsed.sessionId && typeof parsed.sessionId === "string") {
            parsed.sessionId = newSessionId;
          }
          modifiedLines.push(JSON.stringify(parsed));
        } catch {
          // Keep line as-is if it can't be parsed
          modifiedLines.push(line);
        }
      }

      fs.writeFileSync(filePath, modifiedLines.join("\n"), "utf-8");
      this.logger.log(`[claude-code-acp] Updated session ID in file: ${filePath}`);
      return true;
    } catch (err) {
      this.logger.error(`[claude-code-acp] Failed to update session ID in file: ${err}`);
      return false;
    }
  }

  /**
   * Discover the CLI-assigned session ID by looking for new session files.
   * Returns the CLI's session ID if found, or the original sessionId if not.
   */
  private async discoverCliSessionId(
    sessionDir: string,
    beforeFiles: Set<string>,
    fallbackId: string,
    timeout: number = 2000,
  ): Promise<string> {
    const start = Date.now();
    // Pattern for CLI-assigned fork session IDs (agent-xxxxxxx)
    const agentPattern = /^agent-[a-f0-9]+\.jsonl$/;

    while (Date.now() - start < timeout) {
      if (fs.existsSync(sessionDir)) {
        const currentFiles = fs.readdirSync(sessionDir).filter((f) => f.endsWith(".jsonl"));
        // Only look for new files that match the agent-xxx pattern
        // This prevents picking up renamed UUID files from previous forks
        const newFiles = currentFiles.filter((f) => !beforeFiles.has(f) && agentPattern.test(f));

        if (newFiles.length === 1) {
          // Found exactly one new agent session file - this is our fork
          this.logger.log(`[claude-code-acp] Discovered fork session file: ${newFiles[0]}`);
          return newFiles[0].replace(".jsonl", "");
        } else if (newFiles.length > 1) {
          // Multiple new agent files - try to find the most recent one
          let newestFile = "";
          let newestMtime = 0;
          for (const file of newFiles) {
            const filePath = path.join(sessionDir, file);
            const stat = fs.statSync(filePath);
            if (stat.mtimeMs > newestMtime) {
              newestMtime = stat.mtimeMs;
              newestFile = file;
            }
          }
          if (newestFile) {
            this.logger.log(
              `[claude-code-acp] Discovered fork session file (newest of ${newFiles.length}): ${newestFile}`,
            );
            return newestFile.replace(".jsonl", "");
          }
        }
      }

      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    // Timeout - return fallback
    this.logger.log(
      `[claude-code-acp] Could not discover CLI session ID, using fallback: ${fallbackId}`,
    );
    return fallbackId;
  }

  /**
   * Alias for unstable_forkSession for convenience.
   */
  async forkSession(params: ForkSessionRequest): Promise<ForkSessionResponse> {
    return this.unstable_forkSession(params);
  }

  /**
   * Load an existing session to resume a previous conversation.
   * This is the ACP protocol method handler for session/load.
   */
  async loadSession(params: LoadSessionRequest): Promise<LoadSessionResponse> {
    const response = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
      },
    );

    return response;
  }

  /**
   * @deprecated Use loadSession instead. This is kept for backward compatibility.
   */
  async unstable_resumeSession(params: {
    sessionId: string;
    _meta?: { cwd?: string; mcpServers?: any[]; [key: string]: unknown } | null;
  }): Promise<LoadSessionResponse> {
    const meta = params._meta as { cwd?: string; mcpServers?: any[] } | undefined;
    return this.loadSession({
      sessionId: params.sessionId,
      cwd: meta?.cwd ?? process.cwd(),
      mcpServers: meta?.mcpServers ?? [],
      _meta: params._meta,
    });
  }

  async authenticate(_params: AuthenticateRequest): Promise<void> {
    throw new Error("Method not implemented.");
  }

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }

    this.sessions[params.sessionId].cancelled = false;

    const { query, input } = this.sessions[params.sessionId];

    input.push(promptToClaude(params));
    while (true) {
      const { value: message, done } = await query.next();
      if (done || !message) {
        if (this.sessions[params.sessionId].cancelled) {
          return { stopReason: "cancelled" };
        }
        break;
      }

      switch (message.type) {
        case "system":
          switch (message.subtype) {
            case "init":
              break;
            case "compact_boundary":
            case "hook_response":
            case "status":
              // Todo: process via status api: https://docs.claude.com/en/docs/claude-code/hooks#hook-output
              break;
            default:
              unreachable(message, this.logger);
              break;
          }
          break;
        case "result": {
          if (this.sessions[params.sessionId].cancelled) {
            return { stopReason: "cancelled" };
          }

          switch (message.subtype) {
            case "success": {
              if (message.result.includes("Please run /login")) {
                throw RequestError.authRequired();
              }
              if (message.is_error) {
                throw RequestError.internalError(undefined, message.result);
              }
              return { stopReason: "end_turn" };
            }
            case "error_during_execution":
              if (message.is_error) {
                throw RequestError.internalError(
                  undefined,
                  message.errors.join(", ") || message.subtype,
                );
              }
              return { stopReason: "end_turn" };
            case "error_max_budget_usd":
            case "error_max_turns":
            case "error_max_structured_output_retries":
              if (message.is_error) {
                throw RequestError.internalError(
                  undefined,
                  message.errors.join(", ") || message.subtype,
                );
              }
              return { stopReason: "max_turn_requests" };
            default:
              unreachable(message, this.logger);
              break;
          }
          break;
        }
        case "stream_event": {
          for (const notification of streamEventToAcpNotifications(
            message,
            params.sessionId,
            this.toolUseCache,
            this.client,
            this.logger,
          )) {
            await this.client.sessionUpdate(notification);
          }
          break;
        }
        case "user":
        case "assistant": {
          if (this.sessions[params.sessionId].cancelled) {
            break;
          }

          // Slash commands like /compact can generate invalid output... doesn't match
          // their own docs: https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-slash-commands#%2Fcompact-compact-conversation-history
          if (
            typeof message.message.content === "string" &&
            message.message.content.includes("<local-command-stdout>")
          ) {
            this.logger.log(message.message.content);
            break;
          }

          if (
            typeof message.message.content === "string" &&
            message.message.content.includes("<local-command-stderr>")
          ) {
            this.logger.error(message.message.content);
            break;
          }
          // Skip these user messages for now, since they seem to just be messages we don't want in the feed
          if (
            message.type === "user" &&
            (typeof message.message.content === "string" ||
              (Array.isArray(message.message.content) &&
                message.message.content.length === 1 &&
                message.message.content[0].type === "text"))
          ) {
            break;
          }

          if (
            message.type === "assistant" &&
            message.message.model === "<synthetic>" &&
            Array.isArray(message.message.content) &&
            message.message.content.length === 1 &&
            message.message.content[0].type === "text" &&
            message.message.content[0].text.includes("Please run /login")
          ) {
            throw RequestError.authRequired();
          }

          const content =
            message.type === "assistant"
              ? // Handled by stream events above
                message.message.content.filter((item) => !["text", "thinking"].includes(item.type))
              : message.message.content;

          for (const notification of toAcpNotifications(
            content,
            message.message.role,
            params.sessionId,
            this.toolUseCache,
            this.client,
            this.logger,
          )) {
            await this.client.sessionUpdate(notification);
          }
          break;
        }
        case "tool_progress":
          break;
        case "auth_status":
          break;
        default:
          unreachable(message);
          break;
      }
    }
    throw new Error("Session did not end in result");
  }

  async cancel(params: CancelNotification): Promise<void> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }
    this.sessions[params.sessionId].cancelled = true;
    await this.sessions[params.sessionId].query.interrupt();
  }

  /**
   * Handle extension methods from the client.
   *
   * Currently supports:
   * - `_session/flush`: Flush a session to disk for fork-with-flush support
   */
  async extMethod(
    method: string,
    params: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    if (method === "_session/flush") {
      return this.handleSessionFlush(
        params as { sessionId: string; idleTimeout?: number; persistTimeout?: number },
      );
    }
    throw RequestError.methodNotFound(method);
  }

  /**
   * Flush a session to disk by aborting its query subprocess.
   *
   * This is used by the fork-with-flush mechanism to ensure session data
   * is persisted to disk before forking. When the Claude SDK subprocess
   * exits (via abort), it writes the session data to:
   * ~/.claude/projects/<cwd-hash>/<sessionId>.jsonl
   *
   * After this method completes, the session is removed from memory and
   * must be reloaded via loadSession() to continue using it.
   */
  private async handleSessionFlush(params: {
    sessionId: string;
    idleTimeout?: number;
    persistTimeout?: number;
  }): Promise<Record<string, unknown>> {
    const { sessionId, persistTimeout = 5000 } = params;
    const session = this.sessions[sessionId];

    if (!session) {
      return { success: false, error: `Session ${sessionId} not found` };
    }

    try {
      // Step 1: Mark session as cancelled to stop processing
      session.cancelled = true;

      // Step 2: Interrupt any ongoing query work
      await session.query.interrupt();

      // Step 3: End the input stream to signal no more input
      session.input.end();

      // Step 4: Abort the session using the AbortController
      // This forces the Claude SDK subprocess to exit, which triggers disk persistence
      session.abortController.abort();

      // Step 5: Wait for the session file to appear on disk
      // Use stored sessionFilePath for forked sessions (where filename differs from sessionId)
      const sessionFilePath =
        session.sessionFilePath ?? this.getSessionFilePath(sessionId, session.cwd);
      this.logger.log(`[claude-code-acp] Waiting for session file at: ${sessionFilePath}`);
      this.logger.log(`[claude-code-acp] Session cwd: ${session.cwd}`);
      const persisted = await this.waitForSessionFile(sessionFilePath, persistTimeout);

      if (!persisted) {
        this.logger.error(
          `[claude-code-acp] Session file not found at ${sessionFilePath} after ${persistTimeout}ms`,
        );
        // Check if file exists at the path
        const exists = fs.existsSync(sessionFilePath);
        this.logger.error(`[claude-code-acp] File exists check: ${exists}`);
        // Still remove the session from memory
        delete this.sessions[sessionId];
        return { success: false, error: `Session file not created within timeout` };
      }

      // Step 6: Remove session from our map
      // The client will call loadSession() to reload it from disk
      delete this.sessions[sessionId];

      this.logger.log(
        `[claude-code-acp] Session ${sessionId} flushed to disk at ${sessionFilePath}`,
      );
      return { success: true, filePath: sessionFilePath };
    } catch (error) {
      this.logger.error(`[claude-code-acp] Failed to flush session ${sessionId}:`, error);
      // Clean up session on error
      delete this.sessions[sessionId];
      return { success: false, error: String(error) };
    }
  }

  /**
   * Get the file path where Claude Code stores session data.
   *
   * Claude Code stores sessions at:
   * ~/.claude/projects/<cwd-hash>/<sessionId>.jsonl
   *
   * Where <cwd-hash> is the cwd with `/` replaced by `-`
   * Note: We resolve the real path to handle macOS symlinks like /var -> /private/var
   */
  private getSessionFilePath(sessionId: string, cwd: string): string {
    // Resolve the real path to handle macOS symlinks like /var -> /private/var
    const realCwd = fs.realpathSync(cwd);
    // Claude Code replaces both / and _ with - in the cwd hash
    const cwdHash = realCwd.replace(/[/_]/g, "-");
    return path.join(os.homedir(), ".claude", "projects", cwdHash, `${sessionId}.jsonl`);
  }

  /**
   * Wait for a session file to appear on disk.
   *
   * @param filePath - Path to the session file
   * @param timeout - Maximum time to wait in milliseconds
   * @returns true if file appears, false if timeout
   */
  private async waitForSessionFile(filePath: string, timeout: number): Promise<boolean> {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      if (fs.existsSync(filePath)) {
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    return false;
  }

  async unstable_setSessionModel(
    params: SetSessionModelRequest,
  ): Promise<SetSessionModelResponse | void> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }
    await this.sessions[params.sessionId].query.setModel(params.modelId);
  }

  async setSessionMode(params: SetSessionModeRequest): Promise<SetSessionModeResponse> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }

    switch (params.modeId) {
      case "default":
      case "acceptEdits":
      case "bypassPermissions":
      case "dontAsk":
      case "plan":
        this.sessions[params.sessionId].permissionMode = params.modeId;
        try {
          await this.sessions[params.sessionId].query.setPermissionMode(params.modeId);
        } catch (error) {
          const errorMessage =
            error instanceof Error && error.message ? error.message : "Invalid Mode";

          throw new Error(errorMessage);
        }
        return {};
      default:
        throw new Error("Invalid Mode");
    }
  }

  async readTextFile(params: ReadTextFileRequest): Promise<ReadTextFileResponse> {
    const response = await this.client.readTextFile(params);
    return response;
  }

  async writeTextFile(params: WriteTextFileRequest): Promise<WriteTextFileResponse> {
    const response = await this.client.writeTextFile(params);
    return response;
  }

  canUseTool(sessionId: string): CanUseTool {
    return async (toolName, toolInput, { signal, suggestions, toolUseID }) => {
      const session = this.sessions[sessionId];
      if (!session) {
        return {
          behavior: "deny",
          message: "Session not found",
          interrupt: true,
        };
      }

      if (toolName === "ExitPlanMode") {
        const response = await this.client.requestPermission({
          options: [
            {
              kind: "allow_always",
              name: "Yes, and auto-accept edits",
              optionId: "acceptEdits",
            },
            { kind: "allow_once", name: "Yes, and manually approve edits", optionId: "default" },
            { kind: "reject_once", name: "No, keep planning", optionId: "plan" },
          ],
          sessionId,
          toolCall: {
            toolCallId: toolUseID,
            rawInput: toolInput,
            title: toolInfoFromToolUse({ name: toolName, input: toolInput }).title,
          },
        });

        if (signal.aborted || response.outcome?.outcome === "cancelled") {
          throw new Error("Tool use aborted");
        }
        if (
          response.outcome?.outcome === "selected" &&
          (response.outcome.optionId === "default" || response.outcome.optionId === "acceptEdits")
        ) {
          session.permissionMode = response.outcome.optionId;
          await this.client.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "current_mode_update",
              currentModeId: response.outcome.optionId,
            },
          });

          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              { type: "setMode", mode: response.outcome.optionId, destination: "session" },
            ],
          };
        } else {
          return {
            behavior: "deny",
            message: "User rejected request to exit plan mode.",
            interrupt: true,
          };
        }
      }

      if (
        session.permissionMode === "bypassPermissions" ||
        (session.permissionMode === "acceptEdits" && EDIT_TOOL_NAMES.includes(toolName))
      ) {
        return {
          behavior: "allow",
          updatedInput: toolInput,
          updatedPermissions: suggestions ?? [
            { type: "addRules", rules: [{ toolName }], behavior: "allow", destination: "session" },
          ],
        };
      }

      const response = await this.client.requestPermission({
        options: [
          {
            kind: "allow_always",
            name: "Always Allow",
            optionId: "allow_always",
          },
          { kind: "allow_once", name: "Allow", optionId: "allow" },
          { kind: "reject_once", name: "Reject", optionId: "reject" },
        ],
        sessionId,
        toolCall: {
          toolCallId: toolUseID,
          rawInput: toolInput,
          title: toolInfoFromToolUse({ name: toolName, input: toolInput }).title,
        },
      });
      if (signal.aborted || response.outcome?.outcome === "cancelled") {
        throw new Error("Tool use aborted");
      }
      if (
        response.outcome?.outcome === "selected" &&
        (response.outcome.optionId === "allow" || response.outcome.optionId === "allow_always")
      ) {
        // If Claude Code has suggestions, it will update their settings already
        if (response.outcome.optionId === "allow_always") {
          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              {
                type: "addRules",
                rules: [{ toolName }],
                behavior: "allow",
                destination: "session",
              },
            ],
          };
        }
        return {
          behavior: "allow",
          updatedInput: toolInput,
        };
      } else {
        return {
          behavior: "deny",
          message: "User refused permission to run tool",
          interrupt: true,
        };
      }
    };
  }

  private async createSession(
    params: NewSessionRequest,
    creationOpts: { resume?: string; forkSession?: boolean } = {},
  ): Promise<NewSessionResponse> {
    // We want to create a new session id unless it is resume,
    // but not resume + forkSession.
    let sessionId;
    if (creationOpts.forkSession) {
      sessionId = randomUUID();
    } else if (creationOpts.resume) {
      sessionId = creationOpts.resume;
    } else {
      sessionId = randomUUID();
    }

    const input = new Pushable<SDKUserMessage>();

    const settingsManager = new SettingsManager(params.cwd, {
      logger: this.logger,
    });
    await settingsManager.initialize();

    const mcpServers: Record<string, McpServerConfig> = {};
    if (Array.isArray(params.mcpServers)) {
      for (const server of params.mcpServers) {
        if ("type" in server) {
          mcpServers[server.name] = {
            type: server.type,
            url: server.url,
            headers: server.headers
              ? Object.fromEntries(server.headers.map((e) => [e.name, e.value]))
              : undefined,
          };
        } else {
          mcpServers[server.name] = {
            type: "stdio",
            command: server.command,
            args: server.args,
            env: server.env
              ? Object.fromEntries(server.env.map((e) => [e.name, e.value]))
              : undefined,
          };
        }
      }
    }

    // Only add the acp MCP server if built-in tools are not disabled
    if (!params._meta?.disableBuiltInTools) {
      const server = createMcpServer(this, sessionId, this.clientCapabilities);
      mcpServers["acp"] = {
        type: "sdk",
        name: "acp",
        instance: server,
      };
    }

    let systemPrompt: Options["systemPrompt"] = { type: "preset", preset: "claude_code" };
    if (params._meta?.systemPrompt) {
      const customPrompt = params._meta.systemPrompt;
      if (typeof customPrompt === "string") {
        systemPrompt = customPrompt;
      } else if (
        typeof customPrompt === "object" &&
        "append" in customPrompt &&
        typeof customPrompt.append === "string"
      ) {
        systemPrompt.append = customPrompt.append;
      }
    }

    const permissionMode = "default";

    // Extract options from _meta if provided
    const userProvidedOptions = (params._meta as NewSessionMeta | undefined)?.claudeCode?.options;
    const extraArgs = { ...userProvidedOptions?.extraArgs };
    if (creationOpts?.resume === undefined || creationOpts?.forkSession) {
      // Set our own session id if not resuming an existing session.
      // Note: For forked sessions (resume + fork), Claude CLI assigns its own session ID
      // which means chain forking (fork of a fork) is not currently supported.
      extraArgs["session-id"] = sessionId;
    }

    // Configure thinking tokens from environment variable
    const maxThinkingTokens = process.env.MAX_THINKING_TOKENS
      ? parseInt(process.env.MAX_THINKING_TOKENS, 10)
      : undefined;

    const options: Options = {
      systemPrompt,
      settingSources: ["user", "project", "local"],
      stderr: (err) => this.logger.error(err),
      ...(maxThinkingTokens !== undefined && { maxThinkingTokens }),
      ...userProvidedOptions,
      // Override certain fields that must be controlled by ACP
      cwd: params.cwd,
      includePartialMessages: true,
      mcpServers: { ...(userProvidedOptions?.mcpServers || {}), ...mcpServers },
      extraArgs,
      // If we want bypassPermissions to be an option, we have to allow it here.
      // But it doesn't work in root mode, so we only activate it if it will work.
      allowDangerouslySkipPermissions: !IS_ROOT,
      permissionMode,
      canUseTool: this.canUseTool(sessionId),
      // note: although not documented by the types, passing an absolute path
      // here works to find zed's managed node version.
      executable: process.execPath as any,
      ...(process.env.CLAUDE_CODE_EXECUTABLE && {
        pathToClaudeCodeExecutable: process.env.CLAUDE_CODE_EXECUTABLE,
      }),
      tools: { type: "preset", preset: "claude_code" },
      hooks: {
        ...userProvidedOptions?.hooks,
        PreToolUse: [
          ...(userProvidedOptions?.hooks?.PreToolUse || []),
          {
            hooks: [createPreToolUseHook(settingsManager, this.logger)],
          },
        ],
        PostToolUse: [
          ...(userProvidedOptions?.hooks?.PostToolUse || []),
          {
            hooks: [createPostToolUseHook(this.logger)],
          },
        ],
      },
      ...creationOpts,
    };

    const allowedTools = [];
    // Disable this for now, not a great way to expose this over ACP at the moment (in progress work so we can revisit)
    const disallowedTools = ["AskUserQuestion"];

    // Check if built-in tools should be disabled
    const disableBuiltInTools = params._meta?.disableBuiltInTools === true;

    if (!disableBuiltInTools) {
      if (this.clientCapabilities?.fs?.readTextFile) {
        allowedTools.push(acpToolNames.read);
        disallowedTools.push("Read");
      }
      if (this.clientCapabilities?.fs?.writeTextFile) {
        disallowedTools.push("Write", "Edit");
      }
      if (this.clientCapabilities?.terminal) {
        allowedTools.push(acpToolNames.bashOutput, acpToolNames.killShell);
        disallowedTools.push("Bash", "BashOutput", "KillShell");
      }
    } else {
      // When built-in tools are disabled, explicitly disallow all of them
      disallowedTools.push(
        acpToolNames.read,
        acpToolNames.write,
        acpToolNames.edit,
        acpToolNames.bash,
        acpToolNames.bashOutput,
        acpToolNames.killShell,
        "Read",
        "Write",
        "Edit",
        "Bash",
        "BashOutput",
        "KillShell",
        "Glob",
        "Grep",
        "Task",
        "TodoWrite",
        "ExitPlanMode",
        "WebSearch",
        "WebFetch",
        "AskUserQuestion",
        "SlashCommand",
        "Skill",
        "NotebookEdit",
      );
    }

    if (allowedTools.length > 0) {
      options.allowedTools = allowedTools;
    }
    if (disallowedTools.length > 0) {
      options.disallowedTools = disallowedTools;
    }

    // Create our own AbortController for session management
    const sessionAbortController = new AbortController();

    // Handle abort controller from meta options (user can still provide one)
    const userAbortController = userProvidedOptions?.abortController;
    if (userAbortController?.signal.aborted) {
      throw new Error("Cancelled");
    }

    // Pass the abort controller to the query options
    options.abortController = sessionAbortController;

    const q = query({
      prompt: input,
      options,
    });

    this.sessions[sessionId] = {
      query: q,
      input: input,
      cancelled: false,
      permissionMode,
      settingsManager,
      abortController: sessionAbortController,
      cwd: params.cwd,
    };

    const availableCommands = await getAvailableSlashCommands(q);
    const models = await getAvailableModels(q);

    // Needs to happen after we return the session
    setTimeout(() => {
      this.client.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "available_commands_update",
          availableCommands,
        },
      });
    }, 0);

    const availableModes = [
      {
        id: "default",
        name: "Default",
        description: "Standard behavior, prompts for dangerous operations",
      },
      {
        id: "acceptEdits",
        name: "Accept Edits",
        description: "Auto-accept file edit operations",
      },
      {
        id: "plan",
        name: "Plan Mode",
        description: "Planning mode, no actual tool execution",
      },
      {
        id: "dontAsk",
        name: "Don't Ask",
        description: "Don't prompt for permissions, deny if not pre-approved",
      },
    ];
    // Only works in non-root mode
    if (!IS_ROOT) {
      availableModes.push({
        id: "bypassPermissions",
        name: "Bypass Permissions",
        description: "Bypass all permission checks",
      });
    }

    return {
      sessionId,
      models,
      modes: {
        currentModeId: permissionMode,
        availableModes,
      },
    };
  }
}

async function getAvailableModels(query: Query): Promise<SessionModelState> {
  const models = await query.supportedModels();

  // Query doesn't give us access to the currently selected model, so we just choose the first model in the list.
  const currentModel = models[0];
  await query.setModel(currentModel.value);

  const availableModels = models.map((model) => ({
    modelId: model.value,
    name: model.displayName,
    description: model.description,
  }));

  return {
    availableModels,
    currentModelId: currentModel.value,
  };
}

async function getAvailableSlashCommands(query: Query): Promise<AvailableCommand[]> {
  const UNSUPPORTED_COMMANDS = [
    "context",
    "cost",
    "login",
    "logout",
    "output-style:new",
    "release-notes",
    "todos",
  ];
  const commands = await query.supportedCommands();

  return commands
    .map((command) => {
      const input = command.argumentHint
        ? {
            hint: Array.isArray(command.argumentHint)
              ? command.argumentHint.join(" ")
              : command.argumentHint,
          }
        : null;
      let name = command.name;
      if (command.name.endsWith(" (MCP)")) {
        name = `mcp:${name.replace(" (MCP)", "")}`;
      }
      return {
        name,
        description: command.description || "",
        input,
      };
    })
    .filter((command: AvailableCommand) => !UNSUPPORTED_COMMANDS.includes(command.name));
}

function formatUriAsLink(uri: string): string {
  try {
    if (uri.startsWith("file://")) {
      const path = uri.slice(7); // Remove "file://"
      const name = path.split("/").pop() || path;
      return `[@${name}](${uri})`;
    } else if (uri.startsWith("zed://")) {
      const parts = uri.split("/");
      const name = parts[parts.length - 1] || uri;
      return `[@${name}](${uri})`;
    }
    return uri;
  } catch {
    return uri;
  }
}

export function promptToClaude(prompt: PromptRequest): SDKUserMessage {
  const content: any[] = [];
  const context: any[] = [];

  for (const chunk of prompt.prompt) {
    switch (chunk.type) {
      case "text": {
        let text = chunk.text;
        // change /mcp:server:command args -> /server:command (MCP) args
        const mcpMatch = text.match(/^\/mcp:([^:\s]+):(\S+)(\s+.*)?$/);
        if (mcpMatch) {
          const [, server, command, args] = mcpMatch;
          text = `/${server}:${command} (MCP)${args || ""}`;
        }
        content.push({ type: "text", text });
        break;
      }
      case "resource_link": {
        const formattedUri = formatUriAsLink(chunk.uri);
        content.push({
          type: "text",
          text: formattedUri,
        });
        break;
      }
      case "resource": {
        if ("text" in chunk.resource) {
          const formattedUri = formatUriAsLink(chunk.resource.uri);
          content.push({
            type: "text",
            text: formattedUri,
          });
          context.push({
            type: "text",
            text: `\n<context ref="${chunk.resource.uri}">\n${chunk.resource.text}\n</context>`,
          });
        }
        // Ignore blob resources (unsupported)
        break;
      }
      case "image":
        if (chunk.data) {
          content.push({
            type: "image",
            source: {
              type: "base64",
              data: chunk.data,
              media_type: chunk.mimeType,
            },
          });
        } else if (chunk.uri && chunk.uri.startsWith("http")) {
          content.push({
            type: "image",
            source: {
              type: "url",
              url: chunk.uri,
            },
          });
        }
        break;
      // Ignore audio and other unsupported types
      default:
        break;
    }
  }

  content.push(...context);

  return {
    type: "user",
    message: {
      role: "user",
      content: content,
    },
    session_id: prompt.sessionId,
    parent_tool_use_id: null,
  };
}

/**
 * Convert an SDKAssistantMessage (Claude) to a SessionNotification (ACP).
 * Only handles text, image, and thinking chunks for now.
 */
export function toAcpNotifications(
  content: string | ContentBlockParam[] | BetaContentBlock[] | BetaRawContentBlockDelta[],
  role: "assistant" | "user",
  sessionId: string,
  toolUseCache: ToolUseCache,
  client: AgentSideConnection,
  logger: Logger,
): SessionNotification[] {
  if (typeof content === "string") {
    return [
      {
        sessionId,
        update: {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "text",
            text: content,
          },
        },
      },
    ];
  }

  const output = [];
  // Only handle the first chunk for streaming; extend as needed for batching
  for (const chunk of content) {
    let update: SessionNotification["update"] | null = null;
    switch (chunk.type) {
      case "text":
      case "text_delta":
        update = {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "text",
            text: chunk.text,
          },
        };
        break;
      case "image":
        update = {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "image",
            data: chunk.source.type === "base64" ? chunk.source.data : "",
            mimeType: chunk.source.type === "base64" ? chunk.source.media_type : "",
            uri: chunk.source.type === "url" ? chunk.source.url : undefined,
          },
        };
        break;
      case "thinking":
      case "thinking_delta":
        update = {
          sessionUpdate: "agent_thought_chunk",
          content: {
            type: "text",
            text: chunk.thinking,
          },
        };
        break;
      case "tool_use":
      case "server_tool_use":
      case "mcp_tool_use": {
        toolUseCache[chunk.id] = chunk;
        if (chunk.name === "TodoWrite") {
          // @ts-expect-error - sometimes input is empty object
          if (Array.isArray(chunk.input.todos)) {
            update = {
              sessionUpdate: "plan",
              entries: planEntries(chunk.input as { todos: ClaudePlanEntry[] }),
            };
          }
        } else {
          // Register hook callback to receive the structured output from the hook
          registerHookCallback(chunk.id, {
            onPostToolUseHook: async (toolUseId, toolInput, toolResponse) => {
              const toolUse = toolUseCache[toolUseId];
              if (toolUse) {
                const update: SessionNotification["update"] = {
                  _meta: {
                    claudeCode: {
                      toolResponse,
                      toolName: toolUse.name,
                    },
                  } satisfies ToolUpdateMeta,
                  toolCallId: toolUseId,
                  sessionUpdate: "tool_call_update",
                };
                await client.sessionUpdate({
                  sessionId,
                  update,
                });
              } else {
                logger.error(
                  `[claude-code-acp] Got a tool response for tool use that wasn't tracked: ${toolUseId}`,
                );
              }
            },
          });

          let rawInput;
          try {
            rawInput = JSON.parse(JSON.stringify(chunk.input));
          } catch {
            // ignore if we can't turn it to JSON
          }
          update = {
            _meta: {
              claudeCode: {
                toolName: chunk.name,
              },
            } satisfies ToolUpdateMeta,
            toolCallId: chunk.id,
            sessionUpdate: "tool_call",
            rawInput,
            status: "pending",
            ...toolInfoFromToolUse(chunk),
          };
        }
        break;
      }

      case "tool_result":
      case "tool_search_tool_result":
      case "web_fetch_tool_result":
      case "web_search_tool_result":
      case "code_execution_tool_result":
      case "bash_code_execution_tool_result":
      case "text_editor_code_execution_tool_result":
      case "mcp_tool_result": {
        const toolUse = toolUseCache[chunk.tool_use_id];
        if (!toolUse) {
          logger.error(
            `[claude-code-acp] Got a tool result for tool use that wasn't tracked: ${chunk.tool_use_id}`,
          );
          break;
        }

        if (toolUse.name !== "TodoWrite") {
          update = {
            _meta: {
              claudeCode: {
                toolName: toolUse.name,
              },
            } satisfies ToolUpdateMeta,
            toolCallId: chunk.tool_use_id,
            sessionUpdate: "tool_call_update",
            status: "is_error" in chunk && chunk.is_error ? "failed" : "completed",
            ...toolUpdateFromToolResult(chunk, toolUseCache[chunk.tool_use_id]),
          };
        }
        break;
      }

      case "document":
      case "search_result":
      case "redacted_thinking":
      case "input_json_delta":
      case "citations_delta":
      case "signature_delta":
      case "container_upload":
        break;

      default:
        unreachable(chunk, logger);
        break;
    }
    if (update) {
      output.push({ sessionId, update });
    }
  }

  return output;
}

export function streamEventToAcpNotifications(
  message: SDKPartialAssistantMessage,
  sessionId: string,
  toolUseCache: ToolUseCache,
  client: AgentSideConnection,
  logger: Logger,
): SessionNotification[] {
  const event = message.event;
  switch (event.type) {
    case "content_block_start":
      return toAcpNotifications(
        [event.content_block],
        "assistant",
        sessionId,
        toolUseCache,
        client,
        logger,
      );
    case "content_block_delta":
      return toAcpNotifications(
        [event.delta],
        "assistant",
        sessionId,
        toolUseCache,
        client,
        logger,
      );
    // No content
    case "message_start":
    case "message_delta":
    case "message_stop":
    case "content_block_stop":
      return [];

    default:
      unreachable(event, logger);
      return [];
  }
}

export function runAcp() {
  const input = nodeToWebWritable(process.stdout);
  const output = nodeToWebReadable(process.stdin);

  const stream = ndJsonStream(input, output);
  new AgentSideConnection((client) => new ClaudeAcpAgent(client), stream);
}
