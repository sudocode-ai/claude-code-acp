/**
 * Tests for fork-related helper methods in ClaudeAcpAgent.
 *
 * Since the methods are private, we test them by:
 * 1. Creating a test subclass that exposes the private methods
 * 2. Testing the file manipulation logic directly
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { ClaudeAcpAgent } from "../acp-agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

// Create a test subclass that exposes private methods for testing
class TestableClaudeAcpAgent extends ClaudeAcpAgent {
  // Expose private methods for testing
  public testExtractInternalSessionId(filePath: string): string | null {
    return (this as any).extractInternalSessionId(filePath);
  }

  public testPromoteToFullSession(filePath: string): boolean {
    return (this as any).promoteToFullSession(filePath);
  }

  public testUpdateSessionIdInFile(filePath: string, newSessionId: string): boolean {
    return (this as any).updateSessionIdInFile(filePath, newSessionId);
  }

  public testGetSessionDirPath(cwd: string): string {
    return (this as any).getSessionDirPath(cwd);
  }

  public testGetSessionFilePath(sessionId: string, cwd: string): string {
    return (this as any).getSessionFilePath(sessionId, cwd);
  }

  public async testDiscoverCliSessionId(
    sessionDir: string,
    beforeFiles: Set<string>,
    fallbackId: string,
    timeout?: number
  ): Promise<string> {
    return (this as any).discoverCliSessionId(sessionDir, beforeFiles, fallbackId, timeout);
  }

  public async testWaitForSessionFile(filePath: string, timeout: number): Promise<boolean> {
    return (this as any).waitForSessionFile(filePath, timeout);
  }
}

describe("ClaudeAcpAgent fork helpers", () => {
  let tempDir: string;
  let agent: TestableClaudeAcpAgent;
  const mockLogger = {
    log: vi.fn(),
    error: vi.fn(),
  };

  // Create a minimal mock client
  const mockClient = {
    sessionUpdate: vi.fn(),
    readTextFile: vi.fn(),
    writeTextFile: vi.fn(),
    requestPermission: vi.fn(),
  } as unknown as AgentSideConnection;

  beforeEach(async () => {
    tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "acp-agent-fork-test-"));
    agent = new TestableClaudeAcpAgent(mockClient, mockLogger);
    vi.clearAllMocks();
  });

  afterEach(async () => {
    await fs.promises.rm(tempDir, { recursive: true, force: true });
  });

  describe("extractInternalSessionId", () => {
    it("should extract UUID sessionId from valid JSONL file", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      const sessionId = "12345678-1234-1234-1234-123456789abc";

      await fs.promises.writeFile(
        filePath,
        JSON.stringify({ sessionId, type: "init" }) + "\n"
      );

      const result = agent.testExtractInternalSessionId(filePath);
      expect(result).toBe(sessionId);
    });

    it("should return null for non-existent file", () => {
      const result = agent.testExtractInternalSessionId(path.join(tempDir, "nonexistent.jsonl"));
      expect(result).toBeNull();
    });

    it("should return null for file without sessionId", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      await fs.promises.writeFile(
        filePath,
        JSON.stringify({ type: "init" }) + "\n"
      );

      const result = agent.testExtractInternalSessionId(filePath);
      expect(result).toBeNull();
    });

    it("should return null for non-UUID sessionId", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      await fs.promises.writeFile(
        filePath,
        JSON.stringify({ sessionId: "not-a-uuid", type: "init" }) + "\n"
      );

      const result = agent.testExtractInternalSessionId(filePath);
      expect(result).toBeNull();
    });

    it("should handle empty file", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      await fs.promises.writeFile(filePath, "");

      const result = agent.testExtractInternalSessionId(filePath);
      expect(result).toBeNull();
    });

    it("should find sessionId from first non-empty line", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      const sessionId = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";

      await fs.promises.writeFile(
        filePath,
        "\n\n" + JSON.stringify({ sessionId, type: "init" }) + "\n"
      );

      const result = agent.testExtractInternalSessionId(filePath);
      expect(result).toBe(sessionId);
    });
  });

  describe("promoteToFullSession", () => {
    it("should change isSidechain from true to false", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      const lines = [
        JSON.stringify({ sessionId: "test-id", isSidechain: true }),
        JSON.stringify({ type: "message", content: "hello" }),
      ];
      await fs.promises.writeFile(filePath, lines.join("\n"));

      const result = agent.testPromoteToFullSession(filePath);
      expect(result).toBe(true);

      const content = await fs.promises.readFile(filePath, "utf-8");
      const parsedLines = content.split("\n").map(line => JSON.parse(line));

      expect(parsedLines[0].isSidechain).toBe(false);
      expect(parsedLines[1].content).toBe("hello");
    });

    it("should return false for non-existent file", () => {
      const result = agent.testPromoteToFullSession(path.join(tempDir, "nonexistent.jsonl"));
      expect(result).toBe(false);
    });

    it("should preserve other fields when promoting", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      const original = {
        sessionId: "test-id",
        isSidechain: true,
        cwd: "/some/path",
        model: "claude-3",
      };
      await fs.promises.writeFile(filePath, JSON.stringify(original));

      agent.testPromoteToFullSession(filePath);

      const content = await fs.promises.readFile(filePath, "utf-8");
      const parsed = JSON.parse(content);

      expect(parsed.sessionId).toBe("test-id");
      expect(parsed.isSidechain).toBe(false);
      expect(parsed.cwd).toBe("/some/path");
      expect(parsed.model).toBe("claude-3");
    });

    it("should handle file without isSidechain field", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      await fs.promises.writeFile(
        filePath,
        JSON.stringify({ sessionId: "test-id", type: "init" })
      );

      const result = agent.testPromoteToFullSession(filePath);
      expect(result).toBe(true);

      const content = await fs.promises.readFile(filePath, "utf-8");
      const parsed = JSON.parse(content);
      expect(parsed.isSidechain).toBeUndefined();
    });

    it("should log success message", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      await fs.promises.writeFile(
        filePath,
        JSON.stringify({ sessionId: "test-id", isSidechain: true })
      );

      agent.testPromoteToFullSession(filePath);

      expect(mockLogger.log).toHaveBeenCalledWith(
        expect.stringContaining("Promoted sidechain to full session")
      );
    });
  });

  describe("updateSessionIdInFile", () => {
    it("should update sessionId in all lines", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      const oldId = "old-session-id";
      const newId = "new-session-id";

      const lines = [
        JSON.stringify({ sessionId: oldId, type: "init" }),
        JSON.stringify({ sessionId: oldId, type: "message", content: "hello" }),
        JSON.stringify({ sessionId: oldId, type: "tool_use", name: "Read" }),
      ];
      await fs.promises.writeFile(filePath, lines.join("\n"));

      const result = agent.testUpdateSessionIdInFile(filePath, newId);
      expect(result).toBe(true);

      const content = await fs.promises.readFile(filePath, "utf-8");
      const parsedLines = content.split("\n").map(line => JSON.parse(line));

      for (const line of parsedLines) {
        expect(line.sessionId).toBe(newId);
      }
    });

    it("should return false for non-existent file", () => {
      const result = agent.testUpdateSessionIdInFile(
        path.join(tempDir, "nonexistent.jsonl"),
        "new-id"
      );
      expect(result).toBe(false);
    });

    it("should preserve lines without sessionId", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      const lines = [
        JSON.stringify({ sessionId: "old-id", type: "init" }),
        JSON.stringify({ type: "comment", text: "no sessionId here" }),
      ];
      await fs.promises.writeFile(filePath, lines.join("\n"));

      agent.testUpdateSessionIdInFile(filePath, "new-id");

      const content = await fs.promises.readFile(filePath, "utf-8");
      const parsedLines = content.split("\n").map(line => JSON.parse(line));

      expect(parsedLines[0].sessionId).toBe("new-id");
      expect(parsedLines[1].sessionId).toBeUndefined();
      expect(parsedLines[1].text).toBe("no sessionId here");
    });

    it("should handle empty lines gracefully", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      const content = JSON.stringify({ sessionId: "old-id" }) + "\n\n" + JSON.stringify({ sessionId: "old-id" });
      await fs.promises.writeFile(filePath, content);

      const result = agent.testUpdateSessionIdInFile(filePath, "new-id");
      expect(result).toBe(true);

      const newContent = await fs.promises.readFile(filePath, "utf-8");
      const lines = newContent.split("\n");

      expect(JSON.parse(lines[0]).sessionId).toBe("new-id");
      expect(lines[1]).toBe(""); // Empty line preserved
      expect(JSON.parse(lines[2]).sessionId).toBe("new-id");
    });

    it("should log success message", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      await fs.promises.writeFile(
        filePath,
        JSON.stringify({ sessionId: "old-id" })
      );

      agent.testUpdateSessionIdInFile(filePath, "new-id");

      expect(mockLogger.log).toHaveBeenCalledWith(
        expect.stringContaining("Updated session ID in file")
      );
    });
  });

  describe("getSessionDirPath", () => {
    it("should compute correct path with cwd hash", () => {
      const result = agent.testGetSessionDirPath("/private/tmp");
      const homeDir = os.homedir();

      expect(result).toBe(`${homeDir}/.claude/projects/-private-tmp`);
    });

    it("should replace both / and _ with -", () => {
      // Use tempDir which exists
      const testDir = path.join(tempDir, "my_project");
      fs.mkdirSync(testDir, { recursive: true });

      const result = agent.testGetSessionDirPath(testDir);
      const realPath = fs.realpathSync(testDir);
      const expectedHash = realPath.replace(/[/_]/g, "-");

      expect(result).toBe(`${os.homedir()}/.claude/projects/${expectedHash}`);
    });

    it("should resolve symlinks", () => {
      // /var on macOS is a symlink to /private/var
      const result = agent.testGetSessionDirPath("/var/tmp");
      const homeDir = os.homedir();

      // Should use the resolved path
      expect(result).toBe(`${homeDir}/.claude/projects/-private-var-tmp`);
    });
  });

  describe("getSessionFilePath", () => {
    it("should append sessionId.jsonl to session dir", () => {
      const result = agent.testGetSessionFilePath("my-session-id", "/private/tmp");
      const homeDir = os.homedir();

      expect(result).toBe(`${homeDir}/.claude/projects/-private-tmp/my-session-id.jsonl`);
    });
  });

  describe("waitForSessionFile", () => {
    it("should return true immediately if file exists", async () => {
      const filePath = path.join(tempDir, "session.jsonl");
      await fs.promises.writeFile(filePath, "{}");

      const start = Date.now();
      const result = await agent.testWaitForSessionFile(filePath, 1000);
      const elapsed = Date.now() - start;

      expect(result).toBe(true);
      expect(elapsed).toBeLessThan(200);
    });

    it("should return true when file appears before timeout", async () => {
      const filePath = path.join(tempDir, "session.jsonl");

      // Create file after 200ms
      setTimeout(async () => {
        await fs.promises.writeFile(filePath, "{}");
      }, 200);

      const start = Date.now();
      const result = await agent.testWaitForSessionFile(filePath, 2000);
      const elapsed = Date.now() - start;

      expect(result).toBe(true);
      expect(elapsed).toBeGreaterThanOrEqual(200);
      expect(elapsed).toBeLessThan(2000);
    });

    it("should return false when timeout expires", async () => {
      const filePath = path.join(tempDir, "nonexistent.jsonl");

      const start = Date.now();
      const result = await agent.testWaitForSessionFile(filePath, 300);
      const elapsed = Date.now() - start;

      expect(result).toBe(false);
      expect(elapsed).toBeGreaterThanOrEqual(300);
      expect(elapsed).toBeLessThan(500);
    });
  });

  describe("discoverCliSessionId", () => {
    it("should find new agent-xxx file", async () => {
      const sessionDir = path.join(tempDir, "sessions");
      await fs.promises.mkdir(sessionDir, { recursive: true });

      const beforeFiles = new Set<string>();

      // Create an agent-xxx file
      const agentFile = "agent-abc123.jsonl";
      await fs.promises.writeFile(path.join(sessionDir, agentFile), "{}");

      const result = await agent.testDiscoverCliSessionId(sessionDir, beforeFiles, "fallback", 1000);

      expect(result).toBe("agent-abc123");
    });

    it("should ignore non-agent files", async () => {
      const sessionDir = path.join(tempDir, "sessions");
      await fs.promises.mkdir(sessionDir, { recursive: true });

      const beforeFiles = new Set<string>();

      // Create a UUID-named file (not agent-xxx)
      const uuidFile = "12345678-1234-1234-1234-123456789abc.jsonl";
      await fs.promises.writeFile(path.join(sessionDir, uuidFile), "{}");

      const result = await agent.testDiscoverCliSessionId(sessionDir, beforeFiles, "fallback", 500);

      expect(result).toBe("fallback");
    });

    it("should return fallback when no new files found", async () => {
      const sessionDir = path.join(tempDir, "sessions");
      await fs.promises.mkdir(sessionDir, { recursive: true });

      const beforeFiles = new Set<string>();

      const result = await agent.testDiscoverCliSessionId(sessionDir, beforeFiles, "fallback-id", 300);

      expect(result).toBe("fallback-id");
    });

    it("should ignore files that existed before", async () => {
      const sessionDir = path.join(tempDir, "sessions");
      await fs.promises.mkdir(sessionDir, { recursive: true });

      // Create file before
      const existingFile = "agent-existing.jsonl";
      await fs.promises.writeFile(path.join(sessionDir, existingFile), "{}");

      const beforeFiles = new Set<string>([existingFile]);

      const result = await agent.testDiscoverCliSessionId(sessionDir, beforeFiles, "fallback", 300);

      expect(result).toBe("fallback");
    });

    it("should return newest file when multiple agent files appear", async () => {
      const sessionDir = path.join(tempDir, "sessions");
      await fs.promises.mkdir(sessionDir, { recursive: true });

      const beforeFiles = new Set<string>();

      // Create first file - use hex chars to match agent-[a-f0-9]+ pattern
      const firstPath = path.join(sessionDir, "agent-aaa111.jsonl");
      await fs.promises.writeFile(firstPath, "{}");

      // Wait a bit to ensure different mtime, then create second file
      await new Promise(resolve => setTimeout(resolve, 100));
      const secondPath = path.join(sessionDir, "agent-bbb222.jsonl");
      await fs.promises.writeFile(secondPath, "{}");

      // Verify files exist before calling discover
      expect(fs.existsSync(firstPath)).toBe(true);
      expect(fs.existsSync(secondPath)).toBe(true);

      const result = await agent.testDiscoverCliSessionId(sessionDir, beforeFiles, "fallback", 1000);

      expect(result).toBe("agent-bbb222");
    });

    it("should handle non-existent session directory initially", async () => {
      const sessionDir = path.join(tempDir, "nonexistent-dir");
      const beforeFiles = new Set<string>();

      // Start the discovery in parallel with file creation
      const discoverPromise = agent.testDiscoverCliSessionId(sessionDir, beforeFiles, "fallback", 2000);

      // Create directory and file after 200ms (within the 2000ms timeout)
      // Use hex chars to match agent-[a-f0-9]+ pattern
      await new Promise(resolve => setTimeout(resolve, 200));
      await fs.promises.mkdir(sessionDir, { recursive: true });
      await fs.promises.writeFile(path.join(sessionDir, "agent-ccc333.jsonl"), "{}");

      const result = await discoverPromise;

      expect(result).toBe("agent-ccc333");
    });
  });
});
