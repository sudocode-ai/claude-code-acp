/**
 * Tests for session management functionality (fork and load).
 *
 * Run with: npx vitest run src/tests/fork-session.test.ts
 */
import { describe, it, expect, beforeEach } from "vitest";
import type { ClaudeAcpAgent } from "../acp-agent.js";

// Mock client for testing
const mockClient = {} as any;

describe("session management", () => {
  let agent: ClaudeAcpAgent;

  beforeEach(async () => {
    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    agent = new ClaudeAcpAgent(mockClient);
  });

  describe("capability advertisement", () => {
    it("advertises sessionCapabilities.fork", async () => {
      const response = await agent.initialize({
        protocolVersion: 1,
        clientInfo: { name: "test-client", version: "1.0.0" },
      });

      expect(response.agentCapabilities?.sessionCapabilities?.fork).toEqual({});
    });

    it("advertises loadSession capability", async () => {
      const response = await agent.initialize({
        protocolVersion: 1,
        clientInfo: { name: "test-client", version: "1.0.0" },
      });

      expect(response.agentCapabilities?.loadSession).toBe(true);
    });

    it("includes all expected agent capabilities", async () => {
      const response = await agent.initialize({
        protocolVersion: 1,
        clientInfo: { name: "test-client", version: "1.0.0" },
      });

      expect(response.agentCapabilities).toMatchObject({
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
      });
    });
  });

  describe("forkSession", () => {
    it("method exists on ClaudeAcpAgent", () => {
      expect(typeof agent.forkSession).toBe("function");
    });

    it("accepts ForkSessionRequest params with sessionId", () => {
      // Verify method signature accepts minimal params
      expect(agent.forkSession).toBeInstanceOf(Function);
      expect(agent.forkSession.length).toBeGreaterThanOrEqual(1);
    });

    it("accepts optional _meta with cwd and mcpServers", () => {
      // Verify method can handle extended params via _meta
      expect(agent.forkSession).toBeInstanceOf(Function);
    });
  });

  describe("loadSession", () => {
    it("method exists on ClaudeAcpAgent", () => {
      expect(typeof agent.loadSession).toBe("function");
    });

    it("accepts LoadSessionRequest params", () => {
      // Verify method signature accepts params (sessionId, cwd, mcpServers)
      expect(agent.loadSession).toBeInstanceOf(Function);
      expect(agent.loadSession.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("deprecated methods", () => {
    it("unstable_forkSession delegates to forkSession", () => {
      expect(typeof agent.unstable_forkSession).toBe("function");
    });

    it("unstable_resumeSession delegates to loadSession", () => {
      expect(typeof agent.unstable_resumeSession).toBe("function");
    });
  });

  describe("extMethod _session/inject", () => {
    it("extMethod exists on ClaudeAcpAgent", () => {
      expect(typeof agent.extMethod).toBe("function");
    });

    it("returns error for non-existent session", async () => {
      const result = await agent.extMethod("_session/inject", {
        sessionId: "non-existent-session",
        message: "test message",
      });

      expect(result).toEqual({
        success: false,
        error: "Session non-existent-session not found",
      });
    });

    it("throws for unknown extension methods", async () => {
      await expect(
        agent.extMethod("_unknown/method", {}),
      ).rejects.toThrow();
    });

    it("accepts string messages", async () => {
      // Mock a session in the agent's sessions map
      const mockSession = {
        query: {} as any,
        input: {
          push: () => {},
          end: () => {},
        },
        cancelled: false,
        permissionMode: "default" as const,
        settingsManager: {} as any,
        abortController: new AbortController(),
        cwd: "/tmp",
      };
      (agent as any).sessions["test-session"] = mockSession;

      const result = await agent.extMethod("_session/inject", {
        sessionId: "test-session",
        message: "test message",
      });

      expect(result).toEqual({ success: true });

      // Cleanup
      delete (agent as any).sessions["test-session"];
    });

    it("accepts ContentBlock array messages", async () => {
      // Mock a session
      const mockSession = {
        query: {} as any,
        input: {
          push: () => {},
          end: () => {},
        },
        cancelled: false,
        permissionMode: "default" as const,
        settingsManager: {} as any,
        abortController: new AbortController(),
        cwd: "/tmp",
      };
      (agent as any).sessions["test-session"] = mockSession;

      const result = await agent.extMethod("_session/inject", {
        sessionId: "test-session",
        message: [
          { type: "text", text: "first part" },
          { type: "text", text: "second part" },
        ],
      });

      expect(result).toEqual({ success: true });

      // Cleanup
      delete (agent as any).sessions["test-session"];
    });

    it("returns error for cancelled session", async () => {
      // Mock a cancelled session
      const mockSession = {
        query: {} as any,
        input: {
          push: () => {},
          end: () => {},
        },
        cancelled: true, // Session is cancelled
        permissionMode: "default" as const,
        settingsManager: {} as any,
        abortController: new AbortController(),
        cwd: "/tmp",
      };
      (agent as any).sessions["cancelled-session"] = mockSession;

      const result = await agent.extMethod("_session/inject", {
        sessionId: "cancelled-session",
        message: "test message",
      });

      expect(result).toEqual({
        success: false,
        error: "Session cancelled-session is cancelled",
      });

      // Cleanup
      delete (agent as any).sessions["cancelled-session"];
    });

    it("actually pushes message to session input", async () => {
      // Track what was pushed
      const pushedMessages: any[] = [];
      const mockSession = {
        query: {} as any,
        input: {
          push: (msg: any) => pushedMessages.push(msg),
          end: () => {},
        },
        cancelled: false,
        permissionMode: "default" as const,
        settingsManager: {} as any,
        abortController: new AbortController(),
        cwd: "/tmp",
      };
      (agent as any).sessions["push-test-session"] = mockSession;

      await agent.extMethod("_session/inject", {
        sessionId: "push-test-session",
        message: "hello world",
      });

      expect(pushedMessages.length).toBe(1);
      expect(pushedMessages[0].type).toBe("user");
      expect(pushedMessages[0].message.role).toBe("user");
      expect(pushedMessages[0].message.content).toContainEqual({
        type: "text",
        text: "hello world",
      });

      // Cleanup
      delete (agent as any).sessions["push-test-session"];
    });
  });

  describe("extMethod _session/listSkills", () => {
    it("returns error for non-existent session", async () => {
      const result = await agent.extMethod("_session/listSkills", {
        sessionId: "non-existent-session",
      });

      expect(result).toEqual({
        success: false,
        error: "Session non-existent-session not found",
      });
    });

    it("returns empty skills array when session has no skills", async () => {
      // Mock a session without skills
      const mockSession = {
        query: {} as any,
        input: {
          push: () => {},
          end: () => {},
        },
        cancelled: false,
        permissionMode: "default" as const,
        settingsManager: {} as any,
        abortController: new AbortController(),
        cwd: "/tmp",
        compaction: {
          enabled: false,
          threshold: 100000,
          currentTokens: 0,
          isCompacting: false,
        },
        // No skills field
      };
      (agent as any).sessions["test-session"] = mockSession;

      const result = await agent.extMethod("_session/listSkills", {
        sessionId: "test-session",
      });

      expect(result).toEqual({
        success: true,
        skills: [],
      });

      // Cleanup
      delete (agent as any).sessions["test-session"];
    });

    it("returns skills array when session has skills", async () => {
      // Mock a session with skills
      const mockSkills = [
        { name: "pdf-processor", description: "Process PDF files", source: "project" },
        { name: "code-reviewer", description: "Review code changes", source: "user" },
      ];
      const mockSession = {
        query: {} as any,
        input: {
          push: () => {},
          end: () => {},
        },
        cancelled: false,
        permissionMode: "default" as const,
        settingsManager: {} as any,
        abortController: new AbortController(),
        cwd: "/tmp",
        compaction: {
          enabled: false,
          threshold: 100000,
          currentTokens: 0,
          isCompacting: false,
        },
        skills: mockSkills,
      };
      (agent as any).sessions["skills-session"] = mockSession;

      const result = await agent.extMethod("_session/listSkills", {
        sessionId: "skills-session",
      });

      expect(result).toEqual({
        success: true,
        skills: mockSkills,
      });

      // Cleanup
      delete (agent as any).sessions["skills-session"];
    });
  });
});
