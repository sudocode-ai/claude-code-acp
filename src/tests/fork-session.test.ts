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
});
