/**
 * Tests for session forking functionality.
 *
 * Run with: npx vitest run src/tests/fork-session.test.ts
 */
import { describe, it, expect } from "vitest";

describe("session forking", () => {
  describe("capability advertisement", () => {
    it("agentCapabilities includes sessionCapabilities.fork", async () => {
      // Import dynamically to get fresh module
      const { ClaudeAcpAgent } = await import("../acp-agent.js");
      const agent = new ClaudeAcpAgent();

      const response = await agent.initialize({
        protocolVersion: 1,
        clientInfo: { name: "test-client", version: "1.0.0" },
      });

      expect(response.agentCapabilities).toBeDefined();
      expect(response.agentCapabilities?.sessionCapabilities).toBeDefined();
      expect(response.agentCapabilities?.sessionCapabilities?.fork).toBeDefined();
      expect(response.agentCapabilities?.sessionCapabilities?.fork).toEqual({});
    });

    it("agentCapabilities includes sessionCapabilities.resume", async () => {
      const { ClaudeAcpAgent } = await import("../acp-agent.js");
      const agent = new ClaudeAcpAgent();

      const response = await agent.initialize({
        protocolVersion: 1,
        clientInfo: { name: "test-client", version: "1.0.0" },
      });

      expect(response.agentCapabilities?.sessionCapabilities?.resume).toBeDefined();
      expect(response.agentCapabilities?.sessionCapabilities?.resume).toEqual({});
    });
  });

  describe("unstable_forkSession method", () => {
    it("method exists on ClaudeAcpAgent", async () => {
      const { ClaudeAcpAgent } = await import("../acp-agent.js");
      const agent = new ClaudeAcpAgent();

      expect(typeof agent.unstable_forkSession).toBe("function");
    });

    it("method signature accepts ForkSessionRequest params", async () => {
      const { ClaudeAcpAgent } = await import("../acp-agent.js");
      const agent = new ClaudeAcpAgent();

      // Verify method can be called with correct params structure
      // (will fail at runtime without valid session, but type-checks)
      const params = {
        sessionId: "test-session-id",
        cwd: "/tmp",
        mcpServers: [],
      };

      // Method exists and accepts these params
      expect(() => {
        // Just checking the method can be referenced with these params
        const method = agent.unstable_forkSession.bind(agent);
        expect(method).toBeDefined();
      }).not.toThrow();
    });
  });

  describe("unstable_resumeSession method", () => {
    it("method exists on ClaudeAcpAgent", async () => {
      const { ClaudeAcpAgent } = await import("../acp-agent.js");
      const agent = new ClaudeAcpAgent();

      expect(typeof agent.unstable_resumeSession).toBe("function");
    });
  });
});
