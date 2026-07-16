import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

interface McpServerConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
  disabled?: boolean;
}

interface McpConfig {
  mcpServers?: Record<string, McpServerConfig>;
}

interface ConnectedServer {
  name: string;
  config: McpServerConfig;
  client: Client;
  transport: StdioClientTransport;
  tools: any[];
}

const GLOBAL_CONFIG = join(process.env.HOME ?? "", ".pi", "agent", "mcp.json");
const PROJECT_CONFIG = ".pi/mcp.json";

function safeToolName(server: string, tool: string) {
  return `mcp_${server}_${tool}`.replace(/[^A-Za-z0-9_]/g, "_").replace(/^([^A-Za-z_])/, "_$1");
}

function readJson(path: string): McpConfig {
  return JSON.parse(readFileSync(path, "utf8"));
}

function loadConfig(ctx: ExtensionContext): McpConfig {
  const merged: McpConfig = { mcpServers: {} };
  if (existsSync(GLOBAL_CONFIG)) Object.assign(merged.mcpServers!, readJson(GLOBAL_CONFIG).mcpServers ?? {});

  const projectPath = join(ctx.cwd, PROJECT_CONFIG);
  if (ctx.isProjectTrusted() && existsSync(projectPath)) {
    Object.assign(merged.mcpServers!, readJson(projectPath).mcpServers ?? {});
  }
  return merged;
}

function mcpContentToPi(content: any): any[] {
  const items = Array.isArray(content) ? content : [{ type: "text", text: JSON.stringify(content) }];
  return items.map((item) => {
    if (item?.type === "text") return { type: "text", text: String(item.text ?? "") };
    if (item?.type === "image") {
      return {
        type: "image",
        data: String(item.data ?? ""),
        mimeType: String(item.mimeType ?? "image/png"),
      };
    }
    if (item?.type === "resource") return { type: "text", text: JSON.stringify(item.resource ?? item, null, 2) };
    return { type: "text", text: JSON.stringify(item, null, 2) };
  });
}

export default function mcpRuntime(pi: ExtensionAPI) {
  const servers = new Map<string, ConnectedServer>();
  const toolToServer = new Map<string, { serverName: string; toolName: string }>();

  async function disconnectAll() {
    const entries = [...servers.values()];
    servers.clear();
    toolToServer.clear();
    await Promise.allSettled(entries.map(async (server) => {
      await server.client.close();
    }));
  }

  async function connectConfigured(ctx: ExtensionContext) {
    await disconnectAll();
    const config = loadConfig(ctx);
    const configured = Object.entries(config.mcpServers ?? {}).filter(([, server]) => !server.disabled);

    for (const [name, serverConfig] of configured) {
      try {
        const client = new Client({ name: "pi-mcp-runtime", version: "0.1.0" });
        const transport = new StdioClientTransport({
          command: serverConfig.command,
          args: serverConfig.args ?? [],
          env: { ...process.env, ...(serverConfig.env ?? {}) } as Record<string, string>,
          cwd: serverConfig.cwd,
        });
        await client.connect(transport);
        const listed = await client.listTools();
        const connected: ConnectedServer = { name, config: serverConfig, client, transport, tools: listed.tools ?? [] };
        servers.set(name, connected);

        for (const tool of connected.tools) {
          const piName = safeToolName(name, tool.name);
          toolToServer.set(piName, { serverName: name, toolName: tool.name });
          pi.registerTool({
            name: piName,
            label: `MCP ${name}:${tool.name}`,
            description: tool.description ?? `Call MCP tool ${tool.name} on server ${name}`,
            promptSnippet: `Call MCP tool ${name}:${tool.name}`,
            parameters: Type.Unsafe(tool.inputSchema ?? { type: "object", properties: {}, additionalProperties: true }),
            async execute(_toolCallId, params) {
              const mapping = toolToServer.get(piName);
              if (!mapping) throw new Error(`MCP tool ${piName} is no longer connected`);
              const server = servers.get(mapping.serverName);
              if (!server) throw new Error(`MCP server ${mapping.serverName} is not connected`);
              const result: any = await server.client.callTool({ name: mapping.toolName, arguments: params ?? {} });
              return {
                content: mcpContentToPi(result.content ?? result),
                details: { server: mapping.serverName, tool: mapping.toolName, isError: result.isError === true },
                isError: result.isError === true,
              };
            },
          });
        }
      } catch (error: any) {
        ctx.ui.notify(`MCP server '${name}' failed: ${error?.message ?? error}`, "error");
      }
    }

    ctx.ui.setStatus("mcp", `mcp: ${servers.size} server(s), ${toolToServer.size} tool(s)`);
  }

  pi.on("session_start", async (_event, ctx) => {
    await connectConfigured(ctx);
  });

  pi.on("session_shutdown", async () => {
    await disconnectAll();
  });

  pi.registerCommand("mcp", {
    description: "Manage native MCP runtime: /mcp list | /mcp reload | /mcp config",
    handler: async (args, ctx) => {
      const sub = (args ?? "list").trim();
      if (sub === "reload") {
        await connectConfigured(ctx);
        ctx.ui.notify(`Reloaded MCP: ${servers.size} server(s), ${toolToServer.size} tool(s)`, "info");
        return;
      }
      if (sub === "config") {
        ctx.ui.notify(`Global MCP config: ${GLOBAL_CONFIG}\nProject MCP config: ${join(ctx.cwd, PROJECT_CONFIG)} (trusted projects only)`, "info");
        return;
      }
      const lines = [...servers.values()].flatMap((server) => [
        `${server.name} (${server.tools.length} tools)`,
        ...server.tools.map((tool) => `  ${safeToolName(server.name, tool.name)} -> ${server.name}:${tool.name}`),
      ]);
      ctx.ui.notify(lines.length ? lines.join("\n") : "No MCP servers connected. Create ~/.pi/agent/mcp.json then /mcp reload.", "info");
    },
  });
}
