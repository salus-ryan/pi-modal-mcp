# pi-modal-mcp

Drive **Modal** serverless compute (GPU workers + headless Chromium) from any
MCP-aware agent. Zero local dependencies for the remote path; a tiny stdio shim
for stdio-only clients like [pi](https://pi.dev).

Three pieces, one system:

```
agent (pi / Claude Desktop / Cursor / your Linux box)
  │
  │  MCP (stdio shim OR streamable HTTP)
  ▼
Modal frontend  ── FastAPI + MCP-over-HTTP
  │  (.remote.aio fan-out)
  ├─▶ run_model containers   (CPU workers, parallel)
  └─▶ browse_page container  (Playwright + Chromium → PNG)
```

## What's inside

| Path | What |
|---|---|
| `modal-frontend/app.py` | Modal app: FastAPI REST endpoints **and** a Modal-hosted MCP-over-HTTP server (stateless streamable HTTP). Deploys Playwright for screenshots. |
| `modal-mcp/modal_mcp.py` | Stdio MCP shim. Bridges stdio-only MCP clients (e.g. pi) to the Modal REST endpoints. Returns screenshots as MCP image content. |
| `mcp-runtime/index.ts` | Pi extension: a native MCP runtime for pi. Reads `~/.pi/agent/mcp.json`, spawns stdio MCP servers, registers their tools as pi tools. |
| `mcp-runtime/mcp.json.example` | Example pi MCP config. |

## Tools exposed

| Tool | Does |
|---|---|
| `modal_ping` | Health-check the Modal frontend. |
| `modal_models` | Fan out N parallel workers on Modal. Returns JSON: concurrent count, elapsed seconds, per-worker results. |
| `modal_browse` | Open a URL in headless Chromium inside a Modal container. Returns a PNG screenshot (MCP image content). |

## Deploy the Modal app

Requires the [Modal CLI](https://modal.com/docs/guide) authenticated.

```bash
cd modal-frontend
modal deploy app.py
```

You'll get two URLs:

```
web:      https://YOUR-WORKSPACE--pi-frontend-web.modal.run       (REST: /api/*)
mcp_web:  https://YOUR-WORKSPACE--pi-frontend-mcp-web.modal.run/mcp  (MCP streamable HTTP)
```

### Why stateless?

Modal load-balances each HTTP request across containers, so stateful MCP
sessions (which require session affinity) break. The app sets
`stateless_http=True` and disables DNS-rebinding host checks so any host can
reach it. For a private deployment, tighten `transport_security.allowed_hosts`.

## Use it

### Remote MCP client (Claude Desktop / Cursor / your Linux box)

```json
{
  "mcpServers": {
    "modal": {
      "url": "https://YOUR-WORKSPACE--pi-frontend-mcp-web.modal.run/mcp"
    }
  }
}
```

Or the `mcp` Python client:

```bash
pip install mcp
python3 - <<'PY'
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
async def main():
    async with streamablehttp_client("https://YOUR-WORKSPACE--pi-frontend-mcp-web.modal.run/mcp") as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize()
            print([t.name for t in (await s.list_tools()).tools])
            res = await s.call_tool("modal_browse", {"url": "example.com"})
            print([getattr(c,'type',None) for c in res.content])
asyncio.run(main())
PY
```

### pi (native remote transport, no shim)

The `mcp-runtime` extension now speaks the remote (streamable-HTTP) MCP
transport directly, so pi can hit the Modal-hosted MCP server with **zero
local component**.

1. Install the extension:

```bash
mkdir -p ~/.pi/agent/extensions/mcp-runtime
cp mcp-runtime/index.ts mcp-runtime/package.json ~/.pi/agent/extensions/mcp-runtime/
cd ~/.pi/agent/extensions/mcp-runtime && npm install
```

2. Add `~/.pi/agent/mcp.json` (replace workspace + URL):

```json
{
  "mcpServers": {
    "modal": {
      "url": "https://YOUR-WORKSPACE--pi-frontend-mcp-web.modal.run/mcp",
      "transport": "streamable-http"
    }
  }
}
```

3. Restart pi (or `/reload`), then `/mcp list` to see `mcp_modal_modal_ping`,
   `mcp_modal_modal_models`, `mcp_modal_modal_browse` tagged `[remote]`.

### pi (stdio fallback) via the shim

For MCP servers that only expose a stdio interface (or for air-gapped use),
the `modal-mcp` shim still works:

1. Install the shim on PATH:

```bash
cp modal-mcp/run.sh /usr/local/bin/modal-mcp   # or anywhere on PATH
chmod +x /usr/local/bin/modal-mcp
```

2. Add `~/.pi/agent/mcp.json` using the `command` form instead of `url`:

```json
{
  "mcpServers": {
    "modal": {
      "command": "modal-mcp",
      "args": [],
      "env": {
        "MODAL_FRONTEND": "https://YOUR-WORKSPACE--pi-frontend-web.modal.run"
      }
    }
  }
}
```

3. `/mcp list` shows the tools tagged `[stdio]`.

### Why stateless + 405 GET?

Modal load-balances each HTTP request across containers, so stateful MCP
sessions (which require session affinity) break. The app sets
`stateless_http=True` and disables DNS-rebinding host checks so any host can
reach it.

A second subtlety: the MCP **Node** SDK's streamable-HTTP client, on `start()`,
first tries `GET /mcp` to open a listening SSE stream; if the server returns
`200` and holds it open, the client blocks waiting for events that a stateless
server never delivers. A stateless server has no stream to offer, so the app
wraps the MCP ASGI app in a middleware that returns **`405` for `GET /mcp`**,
telling strict clients to skip the GET stream and use POST-inline responses.
This makes the endpoint compatible with the Node SDK, the Python SDK, Claude
Desktop, and Cursor alike.

For a private deployment, tighten `transport_security.allowed_hosts`.

## Transports

The pi `mcp-runtime` extension supports both backends from one config:

- **stdio** — `{ "command": "...", "args": [...] }` spawns a local process.
- **remote** — `{ "url": "https://...", "transport": "streamable-http" }`
  talks to a remote MCP-over-HTTP server directly. `"transport": "sse"`
  selects the legacy SSE client. Optional `"headers": { ... }` for auth.

Remote transport means pi needs **zero local component** for cloud-hosted MCP
servers — no shim, no `npx`/Python process to spawn.

## How many models can I run simultaneously?

Modal scales each function horizontally — there's no fixed concurrency cap.
You can fan out hundreds of concurrent `.remote()` calls; Modal spins up
containers on demand. What actually bounds you:

- **Concurrent containers quota** per workspace (free tiers are low; extra calls queue, they don't fail).
- **GPU availability** for `gpu=...` functions.
- **Cold starts** — first container of a function takes a few seconds; warm containers are ~instant.
- **Your fan-out shape** — `asyncio.gather(*[fn.remote.aio(x) for x in items])` issues all calls concurrently.

## License

MIT
