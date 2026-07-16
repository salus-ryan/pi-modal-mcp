#!/usr/bin/env python3
"""modal-mcp: stdio MCP server exposing a Modal frontend as Pi tools."""
import os, json, sys, urllib.request, base64
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

FRONTEND = os.environ.get("MODAL_FRONTEND")
if not FRONTEND:
    raise SystemExit("modal-mcp: set MODAL_FRONTEND to your deployed web URL, e.g. https://YOUR-WORKSPACE--pi-frontend-web.modal.run")

server = Server("modal-mcp")


def _get(path: str, headers=None) -> tuple[bytes, dict]:
    url = f"{FRONTEND}{path}"
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read(), dict(r.headers)


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="modal_ping",
            description="Health-check the Modal frontend.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="modal_models",
            description="Fan out N parallel workers on Modal. Returns JSON with elapsed time and per-worker results.",
            inputSchema={
                "type": "object",
                "properties": {"n": {"type": "integer", "default": 3, "minimum": 1, "maximum": 64}},
            },
        ),
        Tool(
            name="modal_browse",
            description="Open a URL in a headless Chromium browser inside a Modal container and return a PNG screenshot.",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to visit (e.g. example.com)"}},
                "required": ["url"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "modal_ping":
        body, _ = _get("/api/ping")
        return [TextContent(type="text", text=body.decode())]
    if name == "modal_models":
        n = arguments.get("n", 3)
        body, _ = _get(f"/api/models?n={n}")
        return [TextContent(type="text", text=body.decode())]
    if name == "modal_browse":
        url = arguments["url"]
        import urllib.parse
        path = "/api/browse?url=" + urllib.parse.quote(url, safe="")
        body, hdrs = _get(path)
        elapsed = hdrs.get("X-Elapsed-Seconds", "?")
        b64 = base64.b64encode(body).decode("ascii")
        return [
            TextContent(type="text", text=f"screenshot of {url} ({len(body)} bytes, {elapsed}s)"),
            ImageContent(type="image", data=b64, mimeType="image/png"),
        ]
    raise ValueError(f"unknown tool: {name}")


async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


def main_sync() -> None:
    """Console-script entry point."""
    import asyncio

    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
