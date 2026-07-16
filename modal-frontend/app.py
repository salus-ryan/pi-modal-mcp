import modal

app = modal.App("pi-frontend")

base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.6",
    "uvicorn==0.34.0",
    "mcp>=1.2.0",
)

# Playwright image: chromium + deps, pre-warmed.
pw_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "wget")
    .pip_install("playwright==1.49.1")
    .run_commands("playwright install chromium --with-deps")
)


@app.function(image=base_image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response, JSONResponse
    import asyncio, time

    api = FastAPI(title="pi-frontend")

    HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>pi-frontend</title></head><body style=font-family:monospace;padding:2rem>
<h1>pi-frontend</h1><p>Endpoints:</p><ul>
<li><a href=/api/ping>/api/ping</a></li>
<li><a href=/api/time>/api/time</a></li>
<li><a href=/api/models?n=3>/api/models?n=3</a> - parallel fan-out</li>
<li><a href=/api/browse?url=example.com>/api/browse?url=example.com</a> - headless screenshot (PNG)</li>
</ul></body></html>"""

    @api.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @api.get("/api/ping")
    def ping():
        return {"ok": True, "service": "pi-frontend"}

    @api.get("/api/time")
    def now():
        return {"time": time.time()}

    @api.get("/api/models")
    async def models(n: int = 3):
        n = max(1, min(n, 64))
        t0 = time.time()
        tasks = [run_model.remote.aio(f"model-{i}") for i in range(n)]
        results = await asyncio.gather(*tasks)
        return {"concurrent": n, "elapsed": round(time.time() - t0, 3), "results": results}

    @api.get("/api/browse")
    async def browse(url: str = "example.com"):
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        t0 = time.time()
        png = await browse_page.remote.aio(url)
        return Response(content=png, media_type="image/png",
                        headers={"X-Elapsed-Seconds": f"{round(time.time()-t0,3)}"})

    # --- Modal-hosted MCP server (streamable HTTP transport) ---
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.utilities.types import Image

    mcp_server = FastMCP("pi-frontend-mcp")

    @mcp_server.tool()
    async def modal_ping() -> str:
        """Health-check the Modal frontend."""
        return "pi-frontend is up"

    @mcp_server.tool()
    async def modal_models(n: int = 3) -> str:
        """Fan out N parallel workers on Modal. Returns JSON with elapsed time and per-worker results."""
        n = max(1, min(n, 64))
        t0 = time.time()
        tasks = [run_model.remote.aio(f"model-{i}") for i in range(n)]
        results = await asyncio.gather(*tasks)
        import json
        return json.dumps({"concurrent": n, "elapsed": round(time.time()-t0, 3), "results": results})

    @mcp_server.tool()
    async def modal_browse(url: str) -> Image:
        """Open a URL in a headless Chromium browser inside a Modal container and return a PNG screenshot."""
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        png = await browse_page.remote.aio(url)
        return Image(data=png, format="png")

    api.mount("/mcp", mcp_server.streamable_http_app())

    return api


@app.function(image=base_image)
@modal.asgi_app()
def mcp_web():
    """Dedicated MCP-over-HTTP endpoint. Serves tools at /mcp."""
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.utilities.types import Image
    import asyncio, time, json

    mcp_server = FastMCP("pi-frontend-mcp")
    mcp_server.settings.transport_security.enable_dns_rebinding_protection = False
    mcp_server.settings.transport_security.allowed_hosts = ["*"]
    mcp_server.settings.transport_security.allowed_origins = ["*"]
    mcp_server.settings.stateless_http = True

    @mcp_server.tool()
    async def modal_ping() -> str:
        """Health-check the Modal frontend."""
        return "pi-frontend is up"

    @mcp_server.tool()
    async def modal_models(n: int = 3) -> str:
        """Fan out N parallel workers on Modal. Returns JSON with elapsed time and per-worker results."""
        n = max(1, min(n, 64))
        t0 = time.time()
        tasks = [run_model.remote.aio(f"model-{i}") for i in range(n)]
        results = await asyncio.gather(*tasks)
        return json.dumps({"concurrent": n, "elapsed": round(time.time()-t0, 3), "results": results})

    @mcp_server.tool()
    async def modal_browse(url: str) -> Image:
        """Open a URL in a headless Chromium browser inside a Modal container and return a PNG screenshot."""
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        png = await browse_page.remote.aio(url)
        return Image(data=png, format="png")

    return mcp_server.streamable_http_app()


@app.function(image=base_image)
def run_model(name: str):
    import time, os
    t0 = time.time()
    time.sleep(0.5)
    return {"name": name, "container": os.environ.get("MODAL_TASK_ID", "local"),
            "elapsed": round(time.time() - t0, 3)}


@app.function(image=pw_image, timeout=300)
async def browse_page(url: str) -> bytes:
    import asyncio
    from playwright.async_api import async_playwright

    async def run():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(500)
            png = await page.screenshot(full_page=False)
            await browser.close()
            return png

    return await run()
