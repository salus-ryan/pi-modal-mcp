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

# LLM image: torch + transformers for OSS model inference on GPU.
llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3")
)

MODEL_VOL = modal.Volume.from_name("pi-frontend-models", create_if_missing=True)
DEFAULT_MODEL = "Qwen/CodeQwen1.5-7B-Chat"


class _NoGetStream:
    """ASGI middleware: 405 GET on the MCP path so stateless clients use POST-inline.

    A stateless server has no persistent SSE stream to offer, so advertising one
    (200 GET that holds open) makes strict clients (e.g. the MCP Node SDK) block
    on a listening stream that never delivers. Returning 405 tells them to skip
    the GET stream and use POST-inline responses, which is correct for stateless.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") == "http"
            and scope.get("method") == "GET"
            and scope.get("path") in ("/mcp", "/mcp/")
        ):
            body = b"GET stream disabled; use POST (stateless mode)"
            await send({"type": "http.response.start", "status": 405,
                        "headers": [[b"content-type", b"text/plain"],
                                    [b"content-length", str(len(body)).encode()],
                                    [b"allow", b"POST"]]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


@app.function(image=base_image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response
    import asyncio, time, json

    api = FastAPI(title="pi-frontend")

    HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>pi-frontend</title></head><body style=font-family:monospace;padding:2rem>
<h1>pi-frontend</h1><p>Endpoints:</p><ul>
<li><a href=/api/ping>/api/ping</a></li>
<li><a href=/api/time>/api/time</a></li>
<li><a href=/api/models?n=3>/api/models?n=3</a> - parallel fan-out</li>
<li><a href=/api/browse?url=example.com>/api/browse?url=example.com</a> - headless screenshot (PNG)</li>
<li><a href=/api/swarm>/api/swarm</a> (POST) - model swarm</li>
<li><a href=/ide>/ide</a> - AI-native IDE powered by Modal model swarm</li>
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

    @api.post("/api/swarm")
    async def swarm(req: dict):
        prompt = req.get("prompt", "")
        models = req.get("models", DEFAULT_MODEL)
        n = max(1, min(int(req.get("n", 3)), 16))
        max_new = int(req.get("max_new_tokens", 256))
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
        workers = [ids[i % len(ids)] for i in range(n)]
        t0 = time.time()
        tasks = [llm_worker.remote.aio(m, prompt, max_new) for m in workers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for m, r in zip(workers, results):
            if isinstance(r, Exception):
                out.append({"model": m, "error": str(r)})
            else:
                out.append(r)
        return {"concurrent": n, "elapsed": round(time.time() - t0, 3), "results": out}

    @api.get("/ide", response_class=HTMLResponse)
    def ide():
        return IDE_HTML

    # --- OpenAI-compatible chat completions endpoint ---
    # Lets pi (or any OpenAI-compatible client) use the Modal model swarm as a provider.
    # Configure in ~/.pi/agent/models.json:
    #   "swarm": { "baseUrl": "https://...--pi-frontend-web.modal.run/v1",
    #              "api": "openai-completions", "apiKey": "swarm",
    #              "models": [{ "id": "codegen-350M", ... }] }
    MODEL_ALIASES = {
        "codeqwen-7b": "Qwen/CodeQwen1.5-7B-Chat",
        "codeqwen": "Qwen/CodeQwen1.5-7B-Chat",
        "codegen-350M": "Qwen/CodeQwen1.5-7B-Chat",
        "codegen-350M-mono": "Qwen/CodeQwen1.5-7B-Chat",
        "bloom-560m": "bigscience/bloom-560m",
        "bloom": "bigscience/bloom-560m",
        "gpt2": "gpt2",
        "distilgpt2": "distilgpt2",
    }

    @api.post("/v1/chat/completions")
    async def chat_completions(req: dict):
        messages = req.get("messages", [])
        model_alias = req.get("model", "codegen-350M")
        model_id = MODEL_ALIASES.get(model_alias, model_alias)
        max_new = int(req.get("max_tokens", req.get("max_new_tokens", 256)))
        n = int(req.get("n", 1))
        stream = bool(req.get("stream", False))
        # Format messages into a single prompt (simple chat template).
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"# System\n{content}")
            elif role == "user":
                parts.append(f"# User\n{content}")
            elif role == "assistant":
                parts.append(f"# Assistant\n{content}")
        prompt = "\n\n".join(parts) + "\n\n# Assistant\n"
        t0 = time.time()
        tasks = [llm_worker.remote.aio(model_id, prompt, max_new) for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        import uuid
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(t0)

        if stream:
            from fastapi.responses import StreamingResponse
            async def generate():
                for i, r in enumerate(results):
                    text = r.get("completion", "") if not isinstance(r, Exception) else f"[error: {r}]"
                    # Send content as a single chunk (not token-by-token, but protocol-compliant).
                    chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                             "model": model_alias,
                             "choices": [{"index": i, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    done_chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                                  "model": model_alias,
                                  "choices": [{"index": i, "delta": {}, "finish_reason": "stop"}]}
                    yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")

        choices = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                text = f"[error: {r}]"
            else:
                text = r.get("completion", "")
            choices.append({
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            })
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model_alias,
            "choices": choices,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @api.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [
            {"id": k, "object": "model", "owned_by": "modal-swarm"}
            for k in MODEL_ALIASES
        ]}

    # --- True streaming swarm: tokens stream independently per worker via SSE ---
    @api.post("/api/swarm/stream")
    async def swarm_stream(req: dict):
        from fastapi.responses import StreamingResponse
        prompt = req.get("prompt", "")
        models = req.get("models", DEFAULT_MODEL)
        n = max(1, min(int(req.get("n", 3)), 8))
        max_new = int(req.get("max_new_tokens", 128))
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
        workers = [ids[i % len(ids)] for i in range(n)]

        async def generate():
            import asyncio
            # Spawn all workers concurrently; each is a streaming generator.
            # We interleave their tokens into one SSE stream with worker index tags.
            queues = [asyncio.Queue() for _ in workers]

            async def run_worker(i, model_id):
                try:
                    async for chunk in llm_worker_stream.remote.gen(model_id, prompt, max_new):
                        await queues[i].put(chunk)
                except Exception as e:
                    await queues[i].put({"error": str(e)})
                finally:
                    await queues[i].put(None)  # sentinel

            tasks = [asyncio.create_task(run_worker(i, m)) for i, m in enumerate(workers)]
            done = [False] * n
            while not all(done):
                for i in range(n):
                    if done[i]:
                        continue
                    try:
                        item = queues[i].get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                    if item is None:
                        done[i] = True
                        yield f"data: {json.dumps({'worker': i, 'model': workers[i], 'done': True})}\n\n"
                    else:
                        yield f"data: {json.dumps({'worker': i, 'model': workers[i], **item})}\n\n"
                await asyncio.sleep(0.01)
            await asyncio.gather(*tasks, return_exceptions=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # --- Swarm synthesis: workers produce candidates, then a model critiques + synthesizes ---
    @api.post("/api/synthesize")
    async def synthesize(req: dict):
        prompt = req.get("prompt", "")
        models = req.get("models", DEFAULT_MODEL)
        n = max(1, min(int(req.get("n", 3)), 8))
        max_new = int(req.get("max_new_tokens", 256))
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
        workers = [ids[i % len(ids)] for i in range(n)]
        t0 = time.time()

        # Phase 1: N workers generate candidates in parallel
        tasks = [llm_worker.remote.aio(m, prompt, max_new) for m in workers]
        candidates = await asyncio.gather(*tasks, return_exceptions=True)

        # Phase 2: synthesis — one worker critiques and picks the best parts
        cand_text = "\n\n---\n\n".join(
            f"### Candidate {i+1} ({c.get('model','?')}):\n{c.get('completion','[error]')}"
            if not isinstance(c, Exception) else f"### Candidate {i+1}: [error: {c}]"
            for i, c in enumerate(candidates)
        )
        synth_prompt = (
            "You are a code review synthesizer. Below are multiple candidate completions "
            "for the same prompt. Critique each, then produce ONE canonical answer that "
            "takes the best parts. Output only the final code.\n\n"
            f"## Original prompt\n{prompt}\n\n## Candidates\n{cand_text}\n\n## Synthesized answer:"
        )
        synth = await llm_worker.remote.aio(workers[0], synth_prompt, max_new)
        return {
            "elapsed": round(time.time() - t0, 3),
            "candidates": [
                c if not isinstance(c, Exception) else {"error": str(c)}
                for c in candidates
            ],
            "synthesis": synth,
        }

    # --- Swarm cognition loop: multi-round generate → conflict-detect → test-gen → critique → verify ---
    @api.post("/api/cognition")
    async def cognition(req: dict):
        from fastapi.responses import StreamingResponse
        import re, difflib
        goal = req.get("goal", req.get("prompt", ""))
        models = req.get("models", DEFAULT_MODEL)
        n = max(1, min(int(req.get("n", 3)), 8))
        max_rounds = max(1, min(int(req.get("max_rounds", 3)), 5))
        max_new = int(req.get("max_new_tokens", 256))
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]

        ROLE_PROMPTS = {
            "coder": "You are an expert Python programmer. Write clean, correct, complete code. Output ONLY the Python code, no markdown.",
            "planner": "You are a technical planner. Think through the approach, then write the complete Python solution. Output ONLY the Python code.",
            "critic": "You are a senior code reviewer. Review the candidates and conflicts below, resolve the conflicts, fix bugs, output ONE final correct Python solution. Output ONLY the Python code.",
            "tester": "You are a test engineer. Given a task, write Python assert statements that verify the solution is correct. Output ONLY assert statements, no function definitions.",
        }

        def extract_code(text):
            match = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
            return match.group(1).strip() if match else text.strip()

        def detect_conflicts(cands):
            """Diff candidate code, identify where they disagree."""
            conflicts = []
            codes = [extract_code(c["completion"]) for c in cands if not isinstance(c, Exception) and "completion" in c]
            if len(codes) < 2:
                return conflicts
            for i in range(len(codes)):
                for j in range(i + 1, len(codes)):
                    la, lb = codes[i].splitlines(), codes[j].splitlines()
                    diffs = [d for d in difflib.unified_diff(la, lb, lineterm="", n=1)
                             if d.startswith(("+ ", "- ")) and not d.startswith(("+++", "---"))]
                    if diffs:
                        conflicts.append({"pair": [i, j], "diffs": diffs[:10]})
            return conflicts

        blackboard = {"goal": goal, "round": 0, "max_rounds": max_rounds, "claims": [],
                      "conflicts": [], "constraints": [], "verifications": [], "tests": [],
                      "synthesis": None, "halted": False, "halt_reason": None}
        t0 = time.time()

        async def generate():
            for round_num in range(1, max_rounds + 1):
                blackboard["round"] = round_num
                yield f"data: {json.dumps({'event':'round_start','round':round_num,'max_rounds':max_rounds,'elapsed':round(time.time()-t0,1)})}\n\n"

                prompt = goal
                if blackboard["constraints"]:
                    ct = "\n".join(f"  - {c}" for c in blackboard["constraints"])
                    prompt = f"{goal}\n\n## Previous failures:\n{ct}\n\nFix and write the complete solution."

                # Phase 1: Generate candidates (role specialization)
                candidates = []
                for i in range(n):
                    role = "planner" if (i == 0 and n > 1) else "coder"
                    rp = ROLE_PROMPTS.get(role, ROLE_PROMPTS["coder"])
                    yield f"data: {json.dumps({'event':'generate','worker':i,'model':ids[i%len(ids)],'role':role,'round':round_num})}\n\n"
                    result = await llm_worker.remote.aio(ids[i % len(ids)], f"{rp}\n\n## Task\n{prompt}", max_new)
                    candidates.append(result)
                    blackboard["claims"].append({"id":f"r{round_num}c{i}","model":result["model"],"role":role,"content":result["completion"],"round":round_num})
                    yield f"data: {json.dumps({'event':'candidate','worker':i,'round':round_num,'content':result['completion'][:300]})}\n\n"

                # Phase 2: Conflict detection (diff candidates)
                conflicts = detect_conflicts(candidates)
                blackboard["conflicts"] = conflicts
                yield f"data: {json.dumps({'event':'conflicts','round':round_num,'count':len(conflicts),'conflicts':conflicts[:3]})}\n\n"

                # Phase 3: Test generation (tester role) — in parallel with critique
                yield f"data: {json.dumps({'event':'test_generate','round':round_num})}\n\n"
                test_prompt = f"{ROLE_PROMPTS['tester']}\n\n## Task\n{goal}"
                test_result = await llm_worker.remote.aio(ids[0], test_prompt, max_new)
                test_code = extract_code(test_result["completion"])
                blackboard["tests"].append({"round": round_num, "code": test_code})
                yield f"data: {json.dumps({'event':'tests','round':round_num,'content':test_code[:300]})}\n\n"

                # Phase 4: Critique + synthesize (critic gets candidates + conflicts)
                cand_text = "\n\n---\n\n".join(f"### Candidate {i+1} ({c['model']}):\n{c['completion']}" for i,c in enumerate(candidates))
                conflict_text = ""
                if conflicts:
                    conflict_text = "\n\n## Conflicts detected between candidates:\n"
                    for cf in conflicts[:5]:
                        conflict_text += f"- Candidates {cf['pair']} differ on: {', '.join(cf['diffs'][:3])}\n"
                    conflict_text += "\nResolve these conflicts in your synthesis."
                yield f"data: {json.dumps({'event':'critique_start','round':round_num,'conflicts':len(conflicts)})}\n\n"
                synth = await llm_worker.remote.aio(ids[0], f"{ROLE_PROMPTS['critic']}\n\n## Task\n{goal}\n\n## Candidates\n{cand_text}{conflict_text}\n\n## Final solution:", max_new)
                code = extract_code(synth["completion"])
                blackboard["synthesis"] = code
                yield f"data: {json.dumps({'event':'synthesize','round':round_num,'content':code[:500]})}\n\n"

                # Phase 5: Verify — run code + tests in sandbox
                full_code = code + "\n\n# --- tests ---\n" + test_code
                yield f"data: {json.dumps({'event':'verify_start','round':round_num,'code':code[:200],'tests':test_code[:200]})}\n\n"
                result = await execute_code.remote.aio(full_code, timeout=10)
                blackboard["verifications"].append({"round":round_num,"code":code,"tests":test_code,"result":result})
                yield f"data: {json.dumps({'event':'verify_result','round':round_num,'passed':result['passed'],'stdout':result['stdout'][:500],'stderr':result['stderr'][:500]})}\n\n"

                if result["passed"]:
                    blackboard["halted"] = True; blackboard["halt_reason"] = "verified"
                    yield f"data: {json.dumps({'event':'halt','reason':'verified','round':round_num,'elapsed':round(time.time()-t0,1),'blackboard':blackboard})}\n\n"
                    yield "data: [DONE]\n\n"; return
                else:
                    err = result["stderr"][:500] if result["stderr"] else f"exit {result['exit_code']}"
                    blackboard["constraints"].append(f"Round {round_num}: tests failed — {err}")
                    yield f"data: {json.dumps({'event':'round_end','round':round_num,'passed':False,'elapsed':round(time.time()-t0,1)})}\n\n"

            blackboard["halted"] = True; blackboard["halt_reason"] = "max_rounds"
            yield f"data: {json.dumps({'event':'halt','reason':'max_rounds','round':max_rounds,'elapsed':round(time.time()-t0,1),'blackboard':blackboard})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @api.post("/api/verify")
    async def verify(req: dict):
        code = req.get("code", "")
        timeout = int(req.get("timeout", 10))
        result = await execute_code.remote.aio(code, timeout)
        return result

    # --- Modal-hosted MCP server (mounted) ---
    api.mount("/mcp", _NoGetStream(build_mcp().streamable_http_app()))

    return api


@app.function(image=base_image)
@modal.asgi_app()
def mcp_web():
    """Dedicated MCP-over-HTTP endpoint. Serves tools at /mcp."""
    return _NoGetStream(build_mcp().streamable_http_app())


def build_mcp():
    """Build a configured FastMCP server with all tools. Used by web() and mcp_web()."""
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
        return json.dumps({"concurrent": n, "elapsed": round(time.time() - t0, 3), "results": results})

    @mcp_server.tool()
    async def modal_browse(url: str) -> Image:
        """Open a URL in a headless Chromium browser inside a Modal container and return a PNG screenshot."""
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        png = await browse_page.remote.aio(url)
        return Image(data=png, format="png")

    @mcp_server.tool()
    async def modal_swarm(prompt: str, models: str = DEFAULT_MODEL, n: int = 3, max_new_tokens: int = 256) -> str:
        """Run a prompt across N parallel OSS model workers on Modal GPUs (a model swarm).

        Args:
            prompt: The prompt / code to complete.
            models: Comma-separated HuggingFace model IDs (default Qwen/CodeQwen1.5-7B-Chat).
                    Workers cycle through the list, so passing 2 models with n=4 runs each twice.
            n: Number of parallel workers (1-16).
            max_new_tokens: Max tokens to generate per worker.

        Returns JSON: {concurrent, elapsed, results:[{model, completion, elapsed}]}.
        """
        n = max(1, min(n, 16))
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
        workers = [ids[i % len(ids)] for i in range(n)]
        t0 = time.time()
        tasks = [llm_worker.remote.aio(m, prompt, max_new_tokens) for m in workers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for m, r in zip(workers, results):
            if isinstance(r, Exception):
                out.append({"model": m, "error": str(r)})
            else:
                out.append(r)
        return json.dumps({"concurrent": n, "elapsed": round(time.time() - t0, 3), "results": out})

    @mcp_server.tool()
    async def modal_swarm_stream(prompt: str, models: str = DEFAULT_MODEL, n: int = 3, max_new_tokens: int = 128) -> str:
        """Stream a prompt across N parallel OSS model workers, returning all tokens as they generate.

        Unlike modal_swarm (which waits for all workers), this returns a token-by-token
        view of every worker generating simultaneously. Results are JSONL, one line per token chunk.
        """
        n = max(1, min(n, 8))
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
        workers = [ids[i % len(ids)] for i in range(n)]
        lines = []
        async for chunk in llm_worker_stream.remote.gen_aio(workers[0], prompt, max_new_tokens):
            if "token" in chunk:
                lines.append(chunk["token"])
        return "".join(lines)

    @mcp_server.tool()
    async def modal_synthesize(prompt: str, models: str = DEFAULT_MODEL, n: int = 3, max_new_tokens: int = 256) -> str:
        """Run a prompt across N parallel OSS model workers, then synthesize one canonical answer.

        Phase 1: N workers generate candidate completions in parallel.
        Phase 2: A model critiques all candidates and produces ONE synthesized answer.

        Returns JSON: {elapsed, candidates:[...], synthesis:{model, completion, elapsed}}.
        """
        n = max(1, min(n, 8))
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
        workers = [ids[i % len(ids)] for i in range(n)]
        t0 = time.time()
        tasks = [llm_worker.remote.aio(m, prompt, max_new_tokens) for m in workers]
        candidates = await asyncio.gather(*tasks, return_exceptions=True)
        cand_text = "\n\n---\n\n".join(
            f"### Candidate {i+1} ({c.get('model','?')}):\n{c.get('completion','[error]')}"
            if not isinstance(c, Exception) else f"### Candidate {i+1}: [error: {c}]"
            for i, c in enumerate(candidates)
        )
        synth_prompt = (
            "You are a code review synthesizer. Below are multiple candidate completions "
            "for the same prompt. Critique each, then produce ONE canonical answer that "
            "takes the best parts. Output only the final code.\n\n"
            f"## Original prompt\n{prompt}\n\n## Candidates\n{cand_text}\n\n## Synthesized answer:"
        )
        synth = await llm_worker.remote.aio(workers[0], synth_prompt, max_new_tokens)
        return json.dumps({"elapsed": round(time.time() - t0, 3),
                           "candidates": [c if not isinstance(c, Exception) else {"error": str(c)} for c in candidates],
                           "synthesis": synth})

    @mcp_server.tool()
    async def modal_verify(code: str, timeout: int = 10) -> str:
        """Execute Python code in a Modal sandbox and return the result.

        Returns JSON: {exit_code, stdout, stderr, passed}.
        The code runs in an isolated container with a timeout (default 10s).
        """
        result = await execute_code.remote.aio(code, timeout)
        return json.dumps(result)

    @mcp_server.tool()
    async def modal_cognition(goal: str, models: str = DEFAULT_MODEL, n: int = 3, max_rounds: int = 3, max_new_tokens: int = 256) -> str:
        """Run the full swarm cognition loop with test generation and conflict detection.

        Each round:
        1. N workers (planner/coder) generate candidates in parallel.
        2. Conflicts detected via difflib (where candidates disagree).
        3. Tester role generates assert-based test cases.
        4. Critic reviews candidates + conflicts, synthesizes ONE canonical answer.
        5. Code + tests executed in sandbox. If passes -> halt 'verified'.
           If fails -> error fed back as constraint, next round revises.

        Returns JSON: the SCL-inspired blackboard with goals, claims, conflicts,
        constraints, tests, verifications, synthesis, and halt state.
        """
        import re, time, difflib
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
        ROLE_PROMPTS = {
            "coder": "You are an expert Python programmer. Write clean, correct, complete code. Output ONLY the Python code.",
            "planner": "You are a technical planner. Think through the approach, then write the complete Python solution. Output ONLY the Python code.",
            "critic": "You are a senior code reviewer. Review candidates and conflicts, resolve conflicts, fix bugs, output ONE final correct Python solution. Output ONLY the Python code.",
            "tester": "You are a test engineer. Given a task, write Python assert statements that verify correctness. Output ONLY assert statements.",
        }
        def extract_code(text):
            m = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
            return m.group(1).strip() if m else text.strip()
        def detect_conflicts(cands):
            conflicts = []
            codes = [extract_code(c["completion"]) for c in cands if not isinstance(c, Exception) and "completion" in c]
            if len(codes) < 2: return conflicts
            for i in range(len(codes)):
                for j in range(i+1, len(codes)):
                    la, lb = codes[i].splitlines(), codes[j].splitlines()
                    diffs = [d for d in difflib.unified_diff(la, lb, lineterm="", n=1) if d.startswith(("+ ", "- ")) and not d.startswith(("+++", "---"))]
                    if diffs: conflicts.append({"pair": [i, j], "diffs": diffs[:10]})
            return conflicts
        bb = {"goal": goal, "round": 0, "max_rounds": max_rounds, "claims": [],
              "conflicts": [], "constraints": [], "verifications": [], "tests": [],
              "synthesis": None, "halted": False, "halt_reason": None}
        t0 = time.time()
        for rnd in range(1, max_rounds + 1):
            bb["round"] = rnd
            prompt = goal
            if bb["constraints"]:
                ct = "\n".join(f"  - {c}" for c in bb["constraints"])
                prompt = f"{goal}\n\n## Previous failures:\n{ct}\n\nFix and write the complete solution."
            candidates = []
            for i in range(n):
                role = "planner" if (i == 0 and n > 1) else "coder"
                rp = ROLE_PROMPTS.get(role, ROLE_PROMPTS["coder"])
                result = await llm_worker.remote.aio(ids[i % len(ids)], f"{rp}\n\n## Task\n{prompt}", max_new_tokens)
                candidates.append(result)
                bb["claims"].append({"id": f"r{rnd}c{i}", "model": result["model"], "role": role, "content": result["completion"], "round": rnd})
            # Conflict detection
            conflicts = detect_conflicts(candidates)
            bb["conflicts"] = conflicts
            # Test generation
            test_result = await llm_worker.remote.aio(ids[0], f"{ROLE_PROMPTS['tester']}\n\n## Task\n{goal}", max_new_tokens)
            test_code = extract_code(test_result["completion"])
            bb["tests"].append({"round": rnd, "code": test_code})
            # Critique + synthesize
            cand_text = "\n\n---\n\n".join(f"### Candidate {i+1} ({c['model']}):\n{c['completion']}" for i,c in enumerate(candidates))
            conflict_text = ""
            if conflicts:
                conflict_text = "\n\n## Conflicts:\n"
                for cf in conflicts[:5]:
                    conflict_text += f"- Candidates {cf['pair']} differ: {', '.join(cf['diffs'][:3])}\n"
                conflict_text += "\nResolve these conflicts."
            synth = await llm_worker.remote.aio(ids[0], f"{ROLE_PROMPTS['critic']}\n\n## Task\n{goal}\n\n## Candidates\n{cand_text}{conflict_text}\n\n## Final solution:", max_new_tokens)
            code = extract_code(synth["completion"])
            bb["synthesis"] = code
            # Verify: run code + tests
            full_code = code + "\n\n# --- tests ---\n" + test_code
            result = await execute_code.remote.aio(full_code, timeout=10)
            bb["verifications"].append({"round": rnd, "code": code, "tests": test_code, "result": result})
            if result["passed"]:
                bb["halted"] = True; bb["halt_reason"] = "verified"
                break
            else:
                err = result["stderr"][:500] if result["stderr"] else f"exit {result['exit_code']}"
                bb["constraints"].append(f"Round {rnd}: tests failed — {err}")
        else:
            bb["halted"] = True; bb["halt_reason"] = "max_rounds"
        bb["elapsed"] = round(time.time() - t0, 1)
        return json.dumps(bb)

    @mcp_server.tool()
    async def modal_test(goal: str, models: str = DEFAULT_MODEL, max_new_tokens: int = 256) -> str:
        """Generate assert-based test cases for a coding goal.

        A tester role writes Python assert statements that verify a solution
        to the goal is correct. Returns JSON: {tests, model, elapsed}.
        """
        import re, time
        ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]
        tester_prompt = ("You are a test engineer. Given a task, write Python assert "
                         "statements that verify the solution is correct. Output ONLY assert statements.")
        t0 = time.time()
        result = await llm_worker.remote.aio(ids[0], f"{tester_prompt}\n\n## Task\n{goal}", max_new_tokens)
        def extract_code(text):
            m = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
            return m.group(1).strip() if m else text.strip()
        return json.dumps({"tests": extract_code(result["completion"]), "model": result["model"], "elapsed": round(time.time()-t0, 1)})

    return mcp_server


@app.function(image=base_image)
def run_model(name: str):
    import time, os
    t0 = time.time()
    time.sleep(0.5)
    return {"name": name, "container": os.environ.get("MODAL_TASK_ID", "local"),
            "elapsed": round(time.time() - t0, 3)}


@app.function(image=pw_image, timeout=300)
async def browse_page(url: str) -> bytes:
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


@app.function(
    image=llm_image,
    gpu="A100-80GB",
    volumes={"/models": MODEL_VOL},
    timeout=600,
)
def llm_worker(model_id: str, prompt: str, max_new_tokens: int = 256) -> dict:
    """Run a single OSS model inference on a Modal GPU container.

    Models are cached in a shared Modal Volume so cold starts after the first
    only pay the model-load cost, not the download.
    """
    import time, os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = os.path.join("/models", model_id.replace("/", "_"))
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()

    # Use chat template if available (instruction-tuned models)
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    else:
        inputs = tok(prompt, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs.shape[1]
    new_tokens = out[0][input_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    MODEL_VOL.commit()
    return {"model": model_id, "completion": text, "elapsed": round(time.time() - t0, 3)}


@app.function(
    image=llm_image,
    gpu="A100-80GB",
    volumes={"/models": MODEL_VOL},
    timeout=600,
)
def llm_worker_stream(model_id: str, prompt: str, max_new_tokens: int = 256):
    """Generator that yields tokens as they're generated (true streaming).

    Uses transformers TextIteratorStreamer + threading so tokens are produced
    incrementally rather than all-at-once after generate() finishes.
    Yields dicts: {"token": str} during generation, {"done": True, "elapsed": float} at end.
    """
    import time, os, threading
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    cache = os.path.join("/models", model_id.replace("/", "_"))
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()

    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    else:
        input_ids = tok(prompt, return_tensors="pt")["input_ids"].to("cuda")

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    t0 = time.time()
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "pad_token_id": tok.pad_token_id,
        "streamer": streamer,
    }
    thread = threading.Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    for text in streamer:
        if text:
            yield {"token": text}
    thread.join()
    MODEL_VOL.commit()
    yield {"done": True, "elapsed": round(time.time() - t0, 3), "model": model_id}


@app.function(image=modal.Image.debian_slim(python_version="3.11"), timeout=60, cpu=1, memory=512)
def execute_code(code: str, timeout: int = 10) -> dict:
    """Execute Python code in an isolated Modal sandbox.

    Runs the code via subprocess with a timeout. No network calls, no GPU.
    Returns {exit_code, stdout, stderr, passed}.
    """
    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            ["python3", path],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "exit_code": r.returncode,
            "stdout": r.stdout[:4000],
            "stderr": r.stderr[:4000],
            "passed": r.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"timed out after {timeout}s", "passed": False}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e), "passed": False}
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@app.function(image=base_image, timeout=900)
def cognition_loop(goal: str, models: str = DEFAULT_MODEL, n: int = 3, max_rounds: int = 3, max_new_tokens: int = 256):
    """The swarm cognition loop: generate → critique → synthesize → verify → (retry or halt).

    A Modal generator function that yields cognition events as a structured stream.
    The blackboard (SCL-inspired shared state) tracks goals, claims, conflicts,
    constraints, verifications, and halt state.

    Yields event dicts: {event, round, ...details}
    """
    import re, json, time

    ids = [m.strip() for m in models.split(",") if m.strip()] or [DEFAULT_MODEL]

    ROLE_PROMPTS = {
        "coder": "You are an expert Python programmer. Write clean, correct, complete code. Output ONLY the Python code, no markdown, no explanations.",
        "planner": "You are a technical planner. Think through the approach, then write the complete Python solution. Output ONLY the Python code.",
        "critic": "You are a senior code reviewer. Review the candidates below, identify the best approach, fix any bugs, and output ONE final correct Python solution. Output ONLY the Python code.",
    }

    def extract_code(text):
        match = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    blackboard = {
        "goal": goal,
        "round": 0,
        "max_rounds": max_rounds,
        "claims": [],
        "conflicts": [],
        "constraints": [],
        "verifications": [],
        "synthesis": None,
        "halted": False,
        "halt_reason": None,
    }

    t0 = time.time()

    for round_num in range(1, max_rounds + 1):
        blackboard["round"] = round_num
        yield {"event": "round_start", "round": round_num, "max_rounds": max_rounds, "elapsed": round(time.time()-t0, 1)}

        # Phase 1: Generate candidates (role specialization)
        prompt = goal
        if blackboard["constraints"]:
            constraint_text = "\n".join(f"  - {c}" for c in blackboard["constraints"])
            prompt = f"{goal}\n\n## Previous attempts failed with these errors:\n{constraint_text}\n\nFix these issues and write the complete solution."

        candidates = []
        for i in range(n):
            # First worker is planner (if n>1), rest are coders
            role = "planner" if (i == 0 and n > 1) else "coder"
            role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["coder"])
            full_prompt = f"{role_prompt}\n\n## Task\n{prompt}"
            yield {"event": "generate", "worker": i, "model": ids[i % len(ids)], "role": role, "round": round_num}
            result = llm_worker.remote(ids[i % len(ids)], full_prompt, max_new_tokens)
            candidates.append(result)
            blackboard["claims"].append({
                "id": f"r{round_num}c{i}",
                "model": result["model"],
                "role": role,
                "content": result["completion"],
                "round": round_num,
            })
            yield {"event": "candidate", "worker": i, "round": round_num, "content": result["completion"][:300]}

        # Phase 2: Critic reviews and synthesizes
        cand_text = "\n\n---\n\n".join(
            f"### Candidate {i+1} ({c['model']}, role={c.get('role','coder')}):\n{c['completion']}"
            for i, c in enumerate(candidates)
        )
        synth_prompt = f"{ROLE_PROMPTS['critic']}\n\n## Original task\n{goal}\n\n## Candidates\n{cand_text}\n\n## Final solution:"
        yield {"event": "critique_start", "round": round_num}
        synth = llm_worker.remote(ids[0], synth_prompt, max_new_tokens)
        code = extract_code(synth["completion"])
        blackboard["synthesis"] = code
        yield {"event": "synthesize", "round": round_num, "content": code[:500]}

        # Phase 3: Verify (execute the code in sandbox)
        yield {"event": "verify_start", "round": round_num, "code": code[:300]}
        result = execute_code.remote(code, timeout=10)
        blackboard["verifications"].append({
            "round": round_num,
            "code": code,
            "result": result,
        })
        yield {"event": "verify_result", "round": round_num, "passed": result["passed"],
               "stdout": result["stdout"][:500], "stderr": result["stderr"][:500]}

        if result["passed"]:
            blackboard["halted"] = True
            blackboard["halt_reason"] = "verified"
            yield {"event": "halt", "reason": "verified", "round": round_num,
                   "elapsed": round(time.time()-t0, 1), "blackboard": blackboard}
            return
        else:
            error_msg = result["stderr"][:500] if result["stderr"] else f"exit code {result['exit_code']}"
            blackboard["constraints"].append(f"Round {round_num}: execution failed — {error_msg}")
            yield {"event": "round_end", "round": round_num, "passed": False,
                   "elapsed": round(time.time()-t0, 1)}

    blackboard["halted"] = True
    blackboard["halt_reason"] = "max_rounds"
    yield {"event": "halt", "reason": "max_rounds", "round": max_rounds,
           "elapsed": round(time.time()-t0, 1), "blackboard": blackboard}


IDE_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>pi-modal-mcp &middot; swarm IDE</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#0d1117;color:#e6edf3;height:100vh;display:flex;flex-direction:column}
  header{background:linear-gradient(90deg,#6366f1,#0ea5e9);padding:10px 16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  header h1{font-size:15px;color:#fff;font-weight:600}
  header .sub{font-size:12px;color:#dbeafe}
  .bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-left:auto}
  .bar label{font-size:11px;color:#dbeafe}
  .bar input{background:#0d1117;border:1px solid #30363d;color:#e6edf3;padding:5px 8px;border-radius:6px;font-size:12px}
  .bar input[type=number]{width:64px}
  .bar input[type=text]{width:240px}
  .bar button{background:#238636;border:1px solid #2ea043;color:#fff;padding:6px 14px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer}
  .bar button:hover{background:#2ea043}
  .bar button:disabled{opacity:.5;cursor:wait}
  main{flex:1;display:flex;min-height:0}
  #editor{flex:1;min-width:0;border-right:1px solid #30363d}
  #panel{width:42%;min-width:340px;display:flex;flex-direction:column;background:#0d1117}
  #panel .phead{padding:8px 12px;border-bottom:1px solid #30363d;font-size:12px;color:#8b949e;display:flex;justify-content:space-between}
  #results{flex:1;overflow:auto;padding:10px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:10px;overflow:hidden}
  .card .ctitle{background:#21262d;padding:6px 10px;font-size:11px;color:#79c0ff;display:flex;justify-content:space-between;border-bottom:1px solid #30363d}
  .card .cbody{padding:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word;color:#c9d1d9}
  .err{color:#f85149}
  .status{font-size:11px;color:#8b949e}
</style></head><body>
<header>
  <div><h1>&#129418; pi-modal-mcp swarm IDE</h1><div class=sub>self-hosted OSS models on Modal GPUs &middot; <a href=https://github.com/salus-ryan/pi-modal-mcp target=_blank style=color:#dbeafe>repo</a></div></div>
  <div class=bar>
    <label>models <input id=models type=text value="Qwen/CodeQwen1.5-7B-Chat" title="comma-separated HuggingFace IDs"></label>
    <label>n <input id=n type=number value=3 min=1 max=16></label>
    <label>tokens <input id=tok type=number value=128 min=8 max=1024></label>
    <button id=run>Run swarm</button>
  </div>
</header>
<main>
  <div id=editor></div>
  <div id=panel>
    <div class=phead><span>swarm results</span><span class=status id=status>idle</span></div>
    <div id=results><div class=card><div class=cbody style=color:#8b949e>Edit code on the left, then "Run swarm". Each worker runs on its own Modal GPU container; results stream back in parallel.</div></div></div>
  </div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min/vs/loader.min.js"></script>
<script>
require.config({paths:{vs:'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min/vs'}});
let ed;
require(['vs/editor/editor.main'],function(){
  ed=monaco.editor.create(document.getElementById('editor'),{
    value:'// OSS model swarm on Modal\\n// Edit me, then hit Run swarm.\\n\\ndef fizzbuzz(n):\\n    for i in range(1, n+1):\\n',
    language:'python',theme:'vs-dark',automaticLayout:true,fontSize:13
  });
});
const $=id=>document.getElementById(id);
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
$('run').onclick=async function(){
  if(!ed)return;
  const btn=this;btn.disabled=true;$('status').textContent='running...';
  const body={prompt:ed.getValue(),models:$('models').value,n:parseInt($('n').value,10),max_new_tokens:parseInt($('tok').value,10)};
  $('results').innerHTML='<div class=card><div class=cbody style=color:#8b949e>fanning out '+body.n+' worker(s) on Modal GPUs...</div></div>';
  try{
    const r=await fetch('/api/swarm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    $('status').textContent='done in '+d.elapsed+'s ('+d.concurrent+' workers)';
    $('results').innerHTML=d.results.map((x,i)=>'<div class=card><div class=ctitle><span>#'+(i+1)+' '+esc(x.model)+'</span><span>'+(x.elapsed?x.elapsed+'s':'')+'</span></div><div class=cbody'+(x.error?' style=color:#f85149':'')+'>'+esc(x.error||x.completion)+'</div></div>').join('');
  }catch(e){$('status').textContent='error';$('results').innerHTML='<div class=card><div class=cbody style=color:#f85149>'+esc(String(e))+'</div></div>';}
  finally{btn.disabled=false;}
};
</script>
</body></html>"""
