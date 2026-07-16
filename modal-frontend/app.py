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
COALITION_VOL = modal.Volume.from_name("pi-frontend-coalitions", create_if_missing=True)
DELTA_VOL = modal.Volume.from_name("pi-frontend-deltas", create_if_missing=True)
WORKSPACE_VOL = modal.Volume.from_name("pi-frontend-workspace", create_if_missing=True)
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
        "cognition": "Qwen/CodeQwen1.5-7B-Chat",
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

        # Cognition mode: if model is 'cognition', run the full 7-phase loop
        if model_alias == "cognition":
            import re as _re
            def _msg_content(msg):
                c = msg.get("content", "")
                if isinstance(c, list):
                    return " ".join(b.get("text","") for b in c if isinstance(b,dict) and b.get("type")=="text")
                return c
            goal = "\n".join(_msg_content(m) for m in messages if m.get("role") == "user")
            t0 = time.time()
            # Run the cognition loop inline (simplified non-streaming version)
            ids_cog = [model_id]
            ROLE = {
                "coder": "You are an expert Python programmer. Write clean, correct, complete code. Output ONLY the Python code.",
                "critic": "You are a senior code reviewer. Review candidates, fix bugs, output ONE final correct Python solution. Output ONLY the Python code.",
                "tester": "You are a test engineer. Write assert statements that verify correctness. Output ONLY assert statements.",
                "security": "You are a security engineer. Review for vulnerabilities. If safe, output unchanged. If unsafe, output the fixed version. Output ONLY the Python code.",
            }
            def _extract(text):
                m = _re.search(r'```(?:python)?\s*\n?(.*?)```', text, _re.DOTALL)
                return m.group(1).strip() if m else text.strip()
            # Phase 1: generate candidates
            n_cog = max(1, min(int(req.get("n", 2)), 4))
            max_new_cog = int(req.get("max_tokens", 256))
            cog_tasks = [llm_worker.remote.aio(model_id, f"{ROLE['coder']}\n\n## Task\n{goal}", max_new_cog) for _ in range(n_cog)]
            cands = await asyncio.gather(*cog_tasks, return_exceptions=True)
            cand_text = "\n\n---\n\n".join(f"### Candidate {i+1}:\n{c.get('completion','[error]')}" if not isinstance(c,Exception) else f"### Candidate {i+1}: [error]" for i,c in enumerate(cands))
            # Phase 2: critique + synthesize
            synth = await llm_worker.remote.aio(model_id, f"{ROLE['critic']}\n\n## Task\n{goal}\n\n## Candidates\n{cand_text}\n\n## Final solution:", max_new_cog)
            code = _extract(synth.get("completion",""))
            # Phase 3: test generation
            tests = await llm_worker.remote.aio(model_id, f"{ROLE['tester']}\n\n## Task\n{goal}", max_new_cog)
            test_code = _extract(tests.get("completion",""))
            # Phase 4: security review
            sec = await llm_worker.remote.aio(model_id, f"{ROLE['security']}\n\n## Code\n{code}", max_new_cog)
            code = _extract(sec.get("completion",code))
            # Phase 5: verify
            full_code = code + "\n\n# --- tests ---\n" + test_code
            result = await execute_code.remote.aio(full_code, timeout=10)
            elapsed = round(time.time()-t0, 1)
            content = json.dumps({"code": code, "tests": test_code, "verified": result["passed"],
                                  "stdout": result["stdout"][:200], "stderr": result["stderr"][:200],
                                  "elapsed": elapsed, "candidates": n_cog})
            if stream:
                from fastapi.responses import StreamingResponse
                async def gen():
                    chunk = {"id":"cog","object":"chat.completion.chunk","created":int(t0),"model":"cognition",
                             "choices":[{"index":0,"delta":{"role":"assistant","content":content},"finish_reason":None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield f"data: {json.dumps({'id':'cog','object':'chat.completion.chunk','created':int(t0),'model':'cognition','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(gen(), media_type="text/event-stream")
            return {"id":"cog","object":"chat.completion","created":int(t0),"model":"cognition",
                    "choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

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

        # Adaptive coalition selection: if model_pool provided, select best coalition
        model_pool = req.get("model_pool", "")
        adaptive = bool(req.get("adaptive", False))
        coalition_selected = None
        if adaptive and model_pool:
            selection = await coalition_store.remote.aio("select", pool=model_pool, n=n)
            ids = [m.strip() for m in selection["coalition"].split(",") if m.strip()]
            coalition_selected = selection
            yield_init = f"data: {json.dumps({'event':'coalition_selected','coalition':selection['coalition'],'explored':selection['explored'],'total_possible':selection['total_possible']})}\n\n"
        else:
            yield_init = None

        # Workspace integration: read file context, write results back
        workspace_path = req.get("workspace_path", "")
        workspace_files = req.get("workspace_files", "")  # comma-separated multi-file context
        workspace_context = ""
        ws_files_read = []
        if workspace_files:
            for fp in [f.strip() for f in workspace_files.split(",") if f.strip()]:
                try:
                    ws_result = await workspace.remote.aio("read", path=fp)
                    if "content" in ws_result:
                        workspace_context += f"\n\n## Current code in {fp}:\n```python\n{ws_result['content']}```\n"
                        ws_files_read.append(fp)
                except Exception:
                    pass
        elif workspace_path:
            try:
                ws_result = await workspace.remote.aio("read", path=workspace_path)
                if "content" in ws_result:
                    workspace_context = f"\n\n## Current code in {workspace_path}:\n```python\n{ws_result['content']}```\n"
                    ws_files_read.append(workspace_path)
            except Exception:
                pass
        write_path = workspace_path or (ws_files_read[0] if ws_files_read else "")

        # Frontier model config (speculative verification)
        frontier_base = req.get("frontier_base_url") or req.get("frontier_api_base", "")
        frontier_key = req.get("frontier_api_key", "")
        frontier_model = req.get("frontier_model", "")

        ROLE_PROMPTS = {
            "coder": "You are an expert Python programmer. Write clean, correct, complete code. Output ONLY the Python code, no markdown.",
            "planner": "You are a technical planner. Think through the approach, then write the complete Python solution. Output ONLY the Python code.",
            "critic": "You are a senior code reviewer. Review the candidates and conflicts below, resolve the conflicts, fix bugs, output ONE final correct Python solution. Output ONLY the Python code.",
            "tester": "You are a test engineer. Given a task, write Python assert statements that verify the solution is correct. Output ONLY assert statements, no function definitions.",
            "security": "You are a security engineer. Review the code for injection, path traversal, unsafe eval/exec, and other vulnerabilities. If safe, output it unchanged. If unsafe, output the fixed version. Output ONLY the Python code.",
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

        def format_blackboard(bb):
            """Format the blackboard as SCL context for role prompts."""
            lines = [f"## Blackboard (SCL shared state)"]
            lines.append(f"Goal: {bb['goal'][:200]}")
            lines.append(f"Round: {bb['round']}/{bb['max_rounds']}")
            if bb.get("constraints"):
                lines.append("Constraints (past failures):")
                for c in bb["constraints"][-3:]:
                    lines.append(f"  - {c[:150]}")
            if bb.get("synthesis"):
                lines.append(f"Previous synthesis: {bb['synthesis'][:150]}")
            if bb.get("delta"):
                d = bb["delta"]
                lines.append(f"Frontier delta: {'corrected' if d.get('changed') else 'accepted'}")
            return "\n".join(lines)

        blackboard = {"goal": goal, "round": 0, "max_rounds": max_rounds, "claims": [],
                      "conflicts": [], "constraints": [], "verifications": [], "tests": [],
                      "synthesis": None, "delta": None, "halted": False, "halt_reason": None}
        t0 = time.time()

        async def generate():
            if coalition_selected:
                yield f"data: {json.dumps({'event':'coalition_selected','coalition':coalition_selected['coalition'],'explored':coalition_selected['explored'],'total_possible':coalition_selected['total_possible']})}\n\n"
            for round_num in range(1, max_rounds + 1):
                blackboard["round"] = round_num
                yield f"data: {json.dumps({'event':'round_start','round':round_num,'max_rounds':max_rounds,'elapsed':round(time.time()-t0,1)})}\n\n"

                prompt = goal + workspace_context
                if blackboard["constraints"]:
                    ct = "\n".join(f"  - {c}" for c in blackboard["constraints"])
                    prompt = f"{goal}\n\n## Previous failures:\n{ct}\n\nFix and write the complete solution."
                # Include SCL blackboard context so roles know the full state
                scl_context = format_blackboard(blackboard) if round_num > 1 else ""

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

                # Phase 5: Frontier verification (speculative decoding at semantic action level)
                if frontier_base and frontier_key and frontier_model:
                    yield f"data: {json.dumps({'event':'frontier_verify_start','round':round_num,'model':frontier_model})}\n\n"
                    frontier_result = await frontier_verify.remote.aio(
                        frontier_base, frontier_key, frontier_model,
                        "You are a code verifier. Review the solution below. If correct, output it unchanged. If incorrect, output the corrected version. Output ONLY the Python code.",
                        f"## Task\n{goal}\n\n## Proposed solution\n{code}"
                    )
                    frontier_code = extract_code(frontier_result["content"])
                    if frontier_code != code:
                        blackboard["delta"] = {"round": round_num, "changed": True, "original": code, "corrected": frontier_code}
                        await delta_store.remote.aio("record", record={
                            "goal": goal, "coalition": coalition_selected["coalition"] if coalition_selected else ",".join(ids),
                            "round": round_num, "proposal": code, "correction": frontier_code,
                            "delta_changed": True, "frontier_model": frontier_model})
                        code = frontier_code  # use the frontier-corrected version
                        yield f"data: {json.dumps({'event':'frontier_corrected','round':round_num,'model':frontier_model,'delta':True})}\n\n"
                    else:
                        blackboard["delta"] = {"round": round_num, "changed": False}
                        await delta_store.remote.aio("record", record={
                            "goal": goal, "coalition": coalition_selected["coalition"] if coalition_selected else ",".join(ids),
                            "round": round_num, "proposal": code, "correction": code,
                            "delta_changed": False, "frontier_model": frontier_model})
                        yield f"data: {json.dumps({'event':'frontier_accepted','round':round_num,'model':frontier_model,'delta':False})}\n\n"

                # Phase 6: Security review (security role checks for vulnerabilities)
                yield f"data: {json.dumps({'event':'security_review_start','round':round_num})}\n\n"
                sec_result = await llm_worker.remote.aio(ids[0], f"{ROLE_PROMPTS['security']}\n\n## Code to review\n{code}", max_new)
                sec_code = extract_code(sec_result["completion"])
                if sec_code != code:
                    blackboard["security_correction"] = {"round": round_num, "changed": True, "original": code[:200], "corrected": sec_code[:200]}
                    code = sec_code
                    yield f"data: {json.dumps({'event':'security_corrected','round':round_num})}\n\n"
                else:
                    blackboard["security_correction"] = {"round": round_num, "changed": False}
                    yield f"data: {json.dumps({'event':'security_passed','round':round_num})}\n\n"

                # Phase 7: Verify — run code + tests in sandbox
                full_code = code + "\n\n# --- tests ---\n" + test_code
                yield f"data: {json.dumps({'event':'verify_start','round':round_num,'code':code[:200],'tests':test_code[:200]})}\n\n"
                result = await execute_code.remote.aio(full_code, timeout=10)
                blackboard["verifications"].append({"round":round_num,"code":code,"tests":test_code,"result":result})
                yield f"data: {json.dumps({'event':'verify_result','round':round_num,'passed':result['passed'],'stdout':result['stdout'][:500],'stderr':result['stderr'][:500]})}\n\n"

                if result["passed"]:
                    blackboard["halted"] = True; blackboard["halt_reason"] = "verified"
                    # Write corrected code back to workspace
                    if write_path:
                        try:
                            await workspace.remote.aio("snapshot", path=write_path)
                            await workspace.remote.aio("write", path=write_path, content=code)
                            yield f"data: {json.dumps({'event':'workspace_applied','round':round_num,'path':write_path})}\n\n"
                        except Exception as e:
                            yield f"data: {json.dumps({'event':'workspace_error','round':round_num,'error':str(e)})}\n\n"
                    if coalition_selected:
                        await coalition_store.remote.aio("record", coalition=coalition_selected["coalition"],
                            result={"delta_changed": (blackboard.get("delta") or {}).get("changed",False),"rounds":round_num,"elapsed":round(time.time()-t0,1)})
                    yield f"data: {json.dumps({'event':'halt','reason':'verified','round':round_num,'elapsed':round(time.time()-t0,1),'blackboard':blackboard})}\n\n"
                    yield "data: [DONE]\n\n"; return
                else:
                    err = result["stderr"][:500] if result["stderr"] else f"exit {result['exit_code']}"
                    blackboard["constraints"].append(f"Round {round_num}: tests failed — {err}")
                    yield f"data: {json.dumps({'event':'round_end','round':round_num,'passed':False,'elapsed':round(time.time()-t0,1)})}\n\n"

            blackboard["halted"] = True; blackboard["halt_reason"] = "max_rounds"
            if coalition_selected:
                await coalition_store.remote.aio("record", coalition=coalition_selected["coalition"],
                    result={"delta_changed": (blackboard.get("delta") or {}).get("changed",False),"rounds":max_rounds,"elapsed":round(time.time()-t0,1)})
            yield f"data: {json.dumps({'event':'halt','reason':'max_rounds','round':max_rounds,'elapsed':round(time.time()-t0,1),'blackboard':blackboard})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @api.post("/api/verify")
    async def verify(req: dict):
        code = req.get("code", "")
        timeout = int(req.get("timeout", 10))
        result = await execute_code.remote.aio(code, timeout)
        return result

    # --- Coalition stats + adaptive selection ---
    @api.get("/api/coalition")
    async def coalition_get():
        return await coalition_store.remote.aio("read")

    @api.post("/api/coalition/select")
    async def coalition_select(req: dict):
        pool = req.get("pool", DEFAULT_MODEL)
        n = int(req.get("n", 3))
        return await coalition_store.remote.aio("select", pool=pool, n=n)

    # --- Delta accumulation (learning dataset) ---
    @api.get("/api/deltas")
    async def deltas_stats():
        return await delta_store.remote.aio("stats")

    @api.post("/api/deltas/read")
    async def deltas_read(req: dict):
        limit = int(req.get("limit", 50))
        return await delta_store.remote.aio("read", limit=limit)

    @api.get("/api/deltas/export")
    async def deltas_export():
        return await delta_store.remote.aio("export")

    # --- Repository workspace ---
    @api.get("/api/workspace")
    async def workspace_list():
        return await workspace.remote.aio("list")

    @api.post("/api/workspace/read")
    async def workspace_read(req: dict):
        return await workspace.remote.aio("read", path=req.get("path",""))

    @api.post("/api/workspace/write")
    async def workspace_write(req: dict):
        return await workspace.remote.aio("write", path=req.get("path",""), content=req.get("content",""))

    @api.post("/api/workspace/diff")
    async def workspace_diff(req: dict):
        return await workspace.remote.aio("diff", file1=req.get("file1",""), file2=req.get("file2",""))

    @api.post("/api/workspace/snapshot")
    async def workspace_snapshot(req: dict):
        return await workspace.remote.aio("snapshot", path=req.get("path",""))

    @api.post("/api/workspace/revert")
    async def workspace_revert(req: dict):
        return await workspace.remote.aio("revert", path=req.get("path",""))

    @api.post("/api/workspace/git/init")
    async def workspace_git_init():
        return await workspace.remote.aio("git_init")

    @api.post("/api/workspace/git/commit")
    async def workspace_git_commit(req: dict):
        return await workspace.remote.aio("git_commit", record=req)

    @api.get("/api/workspace/git/diff")
    async def workspace_git_diff():
        return await workspace.remote.aio("git_diff")

    @api.get("/api/workspace/git/log")
    async def workspace_git_log():
        return await workspace.remote.aio("git_log")

    @api.get("/api/workspace/git/status")
    async def workspace_git_status():
        return await workspace.remote.aio("git_status")

    # --- System dashboard: full state in one call ---
    @api.get("/api/status")
    async def system_status():
        delta_stats = await delta_store.remote.aio("stats")
        coalition_stats = await coalition_store.remote.aio("read")
        ws_files = await workspace.remote.aio("list")
        git_log = await workspace.remote.aio("git_log")
        return {
            "version": "1.5.0",
            "models": list(MODEL_ALIASES.keys()),
            "default_model": DEFAULT_MODEL,
            "cognition_phases": ["generate", "conflict_detect", "test_generate", "critique", "frontier_verify", "security_review", "sandbox_verify"],
            "roles": ["planner", "coder", "critic", "tester", "security"],
            "mcp_tools": 12,
            "delta_stats": delta_stats,
            "coalition_stats": coalition_stats,
            "workspace_files": ws_files.get("files", []),
            "git_log": git_log.get("log", ""),
            "endpoints": ["/api/ping", "/api/swarm", "/api/swarm/stream", "/api/synthesize",
                          "/api/cognition", "/api/verify", "/api/coalition", "/api/coalition/select",
                          "/api/deltas", "/api/deltas/read", "/api/workspace", "/api/workspace/git/*",
                          "/v1/chat/completions", "/v1/models", "/ide", "/api/status"],
        }

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
    async def modal_cognition(goal: str, models: str = DEFAULT_MODEL, n: int = 3, max_rounds: int = 3, max_new_tokens: int = 256, frontier_base_url: str = "", frontier_api_key: str = "", frontier_model: str = "") -> str:
        """Run the full swarm cognition loop with test generation, conflict detection, and frontier verification.

        Speculative decoding at the semantic action level: the cheap OSS swarm
        proposes; a frontier model (if configured) verifies and corrects.
        The correction delta is the learning signal.

        Each round:
        1. N workers (planner/coder) generate candidates in parallel.
        2. Conflicts detected via difflib.
        3. Tester role generates assert-based test cases.
        4. Critic reviews candidates + conflicts, synthesizes ONE answer.
        5. Frontier model (if configured) verifies/corrects the synthesis.
        6. Code + tests executed in sandbox. If passes -> halt 'verified'.

        Returns JSON: the SCL-inspired blackboard with goals, claims, conflicts,
        constraints, tests, verifications, synthesis, delta, and halt state.
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
              "synthesis": None, "delta": None, "halted": False, "halt_reason": None}
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
            # Frontier verification (speculative decoding at semantic action level)
            if frontier_base_url and frontier_api_key and frontier_model:
                fr = await frontier_verify.remote.aio(
                    frontier_base_url, frontier_api_key, frontier_model,
                    "You are a code verifier. Review the solution. If correct, output unchanged. If incorrect, output the corrected version. Output ONLY Python code.",
                    f"## Task\n{goal}\n\n## Proposed solution\n{code}")
                frontier_code = extract_code(fr["content"])
                if frontier_code != code:
                    bb["delta"] = {"round": rnd, "changed": True, "original": code, "corrected": frontier_code}
                    await delta_store.remote.aio("record", record={
                        "goal": goal, "coalition": ",".join(ids), "round": rnd,
                        "proposal": code, "correction": frontier_code,
                        "delta_changed": True, "frontier_model": frontier_model})
                    code = frontier_code
                else:
                    bb["delta"] = {"round": rnd, "changed": False}
                    await delta_store.remote.aio("record", record={
                        "goal": goal, "coalition": ",".join(ids), "round": rnd,
                        "proposal": code, "correction": code,
                        "delta_changed": False, "frontier_model": frontier_model})
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
    async def modal_coalition(action: str = "read", pool: str = "", n: int = 3) -> str:
        """Read coalition performance stats or select the best coalition adaptively.

        Actions:
        - 'read': return all coalition stats (acceptance rates, rounds, elapsed)
        - 'select': given a model pool (comma-separated), use epsilon-greedy to select
          the best-known coalition or explore a new one. Returns {coalition, models, explored, total_possible, stats}.

        The system tracks how often the frontier accepts vs corrects each coalition.
        Over time, it converges on the coalition that best approximates the frontier.
        """
        result = await coalition_store.remote.aio(action, pool=pool, n=n)
        return json.dumps(result)

    @mcp_server.tool()
    async def modal_deltas(action: str = "stats", limit: int = 50) -> str:
        """Read the persistent delta (learning) dataset.

        Actions:
        - 'stats': return aggregate stats (total, corrected, accepted, correction_rate, by_coalition)
        - 'read': return the most recent delta entries (swarm_proposal vs frontier_correction pairs)
        - 'export': return preference pairs (goal, proposal, correction) for fine-tuning

        This is the learning signal: every time the frontier model corrects the swarm,
        the (proposal, correction) pair is recorded. Over time, this builds a preference
        dataset for fine-tuning. The 'export' action formats these as training pairs.
        """
        result = await delta_store.remote.aio(action, limit=limit)
        return json.dumps(result)

    @mcp_server.tool()
    async def modal_workspace(action: str, path: str = "", content: str = "", file1: str = "", file2: str = "") -> str:
        """Operate on a persistent repository workspace (file tree on a Modal volume).

        Actions:
        - 'list': list all files in the workspace
        - 'read': read a file by path
        - 'write': write content to a file path
        - 'delete': delete a file
        - 'diff': diff two files (file1 vs file2)
        - 'snapshot': save a .prev version of a file (for revert)
        - 'revert': revert a file to its .prev snapshot
        - 'git_init': initialize a git repo in the workspace
        - 'git_commit': stage all changes and commit (record.message for message)
        - 'git_diff': show uncommitted changes vs HEAD
        - 'git_log': show recent commit log
        - 'git_status': show working tree status

        This lets the cognition loop operate on real files (tree, diff, apply/revert)
        instead of just isolated code snippets.
        """
        result = await workspace.remote.aio(action, path=path, content=content, file1=file1, file2=file2)
        return json.dumps(result)

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


@app.function(image=base_image, timeout=120)
async def frontier_verify(api_base: str, api_key: str, model: str, system_prompt: str, user_prompt: str) -> dict:
    """Call a frontier model via an OpenAI-compatible API to verify/correct the swarm's synthesis.

    This is the 'speculative decoding at the semantic action level' pattern:
    the cheap swarm proposes; the frontier model verifies and corrects.
    The correction delta is the learning signal.
    """
    import urllib.request, json
    url = f"{api_base.rstrip('/')}/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://pi-modal-mcp.modal.run",
        "X-Title": "pi-modal-mcp",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read())
    return {"model": model, "content": result["choices"][0]["message"]["content"]}


@app.function(image=base_image, volumes={"/coalitions": COALITION_VOL}, timeout=60)
def coalition_store(action: str, coalition: str = "", result: dict = None, pool: str = "", n: int = 3):
    """Read/record/select coalition performance stats (adaptive coalition selection).

    Persistent store on a Modal Volume. Tracks per-coalition frontier acceptance
    rates, rounds, and elapsed time. Uses epsilon-greedy for exploration vs exploitation.
    """
    import json, os, random
    from itertools import combinations

    stats_path = "/coalitions/stats.json"
    os.makedirs("/coalitions", exist_ok=True)

    def load():
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                return json.load(f)
        return {}

    def save(stats):
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        COALITION_VOL.commit()

    if action == "read":
        return load()

    elif action == "record":
        stats = load()
        if coalition not in stats:
            stats[coalition] = {"runs": 0, "accepted": 0, "corrected": 0,
                                "total_rounds": 0, "total_elapsed": 0}
        s = stats[coalition]
        s["runs"] += 1
        changed = result.get("delta_changed", False) if result else False
        if changed:
            s["corrected"] += 1
        else:
            s["accepted"] += 1
        s["acceptance_rate"] = round(s["accepted"] / s["runs"], 3)
        s["total_rounds"] += result.get("rounds", 1) if result else 1
        s["total_elapsed"] += result.get("elapsed", 0) if result else 0
        s["avg_rounds"] = round(s["total_rounds"] / s["runs"], 2)
        s["avg_elapsed"] = round(s["total_elapsed"] / s["runs"], 1)
        s["last_delta_changed"] = changed
        save(stats)
        return s

    elif action == "select":
        stats = load()
        models_pool = [m.strip() for m in pool.split(",") if m.strip()] or [DEFAULT_MODEL]
        size = min(n, len(models_pool))
        possible = list(combinations(models_pool, size)) or [tuple(models_pool)]
        possible_strs = [",".join(c) for c in possible]

        epsilon = 0.3  # 30% exploration
        if random.random() < epsilon or not any(k in stats for k in possible_strs):
            chosen = list(random.choice(possible))
        else:
            known = [(c, stats.get(",".join(c), {}).get("acceptance_rate", 0)) for c in possible if ",".join(c) in stats]
            known.sort(key=lambda x: x[1], reverse=True)
            chosen = list(known[0][0]) if known else list(random.choice(possible))

        return {"coalition": ",".join(chosen), "models": chosen,
                "explored": len([k for k in possible_strs if k in stats]),
                "total_possible": len(possible),
                "stats": {k: v for k, v in stats.items() if k in possible_strs}}

    return {"error": f"unknown action: {action}"}


@app.function(image=base_image, volumes={"/deltas": DELTA_VOL}, timeout=60)
def delta_store(action: str = "read", record: dict = None, limit: int = 50) -> dict:
    """Persistent accumulation of (swarm_proposal, frontier_correction) pairs.

    The learning dataset: every time the frontier model corrects the swarm,
    the pair is recorded. Over time this builds a preference dataset for
    fine-tuning or prompt optimization. The swarm literally learns from the frontier.
    """
    import json, os, time
    path = "/deltas/corrections.jsonl"
    os.makedirs("/deltas", exist_ok=True)

    if action == "record":
        entry = {
            "timestamp": time.time(),
            "goal": record.get("goal", ""),
            "coalition": record.get("coalition", ""),
            "round": record.get("round", 1),
            "proposal": record.get("proposal", ""),
            "correction": record.get("correction", ""),
            "delta_changed": record.get("delta_changed", False),
            "frontier_model": record.get("frontier_model", ""),
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        DELTA_VOL.commit()
        count = 0
        if os.path.exists(path):
            with open(path) as f:
                count = sum(1 for _ in f)
        return {"recorded": True, "total": count}

    elif action == "read":
        if not os.path.exists(path):
            return {"deltas": [], "total": 0}
        deltas = []
        with open(path) as f:
            for line in f:
                try:
                    deltas.append(json.loads(line))
                except:
                    pass
        return {"deltas": deltas[-limit:], "total": len(deltas)}

    elif action == "export":
        # Export deltas as fine-tuning preference pairs (goal, proposal, correction)
        if not os.path.exists(path):
            return {"pairs": [], "total": 0}
        pairs = []
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("delta_changed"):
                        pairs.append({
                            "goal": d.get("goal", ""),
                            "proposal": d.get("proposal", ""),
                            "correction": d.get("correction", ""),
                            "coalition": d.get("coalition", ""),
                            "frontier_model": d.get("frontier_model", ""),
                        })
                except:
                    pass
        return {"pairs": pairs, "total": len(pairs), "format": "preference_pairs"}

    elif action == "stats":
        if not os.path.exists(path):
            return {"total": 0, "corrected": 0, "accepted": 0, "correction_rate": 0}
        corrected = total = 0
        coalitions = {}
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    total += 1
                    c = d.get("delta_changed", False)
                    if c:
                        corrected += 1
                    coal = d.get("coalition", "unknown")
                    if coal not in coalitions:
                        coalitions[coal] = {"total": 0, "corrected": 0}
                    coalitions[coal]["total"] += 1
                    if c:
                        coalitions[coal]["corrected"] += 1
                    coalitions[coal]["correction_rate"] = round(coalitions[coal]["corrected"] / coalitions[coal]["total"], 3)
                except:
                    pass
        return {"total": total, "corrected": corrected, "accepted": total - corrected,
                "correction_rate": round(corrected / total, 3) if total else 0,
                "by_coalition": coalitions}

    elif action == "prepare_training":
        # Convert deltas to instruction-tuning format (goal -> correction)
        if not os.path.exists(path):
            return {"examples": 0, "total": 0}
        examples = []
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("delta_changed"):
                        examples.append({
                            "instruction": d.get("goal", ""),
                            "input": "",
                            "output": d.get("correction", ""),
                            "coalition": d.get("coalition", ""),
                        })
                except:
                    pass
        train_path = "/deltas/training.jsonl"
        with open(train_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        DELTA_VOL.commit()
        return {"examples": len(examples), "path": train_path, "format": "instruction_tuning"}

    return {"error": f"unknown action: {action}"}


@app.function(image=modal.Image.debian_slim(python_version="3.11").apt_install("git"), volumes={"/workspace": WORKSPACE_VOL}, timeout=60)
def workspace(action: str, path: str = "", content: str = "", file1: str = "", file2: str = "", record: dict = None) -> dict:
    """Repository workspace: file tree operations on a Modal volume.

    The cognition loop can operate on real files (tree, read, write, diff,
    apply/revert) instead of just isolated code snippets.
    """
    import os, json, difflib
    base = "/workspace"
    os.makedirs(base, exist_ok=True)

    if action == "list":
        tree = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, base)
                tree.append({"path": rel, "size": os.path.getsize(fp)})
        return {"files": sorted(tree, key=lambda x: x["path"]), "count": len(tree)}

    elif action == "read":
        full = os.path.join(base, path)
        if not os.path.exists(full):
            return {"error": "not found", "path": path}
        with open(full) as f:
            return {"path": path, "content": f.read(), "size": os.path.getsize(full)}

    elif action == "write":
        full = os.path.join(base, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        WORKSPACE_VOL.commit()
        return {"written": path, "size": len(content)}

    elif action == "delete":
        full = os.path.join(base, path)
        if os.path.exists(full):
            os.remove(full)
            WORKSPACE_VOL.commit()
            return {"deleted": path}
        return {"error": "not found"}

    elif action == "diff":
        f1 = os.path.join(base, file1)
        f2 = os.path.join(base, file2)
        if not os.path.exists(f1) or not os.path.exists(f2):
            return {"error": "file not found"}
        with open(f1) as a, open(f2) as b:
            diff = list(difflib.unified_diff(a.readlines(), b.readlines(),
                                             fromfile=file1, tofile=file2, lineterm=""))
        return {"file1": file1, "file2": file2, "diff": "\n".join(diff[:100]), "lines_changed": len(diff)}

    elif action == "snapshot":
        # Save current state of a file as a .prev version (for revert)
        full = os.path.join(base, path)
        if not os.path.exists(full):
            return {"error": "not found"}
        prev = full + ".prev"
        import shutil
        shutil.copy2(full, prev)
        WORKSPACE_VOL.commit()
        return {"snapshotted": path}

    elif action == "revert":
        full = os.path.join(base, path)
        prev = full + ".prev"
        if not os.path.exists(prev):
            return {"error": "no snapshot"}
        import shutil
        shutil.copy2(prev, full)
        WORKSPACE_VOL.commit()
        return {"reverted": path}

    elif action == "git_init":
        import subprocess
        subprocess.run(["git", "init"], cwd=base, capture_output=True, timeout=10)
        subprocess.run(["git", "config", "user.email", "swarm@modal.run"], cwd=base, capture_output=True, timeout=5)
        subprocess.run(["git", "config", "user.name", "pi-modal-mcp swarm"], cwd=base, capture_output=True, timeout=5)
        WORKSPACE_VOL.commit()
        return {"git_init": True}

    elif action == "git_commit":
        import subprocess
        subprocess.run(["git", "add", "-A"], cwd=base, capture_output=True, timeout=10)
        msg = record.get("message", "swarm cognition update") if record else "swarm cognition update"
        r = subprocess.run(["git", "commit", "-m", msg], cwd=base, capture_output=True, text=True, timeout=10)
        WORKSPACE_VOL.commit()
        return {"committed": r.returncode == 0, "output": r.stdout[:500] + r.stderr[:500]}

    elif action == "git_diff":
        import subprocess
        r = subprocess.run(["git", "diff", "HEAD"], cwd=base, capture_output=True, text=True, timeout=10)
        return {"diff": r.stdout[:2000], "lines": len(r.stdout.splitlines())}

    elif action == "git_log":
        import subprocess
        r = subprocess.run(["git", "log", "--oneline", "-10"], cwd=base, capture_output=True, text=True, timeout=10)
        return {"log": r.stdout.strip()}

    elif action == "git_status":
        import subprocess
        r = subprocess.run(["git", "status", "--short"], cwd=base, capture_output=True, text=True, timeout=10)
        return {"status": r.stdout.strip(), "clean": len(r.stdout.strip()) == 0}

    return {"error": f"unknown action: {action}"}


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
            full_prompt = f"{role_prompt}\n\n{scl_context}\n\n## Task\n{prompt}" if scl_context else f"{role_prompt}\n\n## Task\n{prompt}"
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
<title>pi-modal-mcp &middot; agent</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bd:#30363d;--tx:#e6edf3;--tx2:#8b949e;--acc:#58a6ff;--grn:#3fb950;--red:#f85149;--yel:#d29922;--pur:#bc8cff}
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--tx);height:100vh;display:flex;flex-direction:column;overflow:hidden}
  header{background:linear-gradient(90deg,#6366f1,#0ea5e9);padding:8px 16px;display:flex;align-items:center;gap:12px;flex-shrink:0}
  header h1{font-size:14px;color:#fff;font-weight:600}
  header .sub{font-size:11px;color:#dbeafe}
  header .dot{width:8px;height:8px;border-radius:50%;background:var(--grn);margin-left:auto}
  .main{display:flex;flex:1;min-height:0}
  /* Left: file tree + editor */
  .left{width:35%;display:flex;flex-direction:column;border-right:1px solid var(--bd);min-width:280px}
  .tree{background:var(--bg2);padding:6px 8px;border-bottom:1px solid var(--bd);max-height:35%;overflow:auto;flex-shrink:0}
  .tree .f{padding:3px 8px;font-size:12px;cursor:pointer;border-radius:4px;color:var(--tx2);font-family:ui-monospace,monospace}
  .tree .f:hover{background:var(--bg3);color:var(--tx)}
  .tree .f.active{background:var(--bg3);color:var(--acc)}
  #editor{flex:1;min-height:0}
  /* Center: chat */
  .center{flex:1;display:flex;flex-direction:column;min-width:0}
  .chat{flex:1;overflow:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
  .msg{max-width:85%;padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.5}
  .msg.user{align-self:flex-end;background:#1a1f2e;border:1px solid var(--bd)}
  .msg.assistant{align-self:flex-start;background:var(--bg2);border:1px solid var(--bd)}
  .msg.assistant pre{margin-top:6px;background:var(--bg);padding:8px 10px;border-radius:6px;font-size:12px;overflow-x:auto;white-space:pre-wrap}
  .msg.system{align-self:center;background:transparent;color:var(--tx2);font-size:12px;font-style:italic}
  .input-bar{padding:10px 16px;border-top:1px solid var(--bd);display:flex;gap:8px;align-items:flex-end}
  .input-bar textarea{flex:1;background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:8px;padding:8px 12px;font-size:14px;resize:none;height:40px;max-height:120px;font-family:inherit}
  .input-bar button{background:#6366f1;border:1px solid #7c3aed;color:#fff;padding:8px 16px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap}
  .input-bar button:disabled{opacity:.4;cursor:wait}
  /* Right: agent activity */
  .right{width:30%;min-width:260px;display:flex;flex-direction:column;border-left:1px solid var(--bd);background:var(--bg2)}
  .right .head{padding:8px 12px;border-bottom:1px solid var(--bd);font-size:12px;color:var(--tx2);display:flex;justify-content:space-between;align-items:center}
  .right .head .spin{width:14px;height:14px;border:2px solid var(--bd);border-top-color:var(--acc);border-radius:50%;animation:sp 0.6s linear infinite}
  @keyframes sp{to{transform:rotate(360deg)}}
  .feed{flex:1;overflow:auto;padding:8px;display:flex;flex-direction:column;gap:4px}
  .ev{font-size:11px;font-family:ui-monospace,monospace;padding:4px 8px;border-radius:4px;color:var(--tx2);line-height:1.4}
  .ev .t{color:var(--tx2);font-size:10px;margin-right:6px}
  .ev.round{color:var(--pur);font-weight:600}
  .ev.generate{color:var(--acc)}
  .ev.candidate{color:var(--tx);background:var(--bg3)}
  .ev.conflicts{color:var(--yel)}
  .ev.synth{color:var(--grn)}
  .ev.frontier{color:var(--pur)}
  .ev.security{color:var(--yel)}
  .ev.verify{font-weight:600}
  .ev.pass{color:var(--grn)}
  .ev.fail{color:var(--red)}
  .ev.halt{color:var(--grn);font-weight:bold;font-size:12px}
  .ev.ws{color:var(--acc)}
</style></head><body>
<header>
  <h1>&#129504; pi-modal-mcp agent</h1>
  <span class=sub>self-hosted OSS swarm on Modal GPUs</span>
  <span class=dot id=dot></span>
</header>
<div class=main>
  <div class=left>
    <div class=tree id=tree></div>
    <div id=editor></div>
  </div>
  <div class=center>
    <div class=chat id=chat></div>
    <div class=input-bar>
      <textarea id=msg placeholder="Describe a coding task..." rows=1></textarea>
      <button id=send>Send</button>
    </div>
  </div>
  <div class=right>
    <div class=head><span>agent activity</span><span class=spin id=spin style=display:none></span></div>
    <div class=feed id=feed></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min/vs/loader.min.js"></script>
<script>
let ed,activeFile='';
require.config({paths:{vs:'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min/vs'}});
require(['vs/editor/editor.main'],function(){
  ed=monaco.editor.create(document.getElementById('editor'),{value:'// workspace files load on the left',language:'python',theme:'vs-dark',automaticLayout:true,fontSize:13,readOnly:true});
});
const $=id=>document.getElementById(id);
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function now(){return new Date().toLocaleTimeString('en',{hour12:false}).slice(0,8);}
// File tree
async function loadTree(){
  try{const r=await fetch('/api/workspace');const d=await r.json();
    $('tree').innerHTML=d.files.map(f=>'<div class=f data-p='+esc(f.path)+'>'+esc(f.path)+'</div>').join('');
    document.querySelectorAll('.tree .f').forEach(el=>el.onclick=()=>openFile(el.dataset.p));
  }catch(e){}
}
async function openFile(path){
  activeFile=path;
  document.querySelectorAll('.tree .f').forEach(el=>el.classList.toggle('active',el.dataset.p===path));
  try{const r=await fetch('/api/workspace/read',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});const d=await r.json();
    if(d.content!==undefined){ed.setValue(d.content);ed.updateOptions({readOnly:false});}
  }catch(e){}
}
// Chat
function addMsg(role,text){
  const div=document.createElement('div');div.className='msg '+role;
  // Render code blocks
  const parts=text.split(/```/);
  let html='';
  for(let i=0;i<parts.length;i++){html+=i%2===1?'<pre>'+esc(parts[i])+'</pre>':esc(parts[i]);}
  div.innerHTML=html||esc(text);
  $('chat').appendChild(div);$('chat').scrollTop=$('chat').scrollHeight;
}
// Activity feed
function addEvent(cls,text){
  const div=document.createElement('div');div.className='ev '+cls;
  div.innerHTML='<span class=t>'+now()+'</span>'+esc(text);
  $('feed').appendChild(div);$('feed').scrollTop=$('feed').scrollHeight;
}
// Agent: send task, stream cognition events, apply results
let busy=false;
$('send').onclick=sendTask;
$('msg').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendTask();}};
async function sendTask(){
  const task=$('msg').value.trim();if(!task||busy)return;
  busy=true;$('send').disabled=true;$('spin').style.display='inline-block';$('dot').style.background='var(--yel)';
  $('msg').value='';
  addMsg('user',task);
  addMsg('system','agent starting cognition loop...');
  addEvent('round','\u2192 task: '+task.slice(0,60));
  const body={goal:task,n:2,max_rounds:3,max_new_tokens:128,workspace_path:activeFile||undefined};
  try{
    const r=await fetch('/api/cognition',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const reader=r.body.getReader();const dec=new TextDecoder();let buf='';let haltReason='';let synthCode='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        const d=line.slice(6).trim();if(d==='[DONE]')continue;
        try{
          const e=JSON.parse(d);const ev=e.event||'';const r=e.round||'';
          if(ev==='round_start')addEvent('round','R'+r+'/'+e.max_rounds+' starting');
          else if(ev==='generate')addEvent('generate','generating: '+e.role+' (worker '+e.worker+')');
          else if(ev==='candidate')addEvent('candidate','candidate '+e.worker+': '+(e.content||'').slice(0,50));
          else if(ev==='conflicts')addEvent('conflicts',e.count+' conflict(s) detected');
          else if(ev==='tests')addEvent('synth','tests: '+(e.content||'').slice(0,50));
          else if(ev==='synthesize'){addEvent('synth','synthesis: '+(e.content||'').slice(0,60));synthCode=e.content||'';}
          else if(ev==='frontier_verify_start')addEvent('frontier','frontier verifying ('+e.model+')...');
          else if(ev==='frontier_corrected')addEvent('frontier','\u26a0 frontier CORRECTED (delta)');
          else if(ev==='frontier_accepted')addEvent('frontier','\u2713 frontier accepted');
          else if(ev==='security_review_start')addEvent('security','security review...');
          else if(ev==='security_corrected')addEvent('security','\u26a0 security fix applied');
          else if(ev==='security_passed')addEvent('security','\u2713 security passed');
          else if(ev==='verify_result')addEvent(e.passed?'pass':'fail',(e.passed?'\u2713 PASS':'\u2717 FAIL')+(e.stdout?' out='+e.stdout.slice(0,20):''));
          else if(ev==='workspace_applied')addEvent('ws','\u2713 applied to '+e.path);
          else if(ev==='halt'){haltReason=e.reason;addEvent('halt','\u25a0 HALT: '+e.reason+' ('+e.elapsed+'s, R'+e.round+')');}
        }catch(err){}
      }
    }
    // Show result in chat
    if(haltReason==='verified'&&synthCode){
      addMsg('assistant','Done! Verified in sandbox. Applied to '+activeFile+'.\n```python\n'+synthCode+'\n```');
    }else if(haltReason==='max_rounds'){
      addMsg('assistant','Reached max rounds without verification. The code may need manual review.\n```python\n'+synthCode+'\n```');
    }else{
      addMsg('assistant','Cognition loop completed ('+haltReason+').');
    }
    loadTree(); // refresh file tree
    if(activeFile)openFile(activeFile); // reload editor with changes
  }catch(e){
    addMsg('assistant','Error: '+e);
  }finally{
    busy=false;$('send').disabled=false;$('spin').style.display='none';$('dot').style.background='var(--grn)';
  }
}
loadTree();
</script>
</body></html>"""
