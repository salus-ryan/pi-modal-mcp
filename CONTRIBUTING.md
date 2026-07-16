# Contributing

Thanks for your interest in `pi-modal-mcp`. This is a small project; keep PRs
focused.

## Setup

```bash
git clone https://github.com/salus-ryan/pi-modal-mcp
cd pi-modal-mcp

# Python lint/format
pip install ruff

# pi extension deps (for local type checking / pi -e)
cd mcp-runtime && npm install && cd ..
```

## Local checks before pushing

```bash
# Python
ruff check modal-mcp modal-frontend
ruff format --check modal-mcp modal-frontend
python -m py_compile modal-mcp/modal_mcp.py modal-frontend/app.py

# JSON
python -c "import json; json.load(open('mcp-runtime/mcp.json.example'))"

# Install the shim as you would locally
pip install ./modal-mcp
python -c "import modal_mcp; print('ok')"
```

CI runs the same suite (`.github/workflows/ci.yml`).

## Deploying to Modal from a fork

Forks and PRs do not have access to `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`,
so the deploy workflow only runs on the canonical repo. To test a deploy from
a fork, deploy manually:

```bash
cd modal-frontend
modal deploy app.py   # uses your own Modal profile
```

## Conventions

- **No secrets in code.** Workspace URLs and tokens come from env / secrets
  only. The repo must stay free of `salus`/`ryan-barrett`/token strings.
- **Keep the shim thin.** `modal_mcp.py` translates MCP calls to HTTP; do real
  work in the Modal app, not the shim.
- **Stateless MCP.** The Modal-hosted MCP endpoint uses `stateless_http=True`
  because Modal does not provide session affinity across containers. Don't add
  stateful session assumptions without solving affinity.
- **Schemas are permissive.** MCP tool `inputSchema` uses
  `Type.Unsafe(...)` because servers define them dynamically; do not over-constrain.

## Releases

Tags `v*` trigger `.github/workflows/release.yml`, which creates a GitHub
Release with an auto-generated changelog. Bump the version in
`pyproject.toml` and the `modal-frontend` app name if appropriate.
