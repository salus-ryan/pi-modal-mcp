# Security Policy

## Reporting a vulnerability

Email security concerns to the maintainer via GitHub (DM the repo owner or
open a private security advisory):

1. Go to the repo **Security** tab → **Advisories** → **New**.
2. Describe the issue and reproduction steps.
3. Mark it private until a fix is coordinated.

Do **not** open a public issue for security problems.

## Scope

- The **Modal-hosted MCP endpoint** (`mcp_web`) disables DNS-rebinding host
  checks and allows all hosts (`allowed_hosts=["*"]`) by design, so any client
  can reach it. **Do not deploy this unmodified if your tools are sensitive.**
  For private deployments, restrict `mcp_server.settings.transport_security`
  in `modal-frontend/app.py`.
- The **stdio shim** (`modal-mcp`) runs locally and forwards to your deployed
  web URL. It does not authenticate the endpoint; assume the network is
  trusted or add auth at the Modal layer.
- No credentials are committed. Modal tokens live in GitHub Actions secrets
  (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) or your local `~/.modal.toml`.

## Disclosure

Acknowledged within 72 hours. Fixes released as patch tags (`vX.Y.Z`).
