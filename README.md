# chef-backend

Backend API and MCP server for **Chef**, an AI-first personal recipe assistant.
The Android app lives in [AlltidSemester1337/chef](https://github.com/AlltidSemester1337/chef).

**Status:** early development. The API for the Chef clients and a remote MCP server for
Claude and other MCP clients are running.

## Stack

- Python 3.13, FastAPI, uv
- Cloud Run (scale to zero), Cloud Firestore, Secret Manager
- MCP server (Streamable HTTP, stateless) via the official MCP Python SDK, with OAuth sign-in
- Tracing with OpenTelemetry/OpenInference to Arize Phoenix (optional)

## Getting started

```bash
uv sync
cp .env.example .env            # placeholders only; never commit .env
gcloud auth application-default login   # Google credentials via ADC, no key files
uv run uvicorn chef_backend.main:app --reload --port 8080
curl localhost:8080/health
```

Run checks:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## Connecting to the MCP server

The MCP server lives at `/mcp` on the deployed service, e.g.
`https://<your-chef-backend>/mcp`. You sign in with your Chef account (OAuth, the client
registers itself), and choose whether Claude may only read your recipes
(`recipes:read`) or also save new ones (`recipes:write`).

**Claude.ai and the Claude apps:** Settings → Connectors → Add custom connector, and enter
the `/mcp` URL. Claude opens the Chef sign-in page the first time it uses the connector.

**Claude Code:**

```bash
claude mcp add --transport http chef https://<your-chef-backend>/mcp
```

Then run `/mcp` in Claude Code and choose `chef` to sign in.

**MCP Inspector** (Node.js 20 or later), for trying tools, resources and prompts by hand:

```bash
npx @modelcontextprotocol/inspector
```

Choose transport *Streamable HTTP*, enter the `/mcp` URL and connect in *Via Proxy*
mode. *Direct* mode calls `/mcp` from the browser, which the server only allows for
origins listed in `CHEF_MCP_ALLOWED_ORIGINS` (e.g. `["http://localhost:6274"]`).

**Locally:** run the server as in *Getting started* and use `http://localhost:8080/mcp`.
Signing in needs the Firebase web app settings (`CHEF_FIREBASE_WEB_*`) and Firestore.

### Tools, resources and prompts

- **Search** your own recipes and the community's, and read any of them in full,
  including the variants saved for your own recipes.
- **Save** a recipe you created or adapted with Claude, or save a variant (e.g. vegetarian,
  for two) of one of your own recipes. Saved recipes are shared with the Chef community,
  and variants of your shared recipes are visible too. Saving needs the `recipes:write`
  scope at sign-in and has a daily limit.
- **Attach** a recipe to the conversation as a resource, so Claude has it in full.
- **Prompts:** `plan_week` plans 1–7 dinners from your recipes with a combined shopping
  list; `shopping_list` builds one shopping list for up to 10 recipes, scaled to the
  servings you need.

#### Examples

- "Which taco recipes from the community can you recommend?"
- "Create and save a variant of my pasta carbonara without onion."
- "List the names of all my quick and easy recipes."
- Run `plan_week` with "vegetarian, under 30 minutes, 2 people".

## Security

No secrets or private configuration are stored in this repository. Secrets are kept in
Google Secret Manager and Google APIs are reached via Application Default Credentials.
CI runs [gitleaks](https://github.com/gitleaks/gitleaks) on every push.

## Releases
1.0-alpha: Backend API replacing direct integration from Android app, first MVP MCP server release, simple read MCP server read tools for recipes

## Demos / how to use

[youtube demos](https://www.youtube.com/playlist?list=PL3z3ETRVg-c4teb_hZ8OzLwx7ySJKoB1Q)


## License

Apache-2.0
