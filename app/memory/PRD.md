# AI Builder — PRD

## Original Problem Statement
User (Romanian) wants an AI app-builder to compete with Emergent: keeps the AI API key secret on a server, generates apps that look human & beautiful, is very good at code, does GitHub commits, has a single very-good review agent that loops until it finds nothing (then 2 confirmation passes), plus tools (calculator, web search via SearXNG, notes). AI provider: Google Gemini (user-provided key). Deployment target: user will host backend on Render.

## Architecture
- Frontend: Expo Router (React Native), dark developer theme, tabs: Build / Tools / Notes / Settings + project detail screen.
- Backend: FastAPI + MongoDB (motor). Gemini via emergentintegrations, key secret in backend/.env with Emergent universal key as automatic fallback.
- Web search: public SearXNG instances (JSON).

## Core Requirements (static)
1. AI key never leaves the server. ✔ (only backend reads it)
2. Chat builder that plans + generates full code files. ✔
3. Single iterative review agent: loops until clean + 2 confirmation passes (3 consecutive clean), background job with polling (no HTTP timeout). ✔
4. GitHub commit of generated files via user PAT. ✔
5. Tools: calculator, web search (SearXNG), notes. ✔
6. Delete a project/chat from Build list and from inside the chat. ✔

## Implemented (2026-08-04)
- Model: gemini-3.5-flash (user's free-tier key works with it; pro models rate-limited, 2.5-flash deprecated for new keys). Emergent key fallback.
- Backend endpoints: /health, projects CRUD (+delete), project chat (Gemini + file parsing), review as background job (POST /review -> job_id, GET /review/{job_id} polling, loops until 3 consecutive clean, cap 8), notes CRUD, calculator, websearch, github repos+commit.
- Frontend: Build (projects list/create/DELETE with confirm), project detail (Chat/Files/Review-live-polling/GitHub tabs, delete-chat in header), Tools (calculator+search), Notes, Settings.
- Delete chat: trash on each Build card + trash in project header, both with confirm modal.
- Verified: health, calculator, project create, Gemini chat + file parsing, review background job (3 passes, live progress), delete UI.

## Deployment note
- On Render: set GEMINI_API_KEY (and optionally EMERGENT_LLM_KEY) as environment variables in the dashboard. Backend binds 0.0.0.0:8001; all routes under /api.

## Backlog / Remaining
- P1: Streaming chat responses (token-by-token) for faster feel.
- P1: Auto-apply review fixes back into files + diff view (partially done: review updates files).
- P2: Multiple custom agent teams, user terminal (Termux-like), APK build/run sandbox — require heavy infra, out of MVP scope.
- P2: Syntax highlighting in Files view.

## Next Tasks
- Run testing agent on backend flows.
- Iterate on code-generation quality prompts.