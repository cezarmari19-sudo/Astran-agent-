# AI Builder — PRD

## Original Problem Statement
User (Romanian) wants an AI app-builder to compete with Emergent: keeps the AI API key secret on a server, generates apps that look human & beautiful, is very good at code, does GitHub commits, has a single very-good review agent that loops until it finds nothing (then 2 confirmation passes), plus tools (calculator, web search via SearXNG, notes). AI provider: Google Gemini (user-provided key). Deployment target: user will host backend on Render.

## Architecture
- Frontend: Expo Router (React Native), dark developer theme, tabs: Build / Tools / Notes / Settings + project detail screen.
- Backend: FastAPI + MongoDB (motor). Gemini via emergentintegrations (`gemini-3.1-pro-preview`), key secret in backend/.env with Emergent universal key as automatic fallback.
- Web search: public SearXNG instances (JSON).

## Core Requirements (static)
1. AI key never leaves the server. ✔ (only backend reads it)
2. Chat builder that plans + generates full code files. ✔
3. Single iterative review agent: loops until clean + 2 confirmation passes (3 consecutive clean), practical cap 5 passes. ✔
4. GitHub commit of generated files via user PAT. ✔
5. Tools: calculator, web search (SearXNG), notes. ✔

## Implemented (2026-08-04)
- Backend endpoints: /health, projects CRUD, project chat (Gemini + file parsing), review agent loop, notes CRUD, calculator, websearch, github repos+commit.
- Frontend: Build (projects list/create), project detail (Chat/Files/Review/GitHub tabs), Tools (calculator+search), Notes, Settings (GitHub token verify, security status, about).
- Verified via curl: health, calculator, project create, Gemini chat reply.

## Backlog / Remaining
- P1: Streaming chat responses (token-by-token) for faster feel.
- P1: Auto-apply review fixes back into files + diff view (partially done: review updates files).
- P2: Multiple custom agent teams, user terminal (Termux-like), APK build/run sandbox — require heavy infra, out of MVP scope.
- P2: Syntax highlighting in Files view.

## Next Tasks
- Run testing agent on backend flows.
- Iterate on code-generation quality prompts.