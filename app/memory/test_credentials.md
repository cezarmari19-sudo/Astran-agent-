# Test Credentials

## AI
- Gemini API key: stored in backend/.env as GEMINI_API_KEY (secret, server-only)
- Fallback: EMERGENT_LLM_KEY (Emergent universal key) in backend/.env
- Model: gemini-3.1-pro-preview

## GitHub
- No test token available. GitHub endpoints require a user-provided Personal Access Token (PAT) with `repo` scope, entered in the app (Settings or project GitHub tab).
- Invalid token should return HTTP 400 from /api/github/repos.

## Notes
- No auth in this app (single-user local tool). All endpoints are open under /api.