from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import re
import ast
import json
import base64
import logging
import operator
import uuid
import requests
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone

from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_MODEL = "gemini-3.5-flash"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
api_router = APIRouter(prefix="/api")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean(doc):
    if doc and "_id" in doc:
        doc = {k: v for k, v in doc.items() if k != "_id"}
    return doc


# ---------------- LLM ----------------
async def _call(chat, prompt):
    if hasattr(chat, "send_message"):
        return await chat.send_message(UserMessage(text=prompt))
    from emergentintegrations.llm.chat import TextDelta, StreamDone
    out = []
    async for ev in chat.stream_message(UserMessage(text=prompt)):
        if isinstance(ev, TextDelta):
            out.append(ev.content)
        elif isinstance(ev, StreamDone):
            break
    return "".join(out)


async def llm_generate(system_message, prompt, session_id, model=None):
    model = model or GEMINI_MODEL
    if model.startswith("gemini"):
        provider = "gemini"
    elif model.startswith("claude"):
        provider = "anthropic"
    else:
        provider = "openai"

    attempts = []
    key_map = {"gemini": GEMINI_API_KEY, "anthropic": ANTHROPIC_API_KEY, "openai": OPENAI_API_KEY}
    own = key_map.get(provider)
    if own:
        attempts.append(own)
    if EMERGENT_LLM_KEY:
        attempts.append(EMERGENT_LLM_KEY)

    last_err = None
    for key in attempts:
        try:
            chat = LlmChat(api_key=key, session_id=session_id,
                           system_message=system_message).with_model(provider, model)
            resp = await _call(chat, prompt)
            if resp:
                return resp
        except Exception as e:
            last_err = e
            logger.error(f"LLM attempt failed ({model}): {e}")
    raise HTTPException(status_code=502, detail=f"AI indisponibil: {last_err}")


AVAILABLE_MODELS = [
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash", "hint": "Rapid • cheia ta Gemini", "provider": "gemini"},
    {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash", "hint": "Rapid • cheia ta Gemini", "provider": "gemini"},
    {"id": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro", "hint": "Puternic • cheia ta Gemini", "provider": "gemini"},
    {"id": "claude-sonnet-5", "label": "Claude Sonnet 5", "hint": "Cheie Anthropic", "provider": "anthropic"},
    {"id": "claude-opus-4-8", "label": "Claude Opus 4.8", "hint": "Cheie Anthropic", "provider": "anthropic"},
    {"id": "claude-fable-5", "label": "Claude Fable 5", "hint": "Cheie Anthropic", "provider": "anthropic"},
    {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "hint": "Cheie Anthropic", "provider": "anthropic"},
    {"id": "gpt-5.5", "label": "ChatGPT 5.5", "hint": "Cheie OpenAI", "provider": "openai"},
    {"id": "gpt-5.4", "label": "ChatGPT 5.4", "hint": "Cheie OpenAI", "provider": "openai"},
]


BUILDER_SYSTEM = (
    "You are Aria, an elite senior full-stack & mobile engineer and product designer, "
    "the core of an AI app-builder that competes with the best. You PLAN carefully before coding, "
    "then write complete, production-ready, bug-free code with clean architecture. "
    "You obsess over beautiful, human, non-generic UI/UX (no AI-slop: avoid purple-on-white gradients, "
    "generic centered layouts, emoji icons; use real design systems, spacing, depth and polish). "
    "When the user asks to build or change an app/feature or write code, respond with: "
    "(1) a short friendly explanation and a concise numbered plan, then "
    "(2) the FULL code for every file, each in this EXACT format:\n"
    "### FILE: relative/path/to/file.ext\n"
    "```lang\n<complete file content>\n```\n"
    "Always output complete files (never omit code with '...'). Keep answers focused. "
    "For questions that are not code, answer helpfully and concisely like a great engineer."
)

REVIEW_SYSTEM = (
    "You are a review orchestrator running THREE specialized agents on a generated app's source code. "
    "Evaluate the APP (not yourself) from three angles and tag every issue with its agent category:\n"
    "1. category=\"human\": Does the app look HUMAN-made, not AI-generated slop? Flag generic centered "
    "layouts, purple-on-white gradients, emoji-as-icons, boilerplate copy, cookie-cutter structure, "
    "everything that screams 'AI wrote this'.\n"
    "2. category=\"beauty\": Is the UI/UX genuinely beautiful and polished? Flag weak spacing, poor color/"
    "contrast, no depth/hierarchy, ugly components, missing states, bad typography.\n"
    "3. category=\"security\": BRUTE-FORCE / hacker perspective. Find bugs & vulnerabilities an attacker "
    "would exploit for themselves: exposed secrets/API keys, injection, missing auth/validation, insecure "
    "storage, logic flaws, crashes, race conditions.\n"
    "Then FIX every issue you can. Respond with STRICT JSON only, no markdown:\n"
    '{"issues":[{"category":"human|beauty|security","severity":"high|medium|low","file":"path",'
    '"description":"...","fix":"..."}],"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
    "If the app is genuinely clean across all three agents, return "
    '{"issues":[],"files":[],"summary":"No issues found."}. '
    "Only include files in \"files\" that you actually changed, with their FULL new content."
)


def parse_files(text):
    files = []
    pattern = re.compile(r"###\s*FILE:\s*(?P<path>[^\n`]+?)\s*\n+```[^\n]*\n(?P<code>.*?)```",
                         re.DOTALL)
    for m in pattern.finditer(text or ""):
        path = m.group("path").strip()
        code = m.group("code")
        if path:
            files.append({"path": path, "content": code})
    return files


def merge_files(existing, new):
    by_path = {f["path"]: f for f in existing}
    for f in new:
        by_path[f["path"]] = f
    return list(by_path.values())


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# ---------------- Models ----------------
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""


class ChatIn(BaseModel):
    message: str
    model: Optional[str] = None


class ReviewIn(BaseModel):
    model: Optional[str] = None


class NoteIn(BaseModel):
    title: str
    content: Optional[str] = ""


class CalcIn(BaseModel):
    expression: str


class SearchIn(BaseModel):
    query: str


class GithubReposIn(BaseModel):
    token: str


class GithubCommitIn(BaseModel):
    token: str
    repo: str  # owner/name
    branch: Optional[str] = "main"
    message: str
    project_id: Optional[str] = None
    files: Optional[List[dict]] = None


# ---------------- Routes ----------------
@api_router.get("/")
async def root():
    return {"message": "AI Builder API online", "model": GEMINI_MODEL}


@api_router.get("/health")
async def health():
    return {"status": "ok", "ai_key_configured": bool(GEMINI_API_KEY or EMERGENT_LLM_KEY)}


@api_router.get("/models")
async def list_models():
    return {
        "models": AVAILABLE_MODELS,
        "default": GEMINI_MODEL,
        "providers_available": {
            "gemini": bool(GEMINI_API_KEY),
            "anthropic": bool(ANTHROPIC_API_KEY),
            "openai": bool(OPENAI_API_KEY),
        },
        "emergent_fallback": bool(EMERGENT_LLM_KEY),
    }


# Projects
@api_router.post("/projects")
async def create_project(body: ProjectCreate):
    proj = {
        "id": str(uuid.uuid4()),
        "name": body.name,
        "description": body.description or "",
        "files": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    await db.projects.insert_one(dict(proj))
    return clean(proj)


@api_router.get("/projects")
async def list_projects():
    docs = await db.projects.find().sort("updated_at", -1).to_list(200)
    return [clean(d) for d in docs]


@api_router.get("/projects/{pid}")
async def get_project(pid: str):
    doc = await db.projects.find_one({"id": pid})
    if not doc:
        raise HTTPException(404, "Proiect inexistent")
    return clean(doc)


@api_router.delete("/projects/{pid}")
async def delete_project(pid: str):
    await db.projects.delete_one({"id": pid})
    await db.messages.delete_many({"project_id": pid})
    return {"ok": True}


@api_router.get("/projects/{pid}/messages")
async def get_messages(pid: str):
    docs = await db.messages.find({"project_id": pid}).sort("created_at", 1).to_list(1000)
    return [clean(d) for d in docs]


@api_router.post("/projects/{pid}/chat")
async def project_chat(pid: str, body: ChatIn):
    project = await db.projects.find_one({"id": pid})
    if not project:
        raise HTTPException(404, "Proiect inexistent")

    history = await db.messages.find({"project_id": pid}).sort("created_at", 1).to_list(1000)
    recent = history[-12:]
    transcript = ""
    for m in recent:
        role = "User" if m["role"] == "user" else "Aria"
        transcript += f"{role}: {m['content']}\n\n"
    prompt = (
        f"Project: {project['name']}\nDescription: {project.get('description','')}\n\n"
        f"Conversation so far:\n{transcript}\nUser: {body.message}\n\nAria:"
    )

    reply = await llm_generate(BUILDER_SYSTEM, prompt, f"proj-{pid}", body.model)

    ts = now_iso()
    user_msg = {"id": str(uuid.uuid4()), "project_id": pid, "role": "user",
                "content": body.message, "created_at": ts}
    ai_msg = {"id": str(uuid.uuid4()), "project_id": pid, "role": "assistant",
              "content": reply, "created_at": now_iso()}
    await db.messages.insert_many([dict(user_msg), dict(ai_msg)])

    new_files = parse_files(reply)
    merged = merge_files(project.get("files", []), new_files) if new_files else project.get("files", [])
    if new_files:
        await db.projects.update_one({"id": pid},
                                     {"$set": {"files": merged, "updated_at": now_iso()}})

    return {"reply": reply, "files": new_files, "all_files": merged,
            "message": clean(ai_msg)}


REVIEW_JOBS = {}


async def _run_review(job_id, pid, model=None):
    job = REVIEW_JOBS[job_id]
    try:
        project = await db.projects.find_one({"id": pid})
        current = {f["path"]: f["content"] for f in project.get("files", [])}
        clean_streak = 0
        max_passes = 8  # generous cap; loops until clean + 2 confirmations
        for i in range(max_passes):
            blob = "\n\n".join([f"### FILE: {p}\n```\n{c}\n```" for p, c in current.items()])
            prompt = f"Review pass #{i+1}. Current project files:\n\n{blob}"
            raw = await llm_generate(REVIEW_SYSTEM, prompt, f"review-{pid}-{i}", model)
            data = extract_json(raw)
            if not data:
                data = {"issues": [{"severity": "low", "file": "-",
                                    "description": "Agentul a raspuns in format liber.",
                                    "fix": raw[:500]}], "files": [], "summary": "Format neuzual."}
            issues = data.get("issues", [])
            fixed = data.get("files", [])
            for f in fixed:
                if f.get("path"):
                    current[f["path"]] = f.get("content", current.get(f["path"], ""))
            job["passes"].append({"pass": i + 1, "issues": issues,
                                  "summary": data.get("summary", ""),
                                  "fixed_count": len(fixed)})
            if not issues:
                clean_streak += 1
            else:
                clean_streak = 0
            final_files = [{"path": p, "content": c} for p, c in current.items()]
            await db.projects.update_one({"id": pid},
                                         {"$set": {"files": final_files, "updated_at": now_iso()}})
            job["files"] = final_files
            # run until clean, then 2 more confirmation passes (3 consecutive clean total)
            if clean_streak >= 3:
                job["stopped_clean"] = True
                break
        job["total_passes"] = len(job["passes"])
        job["done"] = True
    except Exception as e:
        logger.error(f"review job error: {e}")
        job["error"] = str(e)
        job["done"] = True


@api_router.post("/projects/{pid}/review")
async def start_review(pid: str, body: ReviewIn, background_tasks: BackgroundTasks):
    project = await db.projects.find_one({"id": pid})
    if not project:
        raise HTTPException(404, "Proiect inexistent")
    if not project.get("files"):
        raise HTTPException(400, "Nu exista cod de verificat. Genereaza intai o aplicatie in chat.")
    job_id = str(uuid.uuid4())
    REVIEW_JOBS[job_id] = {"job_id": job_id, "project_id": pid, "passes": [],
                           "files": [], "done": False, "error": None,
                           "stopped_clean": False, "total_passes": 0}
    background_tasks.add_task(_run_review, job_id, pid, body.model)
    return {"job_id": job_id}


@api_router.get("/review/{job_id}")
async def review_status(job_id: str):
    job = REVIEW_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job inexistent")
    return job


# Notes
@api_router.post("/notes")
async def create_note(body: NoteIn):
    note = {"id": str(uuid.uuid4()), "title": body.title,
            "content": body.content or "", "created_at": now_iso()}
    await db.notes.insert_one(dict(note))
    return clean(note)


@api_router.get("/notes")
async def list_notes():
    docs = await db.notes.find().sort("created_at", -1).to_list(500)
    return [clean(d) for d in docs]


@api_router.delete("/notes/{nid}")
async def delete_note(nid: str):
    await db.notes.delete_one({"id": nid})
    return {"ok": True}


# Tools: calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv}


def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Doar numere")
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Expresie invalida")


@api_router.post("/tools/calculator")
async def calculator(body: CalcIn):
    try:
        tree = ast.parse(body.expression, mode="eval")
        result = _eval(tree.body)
        return {"expression": body.expression, "result": result}
    except Exception as e:
        raise HTTPException(400, f"Expresie invalida: {e}")


# Tools: web search via public SearXNG
SEARX_INSTANCES = ["https://searx.be", "https://priv.au",
                   "https://searx.tiekoetter.com", "https://search.rhscz.eu"]


@api_router.post("/tools/websearch")
async def websearch(body: SearchIn):
    headers = {"User-Agent": "Mozilla/5.0 (AI-Builder)"}
    for base in SEARX_INSTANCES:
        try:
            r = requests.get(f"{base}/search",
                             params={"q": body.query, "format": "json"},
                             headers=headers, timeout=10)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
                results = []
                for item in data.get("results", [])[:8]:
                    results.append({"title": item.get("title", ""),
                                    "url": item.get("url", ""),
                                    "content": item.get("content", "")})
                if results:
                    return {"query": body.query, "engine": base, "results": results}
        except Exception as e:
            logger.info(f"searx {base} failed: {e}")
            continue
    return {"query": body.query, "engine": None, "results": [],
            "note": "Nicio instanta SearXNG publica nu a raspuns cu JSON. Incearca din nou."}


# GitHub
@api_router.post("/github/repos")
async def github_repos(body: GithubReposIn):
    headers = {"Authorization": f"Bearer {body.token}",
               "Accept": "application/vnd.github+json"}
    r = requests.get("https://api.github.com/user/repos",
                     params={"per_page": 100, "sort": "updated"},
                     headers=headers, timeout=15)
    if r.status_code != 200:
        raise HTTPException(400, f"Token GitHub invalid ({r.status_code})")
    repos = [{"full_name": x["full_name"], "private": x["private"],
              "default_branch": x.get("default_branch", "main")} for x in r.json()]
    user = requests.get("https://api.github.com/user", headers=headers, timeout=15)
    login = user.json().get("login") if user.status_code == 200 else None
    return {"login": login, "repos": repos}


@api_router.post("/github/commit")
async def github_commit(body: GithubCommitIn):
    files = body.files
    if body.project_id and not files:
        project = await db.projects.find_one({"id": body.project_id})
        if not project:
            raise HTTPException(404, "Proiect inexistent")
        files = project.get("files", [])
    if not files:
        raise HTTPException(400, "Niciun fisier de trimis.")

    headers = {"Authorization": f"Bearer {body.token}",
               "Accept": "application/vnd.github+json"}
    results = []
    for f in files:
        path = f["path"]
        url = f"https://api.github.com/repos/{body.repo}/contents/{path}"
        sha = None
        try:
            g = requests.get(url, params={"ref": body.branch}, headers=headers, timeout=15)
            if g.status_code == 200:
                sha = g.json().get("sha")
        except Exception:
            pass
        payload = {"message": body.message, "branch": body.branch,
                   "content": base64.b64encode(f["content"].encode()).decode()}
        if sha:
            payload["sha"] = sha
        try:
            p = requests.put(url, headers=headers, json=payload, timeout=20)
            ok = p.status_code in (200, 201)
            results.append({"path": path, "ok": ok, "status": p.status_code,
                            "error": None if ok else p.json().get("message", "")})
        except Exception as e:
            results.append({"path": path, "ok": False, "status": 0, "error": str(e)})

    committed = sum(1 for r in results if r["ok"])
    return {"repo": body.repo, "branch": body.branch,
            "committed": committed, "total": len(files), "results": results}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()