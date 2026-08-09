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
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash", "hint": "Rapid • recomandat"},
    {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash", "hint": "Rapid"},
    {"id": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro", "hint": "Puternic • poate cere plan"},
    {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro", "hint": "Puternic"},
    {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "hint": "Necesită credit Emergent"},
    {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5", "hint": "Necesită credit Emergent"},
    {"id": "gpt-5.4", "label": "GPT-5.4", "hint": "Necesită credit Emergent"},
    {"id": "gpt-5.4-mini", "label": "GPT-5.4 Mini", "hint": "Necesită credit Emergent"},
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
    return {"models": AVAILABLE_MODELS, "default": GEMINI_MODEL}


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


async def _run_review(job_id, pid):
    job = REVIEW_JOBS[job_id]
    try:
        project = await db.projects.find_one({"id": pid})
        current = {f["path"]: f["content"] for f in project.get("files", [])}
        clean_streak = 0
        max_passes = 8  # generous cap; loops until clean + 2 confirmations
        for i in range(max_passes):
            blob = "\n\n".join([f"### FILE: {p}\n```\n{c}\n```" for p, c in current.items()])
            prompt = f"Review pass #{i+1}. Current project files:\n\n{blob}"
            raw = await llm_generate(REVIEW_SY