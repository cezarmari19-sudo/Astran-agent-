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
import asyncio
import requests
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone

import google.generativeai as genai
import anthropic
import openai

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
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


# ---------------- Stop / cancellation registry ----------------
STOP_FLAGS: dict = {}


def _stop_flags(pid: str):
    if pid not in STOP_FLAGS:
        STOP_FLAGS[pid] = {"chat": False, "review": False}
    return STOP_FLAGS[pid]


def is_stopped(pid: str, kind: str) -> bool:
    return _stop_flags(pid).get(kind, False)


def clear_stop(pid: str, kind: str):
    _stop_flags(pid)[kind] = False


def request_stop(pid: str, kind: str):
    _stop_flags(pid)[kind] = True


# ---------------- LLM: direct calls to official SDKs, one key per provider ----------------
def _provider_for_model(model: str) -> str:
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


async def _call_gemini(model: str, system_message: str, prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY nu este configurata")
    genai.configure(api_key=GEMINI_API_KEY)
    gmodel = genai.GenerativeModel(model_name=model, system_instruction=system_message)
    resp = await asyncio.to_thread(gmodel.generate_content, prompt)
    return resp.text or ""


async def _call_anthropic(model: str, system_message: str, prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY nu este configurata")
    aclient = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = await asyncio.to_thread(
        aclient.messages.create,
        model=model,
        max_tokens=8000,
        system=system_message,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in resp.content if hasattr(block, "text")]
    return "".join(parts)


async def _call_openai(model: str, system_message: str, prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY nu este configurata")
    oclient = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = await asyncio.to_thread(
        oclient.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


async def llm_generate(system_message, prompt, session_id, model=None):
    model = model or GEMINI_MODEL
    provider = _provider_for_model(model)

    try:
        if provider == "gemini":
            resp = await _call_gemini(model, system_message, prompt)
        elif provider == "anthropic":
            resp = await _call_anthropic(model, system_message, prompt)
        else:
            resp = await _call_openai(model, system_message, prompt)
        if resp:
            return resp
        raise RuntimeError("Raspuns gol de la model")
    except Exception as e:
        logger.error(f"LLM call failed ({provider}/{model}): {e}")
        raise HTTPException(
            status_code=502,
            detail=f"AI indisponibil ({provider}): {e}. Verifica cheia API pentru acest provider in Setari."
        )


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


# ---------------- Agent #9: request clarifier ----------------
CLARIFIER_SYSTEM = (
    "You are a requirements clarification agent for an AI app-builder. A user just sent a request that "
    "will be turned into a real, complete application. Your ONLY job here is to decide: is this request "
    "detailed enough to start building, or is it genuinely too vague/generic to build something specific "
    "and good from it?\n\n"
    "Examples of VAGUE requests that need clarification: 'make me an app', 'build me a game like Among "
    "Us', 'create a social media app', 'make something for tracking stuff'.\n"
    "Examples of CLEAR requests that do NOT need clarification: 'change the button color to red', 'add a "
    "dark mode toggle in settings', 'fix the login screen', or any request detailed enough that a senior "
    "engineer could start building immediately without guessing major decisions.\n\n"
    "If the request is CLEAR, respond with STRICT JSON only:\n"
    '{"needs_clarification": false, "brief": "<the original request, optionally lightly cleaned up>"}\n\n'
    "If the request is VAGUE, respond with STRICT JSON only:\n"
    '{"needs_clarification": true, "questions": [{"question": "...", "options": ["...", "...", "..."]}], '
    '"note": "<one short friendly sentence to show the user>"}\n'
    "Ask at most 3 questions, each with 2-4 short concrete options (the user may also answer freely "
    "instead of picking an option, so options are just suggestions, not the only valid answers). Focus "
    "questions on things that materially change what gets built (genre/purpose, key features, visual "
    "style, target platform feel) — not trivial details.\n\n"
    "If you are given a CONVERSATION of previous clarification questions and answers plus a new user "
    "reply, decide whether you now have enough to build. Prefer moving forward unless it is truly still "
    "unclear. When you have enough, respond with needs_clarification=false and a rich 'brief' that "
    "combines everything learned into clear, detailed instructions for the engineer who will write the "
    "code."
)


def extract_json_generic(text):
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


async def run_clarifier(pid: str, latest_history: list, model: str = None):
    transcript = ""
    for m in latest_history:
        role = "User" if m["role"] == "user" else "Clarifier"
        transcript += f"{role}: {m['content']}\n\n"
    prompt = f"Conversation so far:\n\n{transcript}\nDecide now."
    raw = await llm_generate(CLARIFIER_SYSTEM, prompt, f"clarify-{pid}", model)
    data = extract_json_generic(raw)
    if not data:
        return {"needs_clarification": False, "brief": latest_history[-1]["content"]}
    return data


# ---------------- 8 specialized review agents ----------------
AGENT_DEFS = [
    {
        "key": "bruteforce",
        "label": "Brute-force / Bug hunter",
        "system": (
            "You are a brute-force bug hunter reviewing a generated app's source code. "
            "Find real bugs: crashes, unhandled errors, broken logic, race conditions, null/undefined "
            "access, off-by-one errors, broken async flows. Do NOT comment on design or security secrets "
            "here — only functional bugs. Fix every bug you find.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If genuinely no bugs, return "
            '{"issues":[],"files":[],"summary":"No bugs found."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "beauty",
        "label": "Design / Beauty",
        "system": (
            "You are a senior product designer reviewing whether a generated app's UI is genuinely "
            "beautiful and polished. Flag weak spacing, poor color/contrast, no visual hierarchy, ugly "
            "or inconsistent components, missing loading/empty/error states, bad typography choices. "
            "Fix every issue you find with real design improvements.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If genuinely beautiful already, return "
            '{"issues":[],"files":[],"summary":"Design is solid."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "human",
        "label": "Looks human, not AI-slop",
        "system": (
            "You are reviewing whether a generated app looks HUMAN-made rather than generic AI output. "
            "Flag: purple-on-white gradients, generic centered layouts, emoji used as icons, boilerplate "
            "placeholder copy, cookie-cutter structure, anything that screams 'AI wrote this'. "
            "Fix every issue you find so the app feels like it was crafted by a real product team.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If it already looks human-made, return "
            '{"issues":[],"files":[],"summary":"Looks human-made."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "normal_user",
        "label": "Plays as a normal user",
        "system": (
            "You are simulating a NORMAL, non-technical user trying to use this generated app. Since you "
            "cannot literally run the app, carefully trace through the code as if playing it: follow what "
            "happens on every screen, every button press, every navigation, every form submission. Find "
            "points where a normal user would get confused, stuck, see a blank/broken state, or hit a "
            "dead end. Fix every issue you find.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If a normal user could use this smoothly end-to-end, return "
            '{"issues":[],"files":[],"summary":"Smooth for a normal user."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "server_secrets",
        "label": "Server-side secret exposure",
        "system": (
            "You are an attacker trying to extract anything that should be server-side-only: API keys, "
            "secrets, tokens, internal URLs, credentials. Check if any secret is hardcoded in client code, "
            "sent to the client unnecessarily, logged, or exposed via an endpoint that leaks more than it "
            "should. Fix every exposure you find by moving secrets server-side and removing leaks.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If nothing is exposed, return "
            '{"issues":[],"files":[],"summary":"No secret exposure found."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "static_code_review",
        "label": "Static code review",
        "system": (
            "You are doing a careful line-by-line static code review of the source files. Look for code "
            "smells, unused variables, dead code, inconsistent patterns, missing error handling, type "
            "mismatches, and anything a senior engineer would flag in a real code review. Fix what you "
            "find.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If the code is clean, return "
            '{"issues":[],"files":[],"summary":"Code is clean."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "cheater",
        "label": "Cheating / exploit user",
        "system": (
            "You are a user trying to CHEAT the app to gain an unfair advantage: bypassing validation, "
            "manipulating client-side state to unlock things, sending malformed or crafted requests to "
            "the backend, exploiting logic to get free access, skip payment, or manipulate scores/results. "
            "Trace through the code to find where this is possible and fix it (add server-side validation, "
            "close logic gaps).\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If nothing is exploitable, return "
            '{"issues":[],"files":[],"summary":"No exploitable cheats found."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "hacker",
        "label": "Hacker / security break-in",
        "system": (
            "You are a hacker trying to break into this app's server and database, and attack the client "
            "side: injection attacks, missing auth checks on endpoints, insecure direct object references, "
            "CORS misconfiguration, XSS, insecure storage of sensitive data on-device, unvalidated input "
            "reaching the database. Fix every vulnerability you find.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If nothing is vulnerable, return "
            '{"issues":[],"files":[],"summary":"No vulnerabilities found."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
]


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
    return {"status": "ok", "ai_key_configured": bool(GEMINI_API_KEY or ANTHROPIC_API_KEY or OPENAI_API_KEY)}


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
    STOP_FLAGS.pop(pid, None)
    return {"ok": True}


@api_router.get("/projects/{pid}/messages")
async def get_messages(pid: str):
    docs = await db.messages.find({"project_id": pid}).sort("created_at", 1).to_list(1000)
    return [clean(d) for d in docs]


@api_router.post("/projects/{pid}/stop")
async def stop_project_work(pid: str, kind: str = "both"):
    """kind: 'chat' | 'review' | 'both'"""
    if kind in ("chat", "both"):
        request_stop(pid, "chat")
    if kind in ("review", "both"):
        request_stop(pid, "review")
    return {"ok": True, "stopped": kind}


@api_router.post("/projects/{pid}/chat")
async def project_chat(pid: str, body: ChatIn, background_tasks: BackgroundTasks):
    project = await db.projects.find_one({"id": pid})
    if not project:
        raise HTTPException(404, "Proiect inexistent")

    clear_stop(pid, "chat")

    history = await db.messages.find({"project_id": pid}).sort("created_at", 1).to_list(1000)

    in_clarification = bool(history) and history[-1].get("msg_type") == "clarify"

    if in_clarification:
        clar_thread = []
        for m in reversed(history):
            clar_thread.insert(0, m)
            if m["role"] == "user" and m.get("msg_type") != "clarify_answer":
                break
        clar_thread.append({"role": "user", "content": body.message, "msg_type": "clarify_answer"})
    else:
        clar_thread = [{"role": "user", "content": body.message}]

    ts = now_iso()
    user_msg = {"id": str(uuid.uuid4()), "project_id": pid, "role": "user",
                "content": body.message, "created_at": ts,
                "msg_type": "clarify_answer" if in_clarification else "normal"}
    await db.messages.insert_one(dict(user_msg))

    decision = await run_clarifier(pid, clar_thread, body.model)

    if is_stopped(pid, "chat"):
        clear_stop(pid, "chat")
        return {"reply": None, "files": [], "all_files": project.get("files", []),
                "message": None, "auto_review_job_id": None, "stopped": True}

    if decision.get("needs_clarification"):
        clar_msg = {
            "id": str(uuid.uuid4()), "project_id": pid, "role": "assistant",
            "content": decision.get("note", "Am nevoie de câteva detalii înainte să încep."),
            "created_at": now_iso(),
            "msg_type": "clarify",
            "questions": decision.get("questions", []),
        }
        await db.messages.insert_one(dict(clar_msg))
        return {
            "reply": clar_msg["content"], "files": [], "all_files": project.get("files", []),
            "message": clean(clar_msg), "auto_review_job_id": None, "stopped": False,
            "needs_clarification": True,
        }

    brief = decision.get("brief") or body.message

    recent = history[-12:]
    transcript = ""
    for m in recent:
        role = "User" if m["role"] == "user" else "Aria"
        transcript += f"{role}: {m['content']}\n\n"
    prompt = (
        f"Project: {project['name']}\nDescription: {project.get('description','')}\n\n"
        f"Conversation so far:\n{transcript}\nUser (clarified brief): {brief}\n\nAria:"
    )

    reply = await llm_generate(BUILDER_SYSTEM, prompt, f"proj-{pid}", body.model)

    if is_stopped(pid, "chat"):
        clear_stop(pid, "chat")
        return {"reply": None, "files": [], "all_files": project.get("files", []),
                "message": None, "auto_review_job_id": None, "stopped": True}

    ai_msg = {"id": str(uuid.uuid4()), "project_id": pid, "role": "assistant",
              "content": reply, "created_at": now_iso(), "msg_type": "normal"}
    await db.messages.insert_one(dict(ai_msg))

    new_files = parse_files(reply)
    merged = merge_files(project.get("files", []), new_files) if new_files else project.get("files", [])
    if new_files:
        await db.projects.update_one({"id": pid},
                                     {"$set": {"files": merged, "updated_at": now_iso()}})

    auto_job_id = None
    if new_files:
        clear_stop(pid, "review")
        auto_job_id = str(uuid.uuid4())
        REVIEW_JOBS[auto_job_id] = _new_review_job(auto_job_id, pid)
        background_tasks.add_task(_run_review, auto_job_id, pid, body.model)

    return {"reply": reply, "files": new_files, "all_files": merged,
            "message": clean(ai_msg), "auto_review_job_id": auto_job_id, "stopped": False,
            "needs_clarification": False}


REVIEW_JOBS = {}


def _new_review_job(job_id, pid):
    return {
        "job_id": job_id,
        "project_id": pid,
        "agents": {a["key"]: {"label": a["label"], "clean_streak": 0, "done": False} for a in AGENT_DEFS},
        "passes": [],
        "phase": "main",
        "final_round": 0,
        "files": [],
        "done": False,
        "error": None,
        "total_passes": 0,
    }


async def _run_single_agent_pass(pid, model, job, agent_def, current_files, pass_label):
    blob = "\n\n".join([f"### FILE: {p}\n```\n{c}\n```" for p, c in current_files.items()])
    prompt = f"{pass_label} — Current project files:\n\n{blob}"
    raw = await llm_generate(agent_def["system"], prompt, f"review-{pid}-{agent_def['key']}", model)
    data = extract_json(raw)
    if not data:
        data = {"issues": [{"severity": "low", "file": "-",
                            "description": "Agentul a raspuns in format liber.",
                            "fix": raw[:500]}], "files": [], "summary": "Format neuzual."}
    issues = data.get("issues", [])
    fixed = data.get("files", [])
    for f in fixed:
        if f.get("path"):
            current_files[f["path"]] = f.get("content", current_files.get(f["path"], ""))

    job["passes"].append({
        "agent": agent_def["key"],
        "agent_label": agent_def["label"],
        "label": pass_label,
        "issues": issues,
        "summary": data.get("summary", ""),
        "fixed_count": len(fixed),
    })
    job["total_passes"] = len(job["passes"])

    return len(issues) > 0


async def _persist(pid, job, current_files):
    final_files = [{"path": p, "content": c} for p, c in current_files.items()]
    await db.projects.update_one({"id": pid},
                                 {"$set": {"files": final_files, "updated_at": now_iso()}})
    job["files"] = final_files


async def _run_review(job_id, pid, model=None):
    job = REVIEW_JOBS[job_id]
    try:
        project = await db.projects.find_one({"id": pid})
        current = {f["path"]: f["content"] for f in project.get("files", [])}
        max_main_rounds = 30

        for round_num in range(max_main_rounds):
            if is_stopped(pid, "review"):
                job["phase"] = "stopped"
                job["done"] = True
                clear_stop(pid, "review")
                return
            any_agent_still_active = False
            for agent_def in AGENT_DEFS:
                if is_stopped(pid, "review"):
                    job["phase"] = "stopped"
                    job["done"] = True
                    clear_stop(pid, "review")
                    return
                astate = job["agents"][agent_def["key"]]
                if astate["done"]:
                    continue
                any_agent_still_active = True
                found_issue = await _run_single_agent_pass(
                    pid, model, job, agent_def, current,
                    f"Main round {round_num + 1} — {agent_def['label']}"
                )
                await _persist(pid, job, current)
                if found_issue:
                    astate["clean_streak"] = 0
                else:
                    astate["clean_streak"] += 1
                    if astate["clean_streak"] >= 2:
                        astate["done"] = True
            if not any_agent_still_active:
                break

        job["phase"] = "final"

        final_pattern = [2, 2, 1]
        max_final_restarts = 15
        restarts = 0
        while restarts < max_final_restarts:
            if is_stopped(pid, "review"):
                job["phase"] = "stopped"
                job["done"] = True
                clear_stop(pid, "review")
                return
            sequence_clean = True
            for stage_index, repeats in enumerate(final_pattern):
                for rep in range(repeats):
                    if is_stopped(pid, "review"):
                        job["phase"] = "stopped"
                        job["done"] = True
                        clear_stop(pid, "review")
                        return
                    for agent_def in AGENT_DEFS:
                        if is_stopped(pid, "review"):
                            job["phase"] = "stopped"
                            job["done"] = True
                            clear_stop(pid, "review")
                            return
                        found_issue = await _run_single_agent_pass(
                            pid, model, job, agent_def, current,
                            f"Final confirmation stage {stage_index + 1}/3 "
                            f"(pass {rep + 1}/{repeats}) — {agent_def['label']}"
                        )
                        await _persist(pid, job, current)
                        if found_issue:
                            sequence_clean = False
                if not sequence_clean:
                    break
            if sequence_clean:
                break
            restarts += 1

        job["phase"] = "complete"
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
    clear_stop(pid, "review")
    job_id = str(uuid.uuid4())
    REVIEW_JOBS[job_id] = _new_review_job(job_id, pid)
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