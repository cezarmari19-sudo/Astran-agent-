from fastapi import FastAPI, APIRouter, HTTPException
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
GEMINI_MODEL = "gemini-3.1-pro-preview"

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


async def llm_generate(system_message, prompt, session_id):
    attempts = []
    if GEMINI_API_KEY:
        attempts.append(GEMINI_API_KEY)
    if EMERGENT_LLM_KEY and EMERGENT_LLM_KEY not in attempts:
        attempts.append(EMERGENT_LLM_KEY)
    last_err = None
    for key in attempts:
        try:
            chat = LlmChat(api_key=key, session_id=session_id,
                           system_message=system_message).with_model("gemini", GEMINI_MODEL)
            resp = await _call(chat, prompt)
            if resp:
                return resp
        except Exception as e:
            last_err = e
            logger.error(f"LLM attempt failed: {e}")
    raise HTTPException(status_code=502, detail=f"AI indisponibil: {last_err}")


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
    "You are a ruthless senior QA & code-review agent for an AI app-builder. "
    "You receive the current source files of a project. Find EVERY real issue: bugs, crashes, "
    "security leaks (especially leaked API keys / secrets), broken logic, missing error handling, "
    "bad or generic 'AI-looking' UI, accessibility and UX problems. Then FIX them. "
    "Respond with STRICT JSON only, no markdown, in this shape:\n"
    '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
    '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
    "If the code is genuinely clean and you find NOTHING to improve, return "
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
    return {"status