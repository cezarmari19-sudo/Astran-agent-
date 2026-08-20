from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form
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
import time
import requests
import zipfile
import io
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


def _parse_key_list(env_value: str) -> list:
    if not env_value:
        return []
    return [k.strip() for k in env_value.split(",") if k.strip()]


GEMINI_API_KEYS = _parse_key_list(os.environ.get('GEMINI_API_KEYS', ''))
if not GEMINI_API_KEYS and os.environ.get('GEMINI_API_KEY'):
    GEMINI_API_KEYS = [os.environ['GEMINI_API_KEY']]

ANTHROPIC_API_KEYS = _parse_key_list(os.environ.get('ANTHROPIC_API_KEYS', ''))
if not ANTHROPIC_API_KEYS and os.environ.get('ANTHROPIC_API_KEY'):
    ANTHROPIC_API_KEYS = [os.environ['ANTHROPIC_API_KEY']]

OPENAI_API_KEYS = _parse_key_list(os.environ.get('OPENAI_API_KEYS', ''))
if not OPENAI_API_KEYS and os.environ.get('OPENAI_API_KEY'):
    OPENAI_API_KEYS = [os.environ['OPENAI_API_KEY']]

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


# ---------------- LLM: direct calls to official SDKs, with per-provider key rotation ----------------
_key_cursor = {"gemini": 0, "anthropic": 0, "openai": 0}

RETRY_BETWEEN_KEYS_SECONDS = 10          # gap between trying key N and key N+1 in NORMAL use
RECOVERY_CHECK_INTERVAL_SECONDS = 10 * 60  # re-sweep ALL keys every 10 minutes when exhausted
RECOVERY_MAX_SECONDS = 30 * 60 * 60        # give up after 30 hours total

_provider_state = {
    "gemini": {"exhausted": False, "recovery_task": None, "recovered_event": None, "give_up": False},
    "anthropic": {"exhausted": False, "recovery_task": None, "recovered_event": None, "give_up": False},
    "openai": {"exhausted": False, "recovery_task": None, "recovered_event": None, "give_up": False},
}


def _provider_for_model(model: str) -> str:
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


async def _call_gemini_with_key(api_key: str, model: str, system_message: str, prompt: str) -> str:
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model_name=model, system_instruction=system_message)
    resp = await asyncio.to_thread(gmodel.generate_content, prompt)
    return resp.text or ""


async def _call_anthropic_with_key(api_key: str, model: str, system_message: str, prompt: str) -> str:
    aclient = anthropic.Anthropic(api_key=api_key)
    resp = await asyncio.to_thread(
        aclient.messages.create,
        model=model,
        max_tokens=8000,
        system=system_message,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in resp.content if hasattr(block, "text")]
    return "".join(parts)


async def _call_openai_with_key(api_key: str, model: str, system_message: str, prompt: str) -> str:
    oclient = openai.OpenAI(api_key=api_key)
    resp = await asyncio.to_thread(
        oclient.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


_CALL_FN = {
    "gemini": _call_gemini_with_key,
    "anthropic": _call_anthropic_with_key,
    "openai": _call_openai_with_key,
}
_KEYS_BY_PROVIDER = {
    "gemini": GEMINI_API_KEYS,
    "anthropic": ANTHROPIC_API_KEYS,
    "openai": OPENAI_API_KEYS,
}


async def _sweep_all_keys_once(provider: str, model: str) -> Optional[str]:
    """Fire a quick probe at EVERY key for this provider, back to back (no
    artificial delay between them), and return the first one that works —
    or None if all fail. This runs once per recovery cycle (every 10 min)."""
    keys = _KEYS_BY_PROVIDER[provider]
    call_fn = _CALL_FN[provider]
    for idx, key in enumerate(keys):
        try:
            await call_fn(key, model, "ping", "ping")
            return key
        except Exception as e:
            logger.warning(f"{provider} recovery sweep: key #{idx + 1}/{len(keys)} still failing: {e}")
    return None


async def _recovery_loop(provider: str, model: str):
    state = _provider_state[provider]
    started = time.monotonic()
    try:
        while True:
            working_key = await _sweep_all_keys_once(provider, model)
            if working_key:
                keys = _KEYS_BY_PROVIDER[provider]
                _key_cursor[provider] = keys.index(working_key)
                state["exhausted"] = False
                break
            if time.monotonic() - started >= RECOVERY_MAX_SECONDS:
                state["give_up"] = True
                break
            await asyncio.sleep(RECOVERY_CHECK_INTERVAL_SECONDS)
    finally:
        if state["recovered_event"]:
            state["recovered_event"].set()
        state["recovery_task"] = None


async def _wait_for_recovery(provider: str, model: str):
    """Block the caller until the shared recovery task finds a key, or gives up.
    Multiple concurrent requests share the SAME recovery task/sweep."""
    state = _provider_state[provider]
    if state["recovery_task"] is None:
        state["give_up"] = False
        state["recovered_event"] = asyncio.Event()
        state["recovery_task"] = asyncio.create_task(_recovery_loop(provider, model))
    else:
        if state["recovered_event"] is None:
            state["recovered_event"] = asyncio.Event()

    await state["recovered_event"].wait()

    if state["give_up"]:
        raise RuntimeError(
            f"Cheile {provider} par blocate de furnizor (nu doar cota depasita) — "
            f"nicio cheie nu a functionat in {RECOVERY_MAX_SECONDS // 3600} de ore."
        )


async def _call_with_key_rotation(provider: str, model: str, system_message: str, prompt: str) -> str:
    keys = _KEYS_BY_PROVIDER[provider]
    call_fn = _CALL_FN[provider]
    if not keys:
        raise RuntimeError(f"Nicio cheie configurata pentru {provider}")

    state = _provider_state[provider]

    if state["exhausted"] or state["recovery_task"] is not None:
        await _wait_for_recovery(provider, model)

    start = _key_cursor[provider] % len(keys)
    last_err = None
    for offset in range(len(keys)):
        idx = (start + offset) % len(keys)
        key = keys[idx]
        try:
            result = await call_fn(key, model, system_message, prompt)
            if result:
                _key_cursor[provider] = idx
                return result
        except Exception as e:
            last_err = e
            logger.warning(f"{provider} key #{idx + 1}/{len(keys)} failed: {e}")
            if offset < len(keys) - 1:
                await asyncio.sleep(RETRY_BETWEEN_KEYS_SECONDS)
            continue

    state["exhausted"] = True
    await _wait_for_recovery(provider, model)
    working_idx = _key_cursor[provider]
    return await call_fn(keys[working_idx], model, system_message, prompt)


async def llm_generate(system_message, prompt, session_id, model=None):
    model = model or GEMINI_MODEL
    provider = _provider_for_model(model)

    try:
        resp = await _call_with_key_rotation(provider, model, system_message, prompt)
        if resp:
            return resp
        raise RuntimeError("Raspuns gol de la model")
    except Exception as e:
        logger.error(f"LLM call failed ({provider}/{model}): {e}")
        raise HTTPException(
            status_code=502,
            detail=f"AI indisponibil ({provider}): {e}"
        )


AVAILABLE_MODELS = [
    {"id": "gemini-3.7-flash", "label": "Gemini 3.7 Flash", "hint": "Cel mai nou și capabil • cheile tale Gemini", "provider": "gemini"},
    {"id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash", "hint": "Rapid • cheile tale Gemini", "provider": "gemini"},
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash", "hint": "Rapid • cheile tale Gemini", "provider": "gemini"},
    {"id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash-Lite", "hint": "Cel mai rapid și ieftin • cheile tale Gemini", "provider": "gemini"},
    {"id": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro", "hint": "Puternic • cheile tale Gemini", "provider": "gemini"},
    {"id": "gemini-3.1-flash-lite", "label": "Gemini 3.1 Flash-Lite", "hint": "Frontier la cost redus • cheile tale Gemini", "provider": "gemini"},
    {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash", "hint": "Rapid • cheile tale Gemini", "provider": "gemini"},
    {"id": "claude-sonnet-5", "label": "Claude Sonnet 5", "hint": "Cheile tale Anthropic", "provider": "anthropic"},
    {"id": "claude-opus-4-8", "label": "Claude Opus 4.8", "hint": "Cheile tale Anthropic", "provider": "anthropic"},
    {"id": "claude-fable-5", "label": "Claude Fable 5", "hint": "Cheile tale Anthropic", "provider": "anthropic"},
    {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "hint": "Cheile tale Anthropic", "provider": "anthropic"},
    {"id": "gpt-5.5", "label": "ChatGPT 5.5", "hint": "Cheile tale OpenAI", "provider": "openai"},
    {"id": "gpt-5.4", "label": "ChatGPT 5.4", "hint": "Cheile tale OpenAI", "provider": "openai"},
]


BUILDER_SYSTEM = (
    "You are Aria, an elite senior full-stack & mobile engineer and product designer, "
    "the core of an AI app-builder that competes with the best. You PLAN carefully before coding, "
    "then write complete, production-ready, bug-free code with clean architecture. "
    "You obsess over beautiful, human, non-generic UI/UX (no AI-slop: avoid purple-on-white gradients, "
    "generic centered layouts, emoji icons; use real design systems, spacing, depth and polish). "
    "When the user asks to build or change an app/feature or write code, respond with: "
    "(1) a short friendly explanation and a concise numbered plan, then "
    "(2) the FULL code for every file you are creating or changing, each in this EXACT format:\n"
    "### FILE: relative/path/to/file.ext\n"
    "```lang\n<complete file content>\n```\n"
    "Always output complete files (never omit code with '...'). Keep answers focused. "
    "For questions that are not code, answer helpfully and concisely like a great engineer.\n\n"
    "If CURRENT PROJECT FILES are given in the prompt, that is the user's real, existing codebase "
    "(possibly uploaded by them, not written by you). Read it carefully and make the requested change "
    "on top of it — do not regenerate the whole project from scratch, and do not rewrite files that "
    "don't need to change. Only output ### FILE blocks for files you actually created or modified.\n\n"
    "If INSPIRATION / REFERENCE FILES are given, they are READ-ONLY context the user shared for style, "
    "structure or ideas. NEVER output ### FILE blocks for these paths, never copy them verbatim into the "
    "project, and never treat them as part of the editable codebase — only draw inspiration from them."
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
SENIOR_ENGINEER_PREFIX = (
    "You are a senior software engineer with 15+ years of production experience, doing a real code "
    "review — not a checklist exercise. Hold this code to the standard you'd apply before it ships to "
    "real users: correct behavior, strict typing where the language supports it, complete error handling "
    "(no silent failures, no swallowed exceptions, no unchecked nulls/undefined), consistent naming and "
    "style matching the rest of the codebase, and readable structure a teammate could maintain. "
    "For every issue you flag, explain briefly WHY it's a problem (what breaks, when, and for whom) — not "
    "just what to change. Before finishing, explicitly ask yourself: 'what else could go wrong here that "
    "I haven't checked yet, and what would make this app genuinely better, not just less broken?' — and "
    "act on anything real that surfaces from that question, even if it wasn't in your original focus.\n\n"
)

AGENT_DEFS = [
    {
        "key": "bruteforce",
        "label": "Brute-force / Bug hunter",
        "system": (
            SENIOR_ENGINEER_PREFIX +
            "Your focus: brute-force bug hunting. Find real bugs: crashes, unhandled errors, broken "
            "logic, race conditions, null/undefined access, off-by-one errors, broken async flows, "
            "incorrect state updates, memory/resource leaks. Do NOT comment on design or security secrets "
            "here — only functional correctness. For each bug, explain the exact failure scenario (what "
            "input or sequence of actions triggers it), then fix it properly — not just a patch that hides "
            "the symptom.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If genuinely no bugs after real scrutiny, return "
            '{"issues":[],"files":[],"summary":"No bugs found."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "beauty",
        "label": "Design / Beauty",
        "system": (
            SENIOR_ENGINEER_PREFIX +
            "Your focus: is the UI genuinely beautiful, polished, and production-grade? Flag weak "
            "spacing, poor color/contrast (check real accessibility contrast ratios, not just 'looks ok'), "
            "no visual hierarchy, ugly or inconsistent components, missing loading/empty/error states, "
            "bad typography choices, inconsistent spacing scale. For each issue, explain what a real "
            "designer would notice and why it undermines trust or usability. Fix every issue with real "
            "design improvements, not cosmetic patches.\n"
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
            SENIOR_ENGINEER_PREFIX +
            "Your focus: does this look HUMAN-made rather than generic AI output? Flag purple-on-white "
            "gradients, generic centered layouts, emoji used as icons, boilerplate placeholder copy, "
            "cookie-cutter structure, anything that screams 'AI wrote this' instead of a real product team "
            "with opinions and taste. For each issue, name specifically what pattern gives it away and "
            "what a real product team would have done instead. Fix every issue so the app feels crafted, "
            "not generated.\n"
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
            SENIOR_ENGINEER_PREFIX +
            "Your focus: simulate a NORMAL, non-technical user actually trying to use this app. Since you "
            "cannot literally run it, carefully trace through the code as if playing it: follow what "
            "happens on every screen, every button press, every navigation, every form submission, every "
            "edge case (empty input, slow network, going back mid-flow). Find points where a normal user "
            "would get confused, stuck, see a blank/broken state, or hit a dead end — and explain exactly "
            "what they'd be thinking/feeling at that moment. Fix every issue you find.\n"
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
            SENIOR_ENGINEER_PREFIX +
            "Your focus: act as an attacker trying to extract anything that should be server-side-only — "
            "API keys, secrets, tokens, internal URLs, credentials, database connection strings. Check if "
            "any secret is hardcoded in client code, sent to the client unnecessarily in an API response, "
            "logged, or exposed via an endpoint that leaks more than it should. For each finding, explain "
            "exactly how an attacker would extract it (inspect network traffic, decompile the app, read "
            "the bundle) and what damage they could do with it. Fix every exposure by moving secrets "
            "server-side and removing leaks.\n"
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
            SENIOR_ENGINEER_PREFIX +
            "Your focus: a careful line-by-line static code review, the kind a strict senior engineer "
            "would do before approving a pull request. Look for code smells, unused variables, dead code, "
            "inconsistent patterns, missing or incomplete error handling, type mismatches or loose typing, "
            "magic numbers/strings that should be named constants, duplicated logic that should be "
            "extracted, and anything that would slow down a future maintainer. For each finding, explain "
            "the maintainability or correctness cost of leaving it as-is. Fix what you find.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If the code is genuinely clean, return "
            '{"issues":[],"files":[],"summary":"Code is clean."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
    {
        "key": "cheater",
        "label": "Cheating / exploit user",
        "system": (
            SENIOR_ENGINEER_PREFIX +
            "Your focus: act as a user actively trying to CHEAT the app for unfair advantage — bypassing "
            "validation, manipulating client-side state to unlock things, sending malformed or crafted "
            "requests directly to the backend (skipping the UI entirely), exploiting logic to get free "
            "access, skip payment, or manipulate scores/results. For each exploit, explain the exact "
            "sequence of requests or actions that achieves it, and why client-side-only checks are never "
            "enough. Fix it by adding proper server-side validation and closing logic gaps.\n"
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
            SENIOR_ENGINEER_PREFIX +
            "Your focus: act as a hacker trying to break into this app's server and database, and attack "
            "the client side — injection attacks, missing auth checks on endpoints, insecure direct object "
            "references (accessing another user's data by guessing an ID), CORS misconfiguration, XSS, "
            "insecure storage of sensitive data on-device, unvalidated input reaching the database. For "
            "each vulnerability, explain the concrete attack path an intruder would use. Fix every "
            "vulnerability you find with proper server-side defenses, not client-side band-aids.\n"
            "Respond with STRICT JSON only, no markdown:\n"
            '{"issues":[{"severity":"high|medium|low","file":"path","description":"...","fix":"..."}],'
            '"files":[{"path":"...","content":"<full corrected file>"}],"summary":"..."}\n'
            "If nothing is vulnerable, return "
            '{"issues":[],"files":[],"summary":"No vulnerabilities found."}. '
            "Only include files you actually changed, with FULL new content."
        ),
    },
]


MAX_CONTEXT_FILE_CHARS = 6000
MAX_CONTEXT_TOTAL_CHARS = 60000


def build_files_context(files: list, inspiration_files: list) -> str:
    """Render existing editable files and read-only inspiration files as context
    blocks for the builder prompt, so Aria can see and extend an uploaded project."""
    parts = []
    if files:
        parts.append("### CURRENT PROJECT FILES (editable — modify/extend these as needed):")
        total = 0
        for f in files:
            content = f.get("content", "")
            if len(content) > MAX_CONTEXT_FILE_CHARS:
                content = content[:MAX_CONTEXT_FILE_CHARS] + "\n... [truncated, file is longer]"
            block = f"### FILE: {f['path']}\n```\n{content}\n```"
            if total + len(block) > MAX_CONTEXT_TOTAL_CHARS:
                parts.append("... [more files omitted for length]")
                break
            parts.append(block)
            total += len(block)
    if inspiration_files:
        parts.append(
            "\n### INSPIRATION / REFERENCE FILES (READ-ONLY — for style/context only, "
            "never edit or output these back, never merge them into the project files):"
        )
        total = 0
        for f in inspiration_files:
            content = f.get("content", "")
            if len(content) > MAX_CONTEXT_FILE_CHARS:
                content = content[:MAX_CONTEXT_FILE_CHARS] + "\n... [truncated]"
            block = f"### REF FILE: {f['path']}\n```\n{content}\n```"
            if total + len(block) > MAX_CONTEXT_TOTAL_CHARS:
                parts.append("... [more reference files omitted for length]")
                break
            parts.append(block)
            total += len(block)
    return "\n\n".join(parts)


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


# ---------------- ZIP upload: extract text files, skip binaries/junk ----------------
MAX_ZIP_BYTES = 30 * 1024 * 1024  # 30MB hard ceiling (well above the ~10MB typical case)
MAX_FILES_FROM_ZIP = 800
SKIP_DIR_PARTS = {
    "node_modules", ".git", ".expo", ".next", "dist", "build", "__pycache__",
    ".venv", "venv", ".idea", ".vscode", "ios/Pods", "android/.gradle",
}
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".icns",
    ".mp4", ".mov", ".avi", ".mp3", ".wav", ".ogg", ".flac",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".so", ".dylib", ".dll", ".exe", ".bin", ".class", ".jar",
    ".db", ".sqlite", ".sqlite3",
    ".lock",
}


def _looks_binary(data: bytes) -> bool:
    if b"\x00" in data[:4096]:
        return True
    try:
        data[:65536].decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def extract_zip_files(zip_bytes: bytes) -> list:
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise HTTPException(400, f"ZIP prea mare (max {MAX_ZIP_BYTES // (1024*1024)}MB).")

    extracted = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            for name in names:
                if len(extracted) >= MAX_FILES_FROM_ZIP:
                    break
                if name.endswith("/"):
                    continue
                parts = name.split("/")
                if any(p in SKIP_DIR_PARTS for p in parts):
                    continue
                if any(p.startswith(".") and p not in (".env",) for p in parts[:-1]):
                    continue
                ext = "." + name.rsplit(".", 1)[-1].lower() if "." in parts[-1] else ""
                if ext in SKIP_EXTENSIONS:
                    continue
                try:
                    info = zf.getinfo(name)
                    if info.file_size > 2 * 1024 * 1024:  # skip individual files over 2MB
                        continue
                    data = zf.read(name)
                except Exception:
                    continue
                if _looks_binary(data):
                    continue
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                # Normalize path: strip a single common top-level wrapper folder if present
                extracted.append({"path": name, "content": text})
    except zipfile.BadZipFile:
        raise HTTPException(400, "Fișierul nu este un ZIP valid.")

    if not extracted:
        raise HTTPException(400, "Nu s-a găsit niciun fișier text util în ZIP.")

    # Strip a single shared top-level folder (e.g. "myapp-main/") for cleaner paths
    top_levels = {f["path"].split("/", 1)[0] for f in extracted if "/" in f["path"]}
    if len(top_levels) == 1 and all("/" in f["path"] for f in extracted):
        prefix = next(iter(top_levels)) + "/"
        for f in extracted:
            f["path"] = f["path"][len(prefix):]

    return extracted


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
    return {"status": "ok", "ai_key_configured": bool(GEMINI_API_KEYS or ANTHROPIC_API_KEYS or OPENAI_API_KEYS)}


@api_router.get("/models")
async def list_models():
    return {
        "models": AVAILABLE_MODELS,
        "default": GEMINI_MODEL,
        "providers_available": {
            "gemini": bool(GEMINI_API_KEYS),
            "anthropic": bool(ANTHROPIC_API_KEYS),
            "openai": bool(OPENAI_API_KEYS),
        },
        "key_counts": {
            "gemini": len(GEMINI_API_KEYS),
            "anthropic": len(ANTHROPIC_API_KEYS),
            "openai": len(OPENAI_API_KEYS),
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
        "inspiration_files": [],
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


@api_router.post("/projects/{pid}/upload-zip")
async def upload_project_zip(pid: str, file: UploadFile = File(...), mode: str = Form("replace")):
    """Upload a ZIP of the user's own project. mode='replace' overwrites the editable
    files (Set 2) with the ZIP contents, so Aria can edit/extend it directly."""
    project = await db.projects.find_one({"id": pid})
    if not project:
        raise HTTPException(404, "Proiect inexistent")

    zip_bytes = await file.read()
    extracted = extract_zip_files(zip_bytes)

    if mode == "merge":
        merged = merge_files(project.get("files", []), extracted)
    else:
        merged = extracted

    await db.projects.update_one(
        {"id": pid},
        {"$set": {"files": merged, "updated_at": now_iso()}},
    )
    sys_msg = {
        "id": str(uuid.uuid4()), "project_id": pid, "role": "assistant",
        "content": f"Am încărcat proiectul tău din ZIP — {len(extracted)} fișiere. "
                    f"Poți să-mi ceri să adaug ceva nou sau să repar un bug.",
        "created_at": now_iso(), "msg_type": "normal",
    }
    await db.messages.insert_one(dict(sys_msg))
    return {"files_count": len(extracted), "all_files": merged}


@api_router.post("/projects/{pid}/upload-inspiration")
async def upload_inspiration_zip(pid: str, file: UploadFile = File(...)):
    """Upload a ZIP as read-only inspiration/reference — Aria can see it for context
    but never edits it, and it's never committed to GitHub."""
    project = await db.projects.find_one({"id": pid})
    if not project:
        raise HTTPException(404, "Proiect inexistent")

    zip_bytes = await file.read()
    extracted = extract_zip_files(zip_bytes)

    await db.projects.update_one(
        {"id": pid},
        {"$set": {"inspiration_files": extracted, "updated_at": now_iso()}},
    )
    return {"files_count": len(extracted)}


@api_router.delete("/projects/{pid}/inspiration")
async def clear_inspiration(pid: str):
    await db.projects.update_one({"id": pid}, {"$set": {"inspiration_files": []}})
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


# ---------------- Chat as background job (survives the app being closed) ----------------
CHAT_JOBS: dict = {}


def _new_chat_job(job_id, pid):
    return {
        "job_id": job_id,
        "project_id": pid,
        "done": False,
        "error": None,
        "stopped": False,
        "needs_clarification": False,
        "message": None,
        "auto_review_job_id": None,
    }


async def _run_chat_job(job_id: str, pid: str, message: str, model: Optional[str]):
    job = CHAT_JOBS[job_id]
    try:
        project = await db.projects.find_one({"id": pid})
        if not project:
            job["error"] = "Proiect inexistent"
            job["done"] = True
            return

        clear_stop(pid, "chat")

        history = await db.messages.find({"project_id": pid}).sort("created_at", 1).to_list(1000)
        in_clarification = bool(history) and history[-1].get("msg_type") == "clarify"

        if in_clarification:
            clar_thread = []
            for m in reversed(history):
                clar_thread.insert(0, m)
                if m["role"] == "user" and m.get("msg_type") != "clarify_answer":
                    break
            clar_thread.append({"role": "user", "content": message, "msg_type": "clarify_answer"})
        else:
            clar_thread = [{"role": "user", "content": message}]

        ts = now_iso()
        user_msg = {"id": str(uuid.uuid4()), "project_id": pid, "role": "user",
                    "content": message, "created_at": ts,
                    "msg_type": "clarify_answer" if in_clarification else "normal"}
        await db.messages.insert_one(dict(user_msg))

        decision = await run_clarifier(pid, clar_thread, model)

        if is_stopped(pid, "chat"):
            clear_stop(pid, "chat")
            job["stopped"] = True
            job["done"] = True
            return

        if decision.get("needs_clarification"):
            clar_msg = {
                "id": str(uuid.uuid4()), "project_id": pid, "role": "assistant",
                "content": decision.get("note", "Am nevoie de câteva detalii înainte să încep."),
                "created_at": now_iso(),
                "msg_type": "clarify",
                "questions": decision.get("questions", []),
            }
            await db.messages.insert_one(dict(clar_msg))
            job["needs_clarification"] = True
            job["message"] = clean(clar_msg)
            job["done"] = True
            return

        brief = decision.get("brief") or message

        recent = history[-12:]
        transcript = ""
        for m in recent:
            role = "User" if m["role"] == "user" else "Aria"
            transcript += f"{role}: {m['content']}\n\n"
        files_context = build_files_context(
            project.get("files", []), project.get("inspiration_files", [])
        )
        prompt = (
            f"Project: {project['name']}\nDescription: {project.get('description','')}\n\n"
            f"{files_context}\n\n"
            f"Conversation so far:\n{transcript}\nUser (clarified brief): {brief}\n\nAria:"
        )

        reply = await llm_generate(BUILDER_SYSTEM, prompt, f"proj-{pid}", model)

        if is_stopped(pid, "chat"):
            clear_stop(pid, "chat")
            job["stopped"] = True
            job["done"] = True
            return

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
            asyncio.create_task(_run_review(auto_job_id, pid, model))

        job["message"] = clean(ai_msg)
        job["auto_review_job_id"] = auto_job_id
        job["done"] = True
    except Exception as e:
        logger.error(f"chat job error: {e}")
        job["error"] = str(e)
        job["done"] = True


@api_router.post("/projects/{pid}/chat/start")
async def start_chat_job(pid: str, body: ChatIn, background_tasks: BackgroundTasks):
    project = await db.projects.find_one({"id": pid})
    if not project:
        raise HTTPException(404, "Proiect inexistent")
    job_id = str(uuid.uuid4())
    CHAT_JOBS[job_id] = _new_chat_job(job_id, pid)
    background_tasks.add_task(_run_chat_job, job_id, pid, body.message, body.model)
    return {"job_id": job_id}


@api_router.get("/chat/{job_id}")
async def chat_job_status(job_id: str):
    job = CHAT_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job inexistent")
    return job


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
            logger.info(f"github commit: GET {path} -> {g.status_code}")
            if g.status_code == 200:
                sha = g.json().get("sha")
        except Exception as e:
            logger.error(f"github commit: GET {path} failed: {e}")
        payload = {"message": body.message, "branch": body.branch,
                   "content": base64.b64encode(f["content"].encode()).decode()}
        if sha:
            payload["sha"] = sha
        try:
            p = requests.put(url, headers=headers, json=payload, timeout=20)
            ok = p.status_code in (200, 201)
            err_detail = None
            if not ok:
                try:
                    err_detail = p.json().get("message", p.text[:300])
                except Exception:
                    err_detail = p.text[:300]
                logger.error(f"github commit: PUT {path} -> {p.status_code}: {err_detail}")
            results.append({"path": path, "ok": ok, "status": p.status_code,
                            "error": None if ok else err_detail})
        except Exception as e:
            logger.error(f"github commit: PUT {path} raised: {e}")
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