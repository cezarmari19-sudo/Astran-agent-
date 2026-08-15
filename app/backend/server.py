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
import time
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