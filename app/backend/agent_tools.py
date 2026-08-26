"""
Normalized agentic tool-calling layer.

Each provider (Gemini, Anthropic, OpenAI) has a different function-calling
API shape. This module defines the tool schemas ONCE, in a
provider-neutral form, and provides adapters that:
  1. translate that schema into each provider's expected format,
  2. run one "turn" of the conversation (send messages, get back either a
     final text answer or a list of tool calls to execute),
  3. translate our tool results back into that provider's expected format
     for the next turn.

The actual agent loop (server.py) only interacts with `run_agent_turn` and
never needs to know which provider is underneath.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------- Tool schemas (provider-neutral) ----------------

TOOLS = [
    {
        "name": "run_command",
        "description": (
            "Execute a shell command inside the project's isolated sandbox container "
            "(e.g. 'npm install', 'npm test', 'python3 script.py', 'ls -la'). "
            "Returns stdout/stderr combined and the exit code. Use this to actually "
            "verify code works — install dependencies, run tests, run linters, start "
            "and curl a dev server. Has a hard timeout; long-running foreground "
            "processes will be killed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run in /workspace."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "write_files",
        "description": (
            "Write one or more files into the project workspace (creates directories "
            "as needed, overwrites existing files). Use this instead of printing "
            "### FILE blocks when you want the files to actually exist in the sandbox "
            "so you can run commands against them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path, e.g. 'src/App.tsx'."},
                            "content": {"type": "string", "description": "Full file content."},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            "required": ["files"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the current content of one file from the project workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files and directories in the project workspace (or a subfolder).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative subfolder to list. Defaults to the project root."},
            },
            "required": [],
        },
    },
]

MAX_AGENT_STEPS = 25  # hard cap so a confused agent can't loop forever burning API/compute


class ToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class AgentTurnResult:
    """Result of one back-and-forth with the model: either final text, or
    one or more tool calls the caller must execute and feed back."""
    def __init__(self, text: Optional[str], tool_calls: list[ToolCall], raw_assistant_message):
        self.text = text
        self.tool_calls = tool_calls
        self.raw_assistant_message = raw_assistant_message  # provider-native, needed to build history correctly

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


# ---------------- Gemini adapter ----------------

def _gemini_tools():
    import google.generativeai as genai
    return [genai.types.Tool(function_declarations=[
        {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
        for t in TOOLS
    ])]


async def _gemini_turn(api_key, model, system_message, history):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(
        model_name=model, system_instruction=system_message, tools=_gemini_tools()
    )
    chat = gmodel.start_chat(history=history[:-1])
    import asyncio
    resp = await asyncio.to_thread(chat.send_message, history[-1].parts)

    tool_calls = []
    text_parts = []
    for part in resp.candidates[0].content.parts:
        if hasattr(part, "function_call") and part.function_call and part.function_call.name:
            fc = part.function_call
            tool_calls.append(ToolCall(call_id=fc.name, name=fc.name, arguments=dict(fc.args)))
        elif hasattr(part, "text") and part.text:
            text_parts.append(part.text)

    return AgentTurnResult(
        text="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        raw_assistant_message=resp.candidates[0].content,
    )


# ---------------- Anthropic adapter ----------------

def _anthropic_tools():
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in TOOLS
    ]


async def _anthropic_turn(api_key, model, system_message, messages):
    import anthropic
    import asyncio
    aclient = anthropic.Anthropic(api_key=api_key)
    resp = await asyncio.to_thread(
        aclient.messages.create,
        model=model,
        max_tokens=8000,
        system=system_message,
        messages=messages,
        tools=_anthropic_tools(),
    )
    tool_calls = []
    text_parts = []
    for block in resp.content:
        if block.type == "tool_use":
            tool_calls.append(ToolCall(call_id=block.id, name=block.name, arguments=block.input))
        elif block.type == "text":
            text_parts.append(block.text)

    return AgentTurnResult(
        text="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        raw_assistant_message={"role": "assistant", "content": resp.content},
    )


# ---------------- OpenAI adapter ----------------

def _openai_tools():
    return [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["parameters"],
        }}
        for t in TOOLS
    ]


async def _openai_turn(api_key, model, system_message, messages):
    import openai
    import asyncio
    oclient = openai.OpenAI(api_key=api_key)
    full_messages = [{"role": "system", "content": system_message}] + messages
    resp = await asyncio.to_thread(
        oclient.chat.completions.create,
        model=model,
        messages=full_messages,
        tools=_openai_tools(),
    )
    msg = resp.choices[0].message
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(call_id=tc.id, name=tc.function.name, arguments=args))

    # IMPORTANT: raw_assistant_message must be a plain dict, not the SDK's
    # ChatCompletionMessage object — append_assistant_turn puts this
    # straight back into `messages` for the NEXT chat.completions.create
    # call, and the OpenAI client only accepts plain dicts there. Passing
    # the object through unconverted works for a single turn but raises
    # on the second agentic step once tool calls are involved.
    assistant_message: dict = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]

    return AgentTurnResult(
        text=msg.content,
        tool_calls=tool_calls,
        raw_assistant_message=assistant_message,
    )


_PROVIDER_TURN_FN = {
    "gemini": _gemini_turn,
    "anthropic": _anthropic_turn,
    "openai": _openai_turn,
}


async def run_agent_turn(provider: str, api_key: str, model: str, system_message: str, conversation) -> AgentTurnResult:
    """conversation is provider-native (built up by the caller's loop using
    the append_* helpers below). Returns either final text or tool calls to execute."""
    fn = _PROVIDER_TURN_FN.get(provider)
    if not fn:
        raise ValueError(f"Provider necunoscut pentru tool calling: {provider}")
    return await fn(api_key, model, system_message, conversation)


# ---------------- History helpers (provider-specific message shapes) ----------------

def new_conversation(provider: str, user_text: str):
    if provider == "gemini":
        import google.generativeai as genai
        return [genai.protos.Content(role="user", parts=[genai.protos.Part(text=user_text)])]
    if provider == "anthropic":
        return [{"role": "user", "content": user_text}]
    return [{"role": "user", "content": user_text}]  # openai


def append_assistant_turn(provider: str, conversation, turn: AgentTurnResult):
    if provider == "gemini":
        conversation.append(turn.raw_assistant_message)
    elif provider == "anthropic":
        conversation.append(turn.raw_assistant_message)
    else:
        conversation.append(turn.raw_assistant_message)
    return conversation


def append_tool_results(provider: str, conversation, tool_calls: list[ToolCall], results: list[str]):
    if provider == "gemini":
        import google.generativeai as genai
        parts = [
            genai.protos.Part(function_response=genai.protos.FunctionResponse(
                name=tc.name, response={"result": res}
            ))
            for tc, res in zip(tool_calls, results)
        ]
        conversation.append(genai.protos.Content(role="function", parts=parts))
    elif provider == "anthropic":
        content = [
            {"type": "tool_result", "tool_use_id": tc.call_id, "content": res}
            for tc, res in zip(tool_calls, results)
        ]
        conversation.append({"role": "user", "content": content})
    else:  # openai
        for tc, res in zip(tool_calls, results):
            conversation.append({
                "role": "tool", "tool_call_id": tc.call_id, "content": res,
            })
    return conversation
