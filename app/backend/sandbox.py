"""
Sandbox manager (no-Docker fallback): runs project commands directly on the
backend host, one isolated working directory per project, instead of one
Docker container per project.

Why this exists: the original Docker-based sandbox.py requires a reachable
Docker daemon (DOCKER_HOST or /var/run/docker.sock) on the machine running
the backend. Many deploy targets (Railway, Render, Fly.io without DinD,
plain containers, etc.) do not expose one, so sandbox_manager._get_client()
always raised SandboxError and the whole agent loop stopped with
"Agent mode nu a putut porni: sandbox-ul de execuție (Docker) nu este
disponibil pe acest server."

This module keeps the EXACT SAME public interface as the Docker version
(sandbox_manager.start/run_command/write_files/read_file/list_files,
SandboxError) so server.py does not need to change — only the import
target does (see bottom of file for how to swap it in).

Trade-offs vs. the Docker version (be aware of these):
- Isolation is much weaker: commands run as the backend process's own user,
  on the real filesystem, with no container/network/namespace boundary.
  Only a dedicated per-project directory + resource limits are enforced.
- No network isolation at all: a project's `npm install` (or anything else)
  has the same network access as the backend itself.
- Do NOT run this against untrusted multi-tenant input without adding a
  real sandboxing layer (e.g. firejail, bubblewrap, gVisor, or a hosted
  sandbox service like E2B/Modal/Daytona) in front of it. This fallback is
  meant to unblock local/single-tenant/dev use, not to be a security
  boundary.
"""

import asyncio
import logging
import os
import re
import resource
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------- Config ----------------
# Root directory under which every project gets its own subfolder.
# Override with SANDBOX_ROOT env var if you want it elsewhere (e.g. a disk
# with more space, or a tmpfs mount).
SANDBOX_ROOT = Path(os.environ.get("SANDBOX_ROOT", "/tmp/astran-sandboxes"))

PROJECT_IDLE_TIMEOUT_SECONDS = 20 * 60   # "forget" a project after 20 min unused
COMMAND_TIMEOUT_SECONDS = 120             # hard cap per command
MAX_OUTPUT_CHARS = 20_000                 # truncate huge output before it reaches the LLM
COMMAND_MEMORY_LIMIT_BYTES = 1 * 1024 * 1024 * 1024   # 1 GB, best-effort via RLIMIT_AS
COMMAND_CPU_TIME_LIMIT_SECONDS = 110      # best-effort via RLIMIT_CPU, just under the wall clock cap


class SandboxError(Exception):
    pass


class _ProjectHandle:
    def __init__(self, project_id: str, path: Path):
        self.project_id = project_id
        self.path = path
        self.last_used = time.monotonic()

    def touch(self):
        self.last_used = time.monotonic()


def _limit_resources():
    """Runs in the child process (via preexec_fn) right before exec, to cap
    memory and CPU time for the command being run. Best-effort: on some
    platforms/permissions this can silently no-op, which is fine — it's a
    secondary guard, not the only one (wall-clock timeout is primary)."""
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (COMMAND_MEMORY_LIMIT_BYTES, COMMAND_MEMORY_LIMIT_BYTES)
        )
    except Exception:
        pass
    try:
        resource.setrlimit(
            resource.RLIMIT_CPU, (COMMAND_CPU_TIME_LIMIT_SECONDS, COMMAND_CPU_TIME_LIMIT_SECONDS)
        )
    except Exception:
        pass
    try:
        # Detach from the parent's process group so we can kill the whole
        # subtree (e.g. `npm install` spawning children) on timeout.
        os.setsid()
    except Exception:
        pass


class SandboxManager:
    def __init__(self):
        self._projects: dict[str, _ProjectHandle] = {}
        self._lock = asyncio.Lock()
        self._reaper_task: Optional[asyncio.Task] = None

    async def start(self):
        """Call once at app startup to launch the idle-project reaper and
        make sure the sandbox root exists."""
        try:
            SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SandboxError(f"Nu pot crea directorul de sandbox '{SANDBOX_ROOT}': {e}")
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reap_loop())

    async def _reap_loop(self):
        while True:
            await asyncio.sleep(60)
            try:
                await self._reap_idle()
            except Exception as e:
                logger.warning(f"sandbox reaper error: {e}")

    async def _reap_idle(self):
        async with self._lock:
            now = time.monotonic()
            stale = [
                pid for pid, h in self._projects.items()
                if now - h.last_used > PROJECT_IDLE_TIMEOUT_SECONDS
            ]
            for pid in stale:
                # Only drop it from the in-memory map; the files on disk stay
                # (write_files/read_file mirror into Mongo separately in
                # server.py, and re-using the same folder next time is fine).
                self._projects.pop(pid, None)

    async def _get_or_create(self, project_id: str) -> _ProjectHandle:
        async with self._lock:
            handle = self._projects.get(project_id)
            if handle is not None and handle.path.is_dir():
                handle.touch()
                return handle

            safe_id = _safe_project_id(project_id)
            path = SANDBOX_ROOT / safe_id
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise SandboxError(f"Nu pot crea directorul de proiect '{path}': {e}")

            handle = _ProjectHandle(project_id, path)
            self._projects[project_id] = handle
            return handle

    async def write_files(self, project_id: str, files: list[dict]):
        """files: list of {"path": str, "content": str}. Writes them into
        the project's sandbox directory, creating subdirectories as needed."""
        handle = await self._get_or_create(project_id)

        for f in files:
            rel_path = _safe_relpath(f["path"])
            content = f.get("content", "")
            target = handle.path / rel_path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(target.write_text, content, encoding="utf-8")
            except OSError as e:
                raise SandboxError(f"Nu pot scrie fișierul '{f['path']}': {e}")

        handle.touch()

    async def read_file(self, project_id: str, path: str) -> str:
        handle = await self._get_or_create(project_id)
        rel_path = _safe_relpath(path)
        target = handle.path / rel_path
        handle.touch()
        try:
            return await asyncio.to_thread(target.read_text, encoding="utf-8")
        except FileNotFoundError:
            raise SandboxError(f"Nu pot citi {path}: fișierul nu există")
        except OSError as e:
            raise SandboxError(f"Nu pot citi {path}: {e}")

    async def list_files(self, project_id: str, subpath: str = ".") -> str:
        handle = await self._get_or_create(project_id)
        rel_path = _safe_relpath(subpath) if subpath != "." else "."
        base = handle.path / rel_path
        handle.touch()

        if not base.exists():
            return ""

        lines = []
        base_depth = len(base.parts)
        for root, dirnames, filenames in os.walk(base):
            root_path = Path(root)
            dirnames[:] = [d for d in dirnames if d not in ("node_modules", ".git")]
            depth = len(root_path.parts) - base_depth
            if depth >= 4:
                dirnames[:] = []
                continue
            lines.append(str(root_path))
            for fn in filenames:
                lines.append(str(root_path / fn))

        output = "\n".join(lines)
        return output[:MAX_OUTPUT_CHARS]

    async def run_command(self, project_id: str, command: str) -> dict:
        """Runs a shell command inside the project's sandbox directory with
        a hard timeout. Returns {exit_code, stdout, timed_out, truncated}."""
        handle = await self._get_or_create(project_id)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(handle.path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
        except OSError as e:
            raise SandboxError(f"Nu pot porni comanda: {e}")

        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=COMMAND_TIMEOUT_SECONDS
            )
            handle.touch()
            text = stdout.decode(errors="replace") if stdout else ""
            truncated = len(text) > MAX_OUTPUT_CHARS
            if truncated:
                text = text[:MAX_OUTPUT_CHARS] + "\n... [output truncat]"
            return {
                "exit_code": proc.returncode,
                "stdout": text,
                "timed_out": False,
                "truncated": truncated,
            }
        except asyncio.TimeoutError:
            await _kill_process_group(proc)
            return {
                "exit_code": -1,
                "stdout": f"[Comanda a depășit limita de {COMMAND_TIMEOUT_SECONDS}s și a fost întreruptă]",
                "timed_out": True,
                "truncated": False,
            }

    async def destroy(self, project_id: str):
        """Drops the project from the in-memory map and deletes its files
        from disk (mirrors the Docker version's container teardown)."""
        async with self._lock:
            handle = self._projects.pop(project_id, None)
        if handle is not None:
            try:
                await asyncio.to_thread(shutil.rmtree, handle.path, ignore_errors=True)
            except Exception as e:
                logger.warning(f"failed to remove sandbox dir for {project_id}: {e}")


async def _kill_process_group(proc: asyncio.subprocess.Process):
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, 9)  # SIGKILL
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        pass


def _safe_project_id(project_id: str) -> str:
    """Project ids come from our own DB, but sanitize anyway before using
    one as a directory name."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)
    if not safe:
        raise SandboxError("ID de proiect invalid")
    return safe


def _safe_relpath(path: str) -> str:
    """Prevent path traversal outside the project's sandbox directory.
    Rejects absolute paths and any '..' segment rather than trying to
    cleverly normalize them."""
    if not path or path.startswith("/") or path.startswith("~"):
        raise SandboxError(f"Cale invalidă (absolută): {path}")
    parts = path.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise SandboxError(f"Cale invalidă (conține '..'): {path}")
    return path.lstrip("/")


sandbox_manager = SandboxManager()

# ---------------------------------------------------------------------------
# HOW TO USE THIS FILE
#
# In app/backend/, this fallback does NOT require the `docker` pip package
# or a Docker daemon at all — it only uses the Python standard library.
#
# To switch server.py over to it:
#   1. Save this file as app/backend/sandbox.py (replacing the Docker
#      version), or as a new file e.g. app/backend/sandbox_local.py and
#      change the import in server.py:
#         from sandbox import sandbox_manager, SandboxError
#      to:
#         from sandbox_local import sandbox_manager, SandboxError
#   2. Nothing else in server.py needs to change — run_command, write_files,
#      read_file, list_files, and sandbox_manager.start() all have the same
#      signatures as before.
#   3. Optional: set SANDBOX_ROOT in app/backend/.env to point at a
#      directory with enough disk space if /tmp is constrained on your host.
# ---------------------------------------------------------------------------
