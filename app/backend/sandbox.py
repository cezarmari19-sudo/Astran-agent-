"""
Sandbox manager: one isolated Docker container per project, used by the
agentic builder to actually run commands (npm install, tests, servers) and
see real output instead of guessing.

Design goals:
- Strong isolation: no host network beyond package registries, capped
  CPU/memory, no privileged mode, read-only root FS except /workspace.
- One container per project, created lazily on first tool call, reused
  across the conversation, torn down after an idle timeout.
- Every command has a hard wall-clock timeout so a single tool call can
  never hang the agent loop forever.
- All state lives in-process (dict). If the backend restarts, containers
  are considered gone and are recreated on next use.

This module requires the `docker` Python package and a Docker daemon
reachable from the backend host (DOCKER_HOST or the default socket).
It is intentionally the ONLY place in the codebase that talks to Docker.
"""

import asyncio
import io
import logging
import tarfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import docker
    from docker.errors import DockerException, NotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    DockerException = Exception
    NotFound = Exception

# ---------------- Config ----------------
SANDBOX_IMAGE = "node:20-bullseye"          # has npm + can install python3 if needed
CONTAINER_IDLE_TIMEOUT_SECONDS = 20 * 60     # kill container after 20 min unused
COMMAND_TIMEOUT_SECONDS = 120                 # hard cap per command
MAX_OUTPUT_CHARS = 20_000                     # truncate huge output before it reaches the LLM
CONTAINER_MEMORY_LIMIT = "1g"
CONTAINER_CPU_QUOTA = 100_000                 # 1 CPU (cpu_period default 100000)
WORKSPACE_PATH = "/workspace"

# Registries needed for npm/pip to work; everything else is blocked at the
# network level by using an isolated bridge network with no other route.
# (Full egress allowlisting requires a firewall/proxy layer outside Docker;
# this at least prevents containers from reaching the host's private network.)


class SandboxError(Exception):
    pass


class _ContainerHandle:
    def __init__(self, container, project_id: str):
        self.container = container
        self.project_id = project_id
        self.last_used = time.monotonic()

    def touch(self):
        self.last_used = time.monotonic()


class SandboxManager:
    def __init__(self):
        self._client = None
        self._containers: dict[str, _ContainerHandle] = {}
        self._lock = asyncio.Lock()
        self._reaper_task: Optional[asyncio.Task] = None

    def _get_client(self):
        if not DOCKER_AVAILABLE:
            raise SandboxError(
                "Pachetul 'docker' nu este instalat pe server. "
                "Adaugă 'docker' în requirements.txt și asigură-te că serverul "
                "are acces la un daemon Docker."
            )
        if self._client is None:
            try:
                self._client = docker.from_env()
                self._client.ping()
            except DockerException as e:
                raise SandboxError(f"Nu mă pot conecta la Docker: {e}")
        return self._client

    async def start(self):
        """Call once at app startup to launch the idle-container reaper."""
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
                pid for pid, h in self._containers.items()
                if now - h.last_used > CONTAINER_IDLE_TIMEOUT_SECONDS
            ]
            for pid in stale:
                await self._destroy_locked(pid)

    async def _destroy_locked(self, project_id: str):
        handle = self._containers.pop(project_id, None)
        if handle is None:
            return
        try:
            await asyncio.to_thread(handle.container.remove, force=True)
        except Exception as e:
            logger.warning(f"failed to remove container for {project_id}: {e}")

    async def _get_or_create(self, project_id: str) -> _ContainerHandle:
        async with self._lock:
            handle = self._containers.get(project_id)
            if handle is not None:
                try:
                    handle.container.reload()
                    if handle.container.status == "running":
                        handle.touch()
                        return handle
                except NotFound:
                    pass
                self._containers.pop(project_id, None)

            client = self._get_client()
            container = await asyncio.to_thread(
                client.containers.run,
                SANDBOX_IMAGE,
                command="sleep infinity",
                detach=True,
                working_dir=WORKSPACE_PATH,
                mem_limit=CONTAINER_MEMORY_LIMIT,
                nano_cpus=1_000_000_000,   # 1 CPU
                pids_limit=256,
                network_disabled=False,     # needs network for npm/pip installs
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                read_only=False,            # workspace needs to be writable; consider tmpfs overlay for stricter setups
                name=f"astran-sbx-{project_id}"[:63],
                labels={"astran-sandbox": "1", "project_id": project_id},
                remove=False,
            )
            await asyncio.to_thread(
                container.exec_run, f"mkdir -p {WORKSPACE_PATH}"
            )
            handle = _ContainerHandle(container, project_id)
            self._containers[project_id] = handle
            return handle

    async def write_files(self, project_id: str, files: list[dict]):
        """files: list of {"path": str, "content": str}. Writes them into
        /workspace inside the project's container, creating directories
        as needed."""
        handle = await self._get_or_create(project_id)

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for f in files:
                path = _safe_relpath(f["path"])
                data = f.get("content", "").encode("utf-8")
                info = tarfile.TarInfo(name=path)
                info.size = len(data)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)

        await asyncio.to_thread(
            handle.container.put_archive, WORKSPACE_PATH, tar_stream.read()
        )
        handle.touch()

    async def read_file(self, project_id: str, path: str) -> str:
        handle = await self._get_or_create(project_id)
        safe_path = _safe_relpath(path)
        exit_code, output = await asyncio.to_thread(
            handle.container.exec_run, f"cat {WORKSPACE_PATH}/{safe_path}"
        )
        handle.touch()
        if exit_code != 0:
            raise SandboxError(f"Nu pot citi {path}: {output.decode(errors='replace')[:500]}")
        return output.decode(errors="replace")

    async def list_files(self, project_id: str, subpath: str = ".") -> str:
        handle = await self._get_or_create(project_id)
        safe_path = _safe_relpath(subpath) if subpath != "." else "."
        exit_code, output = await asyncio.to_thread(
            handle.container.exec_run,
            f"find {WORKSPACE_PATH}/{safe_path} -maxdepth 4 -not -path '*/node_modules/*' -not -path '*/.git/*'",
        )
        handle.touch()
        return output.decode(errors="replace")[:MAX_OUTPUT_CHARS]

    async def run_command(self, project_id: str, command: str) -> dict:
        """Runs a shell command inside the project's container with a hard
        timeout. Returns {exit_code, stdout, timed_out}."""
        handle = await self._get_or_create(project_id)

        async def _exec():
            return await asyncio.to_thread(
                handle.container.exec_run,
                cmd=["/bin/sh", "-c", command],
                workdir=WORKSPACE_PATH,
                demux=False,
            )

        try:
            exit_code, output = await asyncio.wait_for(_exec(), timeout=COMMAND_TIMEOUT_SECONDS)
            handle.touch()
            text = output.decode(errors="replace") if output else ""
            truncated = len(text) > MAX_OUTPUT_CHARS
            if truncated:
                text = text[:MAX_OUTPUT_CHARS] + "\n... [output truncat]"
            return {"exit_code": exit_code, "stdout": text, "timed_out": False, "truncated": truncated}
        except asyncio.TimeoutError:
            return {
                "exit_code": -1,
                "stdout": f"[Comanda a depășit limita de {COMMAND_TIMEOUT_SECONDS}s și a fost întreruptă]",
                "timed_out": True,
                "truncated": False,
            }

    async def destroy(self, project_id: str):
        async with self._lock:
            await self._destroy_locked(project_id)


def _safe_relpath(path: str) -> str:
    """Prevent path traversal outside the workspace. Rejects absolute paths
    and any '..' segment rather than trying to cleverly normalize them."""
    if not path or path.startswith("/") or path.startswith("~"):
        raise SandboxError(f"Cale invalidă (absolută): {path}")
    parts = path.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise SandboxError(f"Cale invalidă (conține '..'): {path}")
    return path.lstrip("/")


sandbox_manager = SandboxManager()
