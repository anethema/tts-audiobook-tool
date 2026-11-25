"""Read numeric prompt progress from a local Ollama llama.cpp worker, when available."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread, current_thread
from time import monotonic

import psutil
import requests


_POLL_SECONDS = 2.0
_DISCOVERY_SECONDS = 6.0


def _binary(process: psutil.Process) -> str:
    return Path(process.exe()).name.lower().removesuffix(".exe")


def _ollama_server() -> psutil.Process | None:
    for connection in psutil.net_connections(kind="tcp4"):
        if (connection.status == psutil.CONN_LISTEN and connection.laddr == ("127.0.0.1", 11434)
                and connection.pid):
            process = psutil.Process(connection.pid)
            if _binary(process) == "ollama" and process.cmdline()[1:2] == ["serve"]:
                return process
    return None


def _runner_ports(server: psutil.Process) -> list[tuple[psutil.Process, int]]:
    runners = []
    for child in server.children():
        try:
            if _binary(child) == "llama-server":
                for conn in child.net_connections(kind="tcp4"):
                    if conn.status == psutil.CONN_LISTEN and conn.laddr.ip == "127.0.0.1":
                        runners.append((child, conn.laddr.port))
        except (psutil.Error, OSError):
            continue
    return runners


def _read_slot(http: requests.Session, port: int, seed: int) -> tuple[int, int] | None:
    try:
        with http.get(f"http://127.0.0.1:{port}/slots", allow_redirects=False, timeout=0.5) as response:
            if response.status_code != 200:
                return None
            slots = response.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(slots, list):
        return None

    matches = []
    for slot in slots:
        if not isinstance(slot, dict) or slot.get("is_processing") is not True:
            continue
        params = slot.get("params")
        if not isinstance(params, dict) or params.get("seed") != seed:
            continue
        task = slot.get("id_task")
        done = slot.get("n_prompt_tokens_processed")
        # n_prompt_tokens is the current sequence length, not the final prompt size.
        if type(task) is int and type(done) is int and min(task, done) >= 0:
            matches.append((task, done))
    return matches[0] if len(matches) == 1 else None


def _observe(seed: int, report: Callable[[str], None], stopped: Event) -> None:
    # The normal chat stream works independently of this optional worker-specific API.
    try:
        if stopped.wait(_POLL_SECONDS):
            return
        server = _ollama_server()
        if server is None:
            return
        with requests.Session() as http:
            http.trust_env = False
            runners = []
            rediscover_at = 0.0
            active_task = None
            last_done = 0
            while not stopped.is_set() and server.is_running():
                now = monotonic()
                if active_task is None and now >= rediscover_at:
                    runners = _runner_ports(server)
                    rediscover_at = now + _DISCOVERY_SECONDS
                for runner, port in runners:
                    if not runner.is_running() or stopped.is_set():
                        continue
                    slot = _read_slot(http, port, seed)
                    if slot is None:
                        continue
                    task, done = slot
                    if active_task is not None and active_task != task:
                        return
                    active_task = task
                    runners = [(runner, port)]
                    if done > last_done and not stopped.is_set():
                        report(f"Qwen is reading the prompt · {done:,} input tokens processed")
                        last_done = done
                    break
                if stopped.wait(_POLL_SECONDS):
                    return
    except Exception:
        # Process permissions, unsupported runner layouts, and reporting failures are nonfatal.
        return


@contextmanager
def observe_prompt_reading(seed: int, on_activity: Callable[[str], None] | None):
    if on_activity is None:
        yield lambda: None
        return

    stopped = Event()
    worker = Thread(target=_observe, args=(seed, on_activity, stopped), name="qwen-prompt-progress", daemon=True)
    try:
        worker.start()
    except RuntimeError:
        yield lambda: None
        return

    def stop() -> None:
        stopped.set()
        if current_thread() is not worker:
            worker.join()

    try:
        yield stop
    finally:
        stop()
