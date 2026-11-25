"""Run the browser's character and delivery analysis with Qwen on local Ollama."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import json
from time import monotonic

from pydantic_core import from_json
import requests
from urllib3.exceptions import ReadTimeoutError

from tts_audiobook_tool.app_types.delivery_cue import DeliveryCue
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult, SPEAKER_ASSIGN_LLM_TIMEOUT


LOCAL_QWEN_MODEL = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_M"
CONTEXT_LENGTH = 40960
_OLLAMA = "http://127.0.0.1:11434"
# Ollama can spend tens of minutes processing a chapter before it sends its first byte.
_CHAT_IDLE_MINUTES = 60
_CHAT_TIMEOUT = (SPEAKER_ASSIGN_LLM_TIMEOUT[0], _CHAT_IDLE_MINUTES * 60)
_CHAT_IDLE_MESSAGE = f"Ollama sent no data for {_CHAT_IDLE_MINUTES} minutes. Retry the analysis."


class LocalTaggerError(RuntimeError):
    """Local model setup or inference failed."""


def _writing_activity(fragments: list[str], characters: int, elapsed: float) -> str:
    details = ["Qwen is writing tags"]
    try:
        # Partial parsing omits unfinished strings. Only the full reply is used for analysis.
        draft = from_json("".join(fragments), allow_partial=True, cache_strings=False)
    except ValueError:
        draft = None
    if isinstance(draft, dict):
        speakers = draft.get("speakers")
        if isinstance(speakers, dict):
            count = sum(
                1 for key, label in speakers.items()
                if key.isdecimal() and isinstance(label, str) and label.strip()
            )
            details.append(f"{count:,} speaker label{'s' if count != 1 else ''} drafted")
        delivery = draft.get("delivery")
        if isinstance(delivery, dict):
            count = sum(1 for key, cue in delivery.items() if key.isdecimal() and DeliveryCue.from_dict(cue))
            details.append(f"{count:,} delivery cue{'s' if count != 1 else ''} drafted")
    details.extend((f"{characters:,} reply characters", f"{elapsed:.0f}s"))
    return " · ".join(details)


class LocalQwen:
    def __init__(
        self, http: requests.Session, log: Callable[[str], None],
        *, on_activity: Callable[[str], None] | None = None,
    ):
        self.http = http
        self.log = log
        self.on_activity = on_activity
        self.used = False

    def _post(self, route: str, **kwargs):
        timeout = _CHAT_TIMEOUT if route == "/api/chat" else SPEAKER_ASSIGN_LLM_TIMEOUT
        try:
            return self.http.post(
                _OLLAMA + route, allow_redirects=False, timeout=timeout, **kwargs,
            )
        except requests.ConnectionError as exc:
            raise LocalTaggerError("Cannot reach Ollama at 127.0.0.1:11434. Start Ollama and retry.") from exc
        except requests.Timeout as exc:
            if route == "/api/chat":
                raise LocalTaggerError(_CHAT_IDLE_MESSAGE) from exc
            raise LocalTaggerError(f"Ollama timed out during {route}.") from exc

    def prepare(self) -> None:
        if self.on_activity:
            self.on_activity("Checking for the local Qwen model…")
        with self._post("/api/show", json={"model": LOCAL_QWEN_MODEL}) as response:
            if response.status_code == 200:
                return
            if response.status_code != 404:
                raise LocalTaggerError(f"Ollama could not check the local model (HTTP {response.status_code}).")
        self.log("Downloading Qwen3.8-27B UD-Q5_K_M from Hugging Face.\n")
        if self.on_activity:
            self.on_activity("Downloading the local Qwen model…")
        with self._post("/api/pull", json={"model": LOCAL_QWEN_MODEL, "stream": True}, stream=True) as response:
            if response.status_code != 200:
                raise LocalTaggerError(f"Ollama could not download Qwen (HTTP {response.status_code}).")
            last_progress = None
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    update = json.loads(line)
                    if update.get("error"):
                        raise LocalTaggerError(f"Ollama download failed: {update['error']}")
                    if update.get("status") == "success":
                        return
                    total = update.get("total", 0)
                    if total:
                        percent = min(100, 100 * update.get("completed", 0) // total)
                        progress = (update.get("digest"), percent // 10)
                        if progress != last_progress:
                            detail = f"Downloading model layer ({total / 1e9:.2f} GB), {percent}%."
                            self.log(detail + "\n")
                            if self.on_activity:
                                self.on_activity(detail)
                            last_progress = progress
            except (ValueError, requests.RequestException) as exc:
                raise LocalTaggerError("Ollama download was interrupted. Retry to resume.") from exc
        raise LocalTaggerError("Ollama download ended before reporting success. Retry to resume.")

    def send(self, system: str, prompt: str) -> LlmRequestResult:
        if not self.used:
            self.prepare()
            self.used = True
            self.log(f"Tagging locally with {LOCAL_QWEN_MODEL} and a {CONTEXT_LENGTH:,}-token context.\n")
        started = monotonic()
        if self.on_activity:
            self.on_activity("Qwen is preparing the prompt…")
        with self._post("/api/chat", stream=True, json={
            "model": LOCAL_QWEN_MODEL, "stream": True, "format": "json", "think": False, "truncate": False,
            "options": {"num_ctx": CONTEXT_LENGTH, "temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }) as response:
            if response.status_code != 200:
                raise LocalTaggerError(f"Local Qwen tagging failed (HTTP {response.status_code}). Check the Ollama log.")
            return self._read_chat(response, started)

    def _read_chat(self, response: requests.Response, started: float) -> LlmRequestResult:
        fragments: list[str] = []
        characters = 0
        last_report = None
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                update = json.loads(line)
                if not isinstance(update, dict):
                    raise ValueError
                if "error" in update:
                    raise LocalTaggerError("Local Qwen tagging failed during streaming. Check the Ollama log.")
                fragment = update["message"]["content"]
                if not isinstance(fragment, str) or not isinstance(update.get("done"), bool):
                    raise ValueError
                if fragment:
                    fragments.append(fragment)
                    characters += len(fragment)
                    if self.on_activity:
                        now = monotonic()
                        if last_report is None or now - last_report >= 1:
                            self.on_activity(_writing_activity(fragments, characters, now - started))
                            last_report = now
                if update["done"]:
                    return LlmRequestResult(
                        "".join(fragments), "", monotonic() - started, finish_reason=update.get("done_reason"),
                        input_tokens=update.get("prompt_eval_count"), output_tokens=update.get("eval_count"),
                    )
        except (KeyError, TypeError, ValueError) as exc:
            raise LocalTaggerError("Ollama returned an incomplete chat response.") from exc
        except requests.RequestException as exc:
            # Requests wraps urllib3's socket read timeout in ConnectionError during iteration.
            if isinstance(exc, requests.ReadTimeout) or (exc.args and isinstance(exc.args[0], ReadTimeoutError)):
                raise LocalTaggerError(_CHAT_IDLE_MESSAGE) from exc
            raise LocalTaggerError("The Ollama response was interrupted. Retry the analysis.") from exc
        raise LocalTaggerError("Ollama ended the chat before reporting completion.")

    def unload(self) -> None:
        if self.used:
            try:
                with self._post("/api/generate", json={"model": LOCAL_QWEN_MODEL, "keep_alive": 0, "stream": False}) as response:
                    if response.status_code != 200:
                        raise LocalTaggerError(f"HTTP {response.status_code}")
            except (LocalTaggerError, requests.RequestException) as exc:
                self.log(f"Could not unload Qwen. Check Ollama before rendering ({exc}).\n")


@contextmanager
def local_qwen(log: Callable[[str], None], *, on_activity: Callable[[str], None] | None = None):
    with requests.Session() as http:
        http.trust_env = False
        tagger = LocalQwen(http, log, on_activity=on_activity)
        try:
            yield tagger
        finally:
            tagger.unload()
