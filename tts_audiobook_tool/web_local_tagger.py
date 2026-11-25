"""Run the browser's character and delivery analysis with Qwen on local Ollama."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import json
from time import monotonic

import requests

from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult, SPEAKER_ASSIGN_LLM_TIMEOUT


LOCAL_QWEN_MODEL = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_M"
CONTEXT_LENGTH = 40960
_OLLAMA = "http://127.0.0.1:11434"


class LocalTaggerError(RuntimeError):
    """Local model setup or inference failed."""


class LocalQwen:
    def __init__(self, http: requests.Session, log: Callable[[str], None]):
        self.http = http
        self.log = log
        self.used = False

    def _post(self, route: str, **kwargs):
        try:
            return self.http.post(
                _OLLAMA + route, allow_redirects=False, timeout=SPEAKER_ASSIGN_LLM_TIMEOUT, **kwargs,
            )
        except requests.ConnectionError as exc:
            raise LocalTaggerError("Cannot reach Ollama at 127.0.0.1:11434. Start Ollama and retry.") from exc
        except requests.Timeout as exc:
            raise LocalTaggerError(f"Ollama timed out during {route}.") from exc

    def prepare(self) -> None:
        with self._post("/api/show", json={"model": LOCAL_QWEN_MODEL}) as response:
            if response.status_code == 200:
                return
            if response.status_code != 404:
                raise LocalTaggerError(f"Ollama could not check the local model (HTTP {response.status_code}).")
        self.log("Downloading Qwen3.8-27B UD-Q5_K_M from Hugging Face.\n")
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
                            self.log(f"Downloading model layer ({total / 1e9:.2f} GB), {percent}%.\n")
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
        with self._post("/api/chat", json={
            "model": LOCAL_QWEN_MODEL, "stream": False, "format": "json", "think": False, "truncate": False,
            "options": {"num_ctx": CONTEXT_LENGTH, "temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }) as response:
            if response.status_code != 200:
                raise LocalTaggerError(f"Local Qwen tagging failed (HTTP {response.status_code}). Check the Ollama log.")
            try:
                body = response.json()
                reply = body["message"]["content"]
                if not isinstance(reply, str) or body.get("done") is not True:
                    raise ValueError
                return LlmRequestResult(
                    reply, "", monotonic() - started, finish_reason=body.get("done_reason"),
                    input_tokens=body.get("prompt_eval_count"), output_tokens=body.get("eval_count"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LocalTaggerError("Ollama returned an incomplete chat response.") from exc

    def unload(self) -> None:
        if self.used:
            try:
                with self._post("/api/generate", json={"model": LOCAL_QWEN_MODEL, "keep_alive": 0, "stream": False}) as response:
                    if response.status_code != 200:
                        raise LocalTaggerError(f"HTTP {response.status_code}")
            except (LocalTaggerError, requests.RequestException) as exc:
                self.log(f"Could not unload Qwen. Check Ollama before rendering ({exc}).\n")


@contextmanager
def local_qwen(log: Callable[[str], None]):
    with requests.Session() as http:
        http.trust_env = False
        tagger = LocalQwen(http, log)
        try:
            yield tagger
        finally:
            tagger.unload()
