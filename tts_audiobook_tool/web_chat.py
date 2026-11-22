"""Browser voice chat using the native conversation history and a scoped HTTP transport."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from hashlib import sha256
from ipaddress import ip_address
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from urllib.parse import urlsplit, urlunsplit

import requests

from tts_audiobook_tool.constants import ASSETS_DIR_NAME, CHAT_SYSTEM_PROMPTS, package_dir
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.web_backend import (
    WebBackendError, _request_error_message, _request_settings, _safe_error_code, _SanitizedHttpError,
)
from tts_audiobook_tool.web_models import ModelRunError, transcribe_model_reference


class WebChatError(ValueError):
    """A message safe to display in the browser chat panel."""


class ChatStopped(WebChatError):
    pass


@dataclass(frozen=True)
class ChatState:
    turns: tuple[tuple[str, str], ...] = ()
    context: str = ""
    ended: bool = False

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": role, "content": text}
            for pair in self.turns for role, text in zip(("user", "assistant"), pair, strict=True)
        ]


def default_system_prompt() -> str:
    return Path(package_dir, ASSETS_DIR_NAME, CHAT_SYSTEM_PROMPTS[0][0]).read_text(encoding="utf-8").strip()


def transcribe_chat_input(path: str | None, language: str) -> str:
    if not path:
        raise WebChatError("Record or upload a message first.")
    try:
        text = transcribe_model_reference(path, language_code=language or "en").strip()
    except (ModelRunError, OSError) as exc:
        raise WebChatError("Could not transcribe the message. Type it or retry transcription.") from exc
    if not text:
        raise WebChatError("No speech was recognized. Type your message or record again.")
    return text


def _is_local_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def prepare_chat_turn(
    state: ChatState, message: str, *, api_key: str, model: str, endpoint: str,
    effort: str, extra_json: str, system_prompt: str, echo: bool, stream: bool,
) -> ChatTurn:
    """Validate one draft before showing it in chat or starting any provider call."""
    if state.ended:
        raise WebChatError("Start a new chat to send another message.")
    message = message.strip()
    if not message:
        raise WebChatError("Enter or record a message first.")
    api_key = api_key.strip()
    system_prompt = system_prompt.strip()
    if api_key and any(api_key in field for field in (message, system_prompt, model, endpoint, extra_json)):
        raise WebChatError("Move the API key to the API key field before sending.")

    session = None
    context = "echo"
    if not echo:
        try:
            token, prefs = _request_settings(api_key, model, effort, endpoint, extra_json)
        except WebBackendError as exc:
            raise WebChatError(str(exc).replace("character analysis", "voice chat")) from None
        address = urlsplit(prefs.llm_url)
        if token and address.scheme != "https" and not _is_local_host(address.hostname or ""):
            raise WebChatError("Use an HTTPS endpoint when sending an API key to another machine.")
        context_fields = json.dumps([prefs.llm_url, prefs.llm_model, system_prompt], ensure_ascii=False)
        context = sha256(context_fields.encode()).hexdigest()
        session = _ChatLlmSession(
            prefs.llm_url, token, prefs.llm_model, system_prompt=system_prompt,
            extra_params=prefs.llm_speaker_assign_extra_params,
        )
        session.history = state.messages()
    if state.context and context != state.context:
        raise WebChatError("Start a new chat to change the provider, model, system prompt, or Echo mode.")
    return ChatTurn(state, message, context, session, stream)


class ChatTurn:
    def __init__(self, state: ChatState, message: str, context: str, llm: LlmSession | None, stream: bool):
        self.state = state
        self.message = message
        self.context = context
        self.llm = llm
        self.stream = stream
        self.cancel_event = Event()

    def stop(self) -> None:
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        if self.llm is not None:
            Thread(target=self.llm.cancel_active_request, name="stop-voice-chat-request", daemon=True).start()

    def commit(self, reply: str) -> ChatState:
        return replace(self.state, turns=(*self.state.turns, (self.message, reply)), context=self.context)

    def reply(self) -> Iterator[str]:
        """Yield cumulative reply text and leave the caller to commit only a successful turn."""
        if self.cancel_event.is_set():
            raise ChatStopped("Reply stopped.")
        if self.llm is None:
            yield self.message
            return

        events: Queue[tuple[str, str]] = Queue()

        def send():
            try:
                on_chunk = (lambda chunk: events.put(("chunk", chunk))) if self.stream else None
                result = self.llm.send(self.message, on_chunk, interrupt_event=self.cancel_event)
                if result.strip():
                    events.put(("reply", result))
                elif self.llm.hit_output_limit:
                    events.put(("error", "The LLM reached its output limit before producing a reply."))
                else:
                    events.put(("error", "The LLM returned an empty reply."))
            except WebChatError as exc:
                events.put(("error", str(exc)))
            except requests.RequestException as exc:
                message = (
                    _request_error_message(exc).replace("the OpenAI", "the LLM provider")
                    .replace("The OpenAI", "The LLM provider").replace("OpenAI", "the LLM provider")
                )
                events.put(("error", message[0].upper() + message[1:]))
            except Exception:
                events.put(("error", "Could not generate a reply. Check the LLM settings and retry."))

        Thread(target=send, name="voice-chat-request", daemon=True).start()
        text = ""
        finished = False
        try:
            while not self.cancel_event.is_set():
                try:
                    kind, value = events.get(timeout=0.1)
                except Empty:
                    continue
                if self.cancel_event.is_set():
                    break
                if kind == "chunk":
                    text += value
                    yield text
                    continue
                finished = True
                if kind == "error":
                    raise WebChatError(value)
                if text != value:
                    yield value
                return
            raise ChatStopped("Reply stopped.")
        finally:
            if not finished:
                self.stop()


class _ChatLlmSession(LlmSession):
    """Use the native chat protocol without ambient auth, redirects, or raw provider logs."""

    def __init__(self, endpoint: str, *args, **kwargs):
        address = urlsplit(endpoint)
        # The native request lifecycle logs api_endpoint_url. Log only the origin.
        super().__init__(urlunsplit((address.scheme, address.netloc, "", "", "")), *args, api_type="openai", **kwargs)
        self._request_url = endpoint

    def call_openai(self, payload, on_chunk, interrupt_event=None, on_reasoning=None) -> str:
        if interrupt_event is not None and interrupt_event.is_set():
            return ""
        if "temperature" not in self.extra_params:
            payload.pop("temperature", None)
        if urlsplit(self._request_url).hostname == "api.openai.com":
            payload["store"] = False
        if on_chunk is not None:
            payload["stream"] = True
        if self.token and self.token in json.dumps(payload, ensure_ascii=False):
            raise WebChatError("Move the API key to the API key field before sending.")

        try:
            with requests.Session() as http:
                http.trust_env = False
                with http.post(
                    self._request_url, headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
                    json=payload, timeout=self.timeout, allow_redirects=False, stream=True,
                ) as response:
                    self._register_active_response(response)
                    try:
                        if response.status_code != 200:
                            raise _SanitizedHttpError(response.status_code, _safe_error_code(response))
                        if on_chunk is not None:
                            return self._read_stream(response, on_chunk, interrupt_event)
                        choice = response.json()["choices"][0]
                        self.last_finish_reason = choice.get("finish_reason")
                        return self._content(choice["message"].get("content"))
                    finally:
                        self._clear_active_response(response)
        except _SanitizedHttpError:
            raise
        except requests.Timeout:
            raise requests.Timeout("The LLM request timed out.") from None
        except WebChatError:
            raise
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            raise WebChatError("The LLM provider returned an invalid reply.") from None
        except requests.RequestException:
            raise requests.ConnectionError("Could not connect to the LLM provider.") from None
        except Exception:
            raise WebChatError("Could not read the reply from the LLM provider.") from None

    def _content(self, content) -> str:
        if content is None:
            return ""
        if not isinstance(content, str) or (self.token and self.token in content):
            raise WebChatError("The LLM provider returned an invalid reply.")
        return content

    def _read_stream(self, response, on_chunk: Callable[[str], None], interrupt_event: Event | None) -> str:
        chunks = []
        hold = ""
        for line in response.iter_lines(chunk_size=1):
            if interrupt_event is not None and interrupt_event.is_set():
                break
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if not data:
                continue
            event = json.loads(data)
            if event.get("error"):
                raise WebChatError("The LLM provider returned an error during the reply.")
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if finish := choice.get("finish_reason"):
                self.last_finish_reason = finish
            text = self._content((choice.get("delta") or {}).get("content"))
            if not text:
                continue
            # Hold any suffix that could complete the pasted token in the next event.
            safe = self._content(hold + text)
            hold = ""
            for length in range(min(len(safe), len(self.token) - 1), 0, -1):
                if safe.endswith(self.token[:length]):
                    safe, hold = safe[:-length], safe[-length:]
                    break
            if safe:
                chunks.append(safe)
                on_chunk(safe)
        if hold and (interrupt_event is None or not interrupt_event.is_set()):
            chunks.append(hold)
            on_chunk(hold)
        return "".join(chunks)
