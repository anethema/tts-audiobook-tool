"""EPUB and speaker assignment operations for the local browser interface."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
import shutil
import tempfile
from time import monotonic
from urllib.parse import urlsplit
from typing import TYPE_CHECKING

import requests

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_DEFAULT
from tts_audiobook_tool.constants_config import PROJECT_DEFAULT_SEGMENTATION_STRATEGY
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan, PlannedChunk
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution
from tts_audiobook_tool.speaker_assign.attribution_prompt_util import ATTRIBUTION_SYSTEM_PROMPT
from tts_audiobook_tool.speaker_assign.attribution_reply_util import AttributionReplyUtil
from tts_audiobook_tool.speaker_assign.attribution_runner import AttributionRunner, AttributionRunListener
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult, SPEAKER_ASSIGN_LLM_TIMEOUT
from tts_audiobook_tool.speaker_assign.roster import Roster
from tts_audiobook_tool.speaker_assign.roster_util import ROSTER_SYSTEM_PROMPT, RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor

if TYPE_CHECKING:
    from tts_audiobook_tool.web_render import RenderVoice


DEFAULT_OPENAI_MODEL = "gpt-6.1-terra"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_LLM_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_FINISH_REASONS = ("stop", "length", "content_filter", "tool_calls", "function_call")
_WORKSPACE = tempfile.TemporaryDirectory(prefix="tts-audiobook-web-", ignore_cleanup_errors=True)


class WebBackendError(ValueError):
    """An error whose message can be shown in the browser."""


@dataclass(frozen=True, eq=False)
class BookSession:
    """A book revision with identity-based change tracking in the browser."""

    book: Book
    project_dir: str
    warnings: tuple[str, ...] = ()
    roster: Roster | None = None
    attributions: AttributionsFile = field(default_factory=AttributionsFile)
    attribution_notice: str = ""
    attribution_dir: str | None = None
    project_language: str | None = None
    word_substitutions: dict[str, str] = field(default_factory=dict)
    line_voices: dict[int, RenderVoice] = field(default_factory=dict)

    @property
    def language_code(self) -> str:
        if self.project_language is not None:
            return self.project_language
        return self.book.segmentation_settings.language_code or "en"

    def summary(self) -> str:
        return (
            f"{len(self.book.sections):,} sections · {len(self.book.phrase_groups):,} lines"
            f" · {len(self.book.dialog_line_indices()):,} dialogue lines"
        )


@dataclass(frozen=True)
class CharacterRow:
    speaker_id: str
    name: str
    line_count: int
    voice_hint: str


def import_epub(upload_path: str, *, language_code: str = "en") -> BookSession:
    path = Path(upload_path or "")
    if path.suffix.lower() != ".epub" or not path.is_file():
        raise WebBackendError("Choose an EPUB file to import.")
    language_code = language_code.strip() or "en"

    try:
        imported = EpubExtractor.import_epub(
            str(path), MAX_WORDS_PER_SEGMENT_DEFAULT, PROJECT_DEFAULT_SEGMENTATION_STRATEGY,
            language_code, dialog_segmentation=True,
        )
    except Exception:
        raise WebBackendError("The EPUB could not be read. Check that the file opens in an EPUB reader.") from None
    if not imported.phrase_groups:
        raise WebBackendError("The EPUB contains no readable text.")

    book = ProjectTextIOUtil.make_import_book(
        imported.phrase_groups, PROJECT_DEFAULT_SEGMENTATION_STRATEGY, MAX_WORDS_PER_SEGMENT_DEFAULT,
        language_code, dialog_segmentation=True, title=imported.book_title.strip() or path.stem,
        text_source_kind="epub", section_start_indices=imported.section_start_indices,
        section_titles=[chapter.title for chapter in imported.chapters],
    )
    return new_book_session(book, warnings=tuple(imported.significant_warnings))


def new_book_session(
        book: Book,
        *,
        project_dir: str | None = None,
        warnings: tuple[str, ...] = (),
        roster: Roster | None = None,
        attributions: AttributionsFile | None = None,
        project_language: str | None = None,
        word_substitutions: dict[str, str] | None = None,
        line_voices: dict[int, RenderVoice] | None = None,
) -> BookSession:
    """Create an isolated analysis workspace, optionally opening a saved book."""
    workspace = tempfile.mkdtemp(dir=_WORKSPACE.name)
    loaded = attributions if attributions is not None else AttributionsFile()
    session = BookSession(
        book=book,
        project_dir=project_dir or workspace,
        warnings=warnings,
        roster=roster,
        attributions=loaded,
        attribution_dir=workspace,
        project_language=project_language,
        word_substitutions=dict(word_substitutions or {}),
        line_voices=dict(line_voices or {}),
    )
    if loaded.ranges and SpeakerAssignFileUtil.save_attributions_file(workspace, loaded):
        close_book_session(session)
        raise WebBackendError("Saved dialogue assignments could not be copied into the session workspace.")
    return session


def close_book_session(session: BookSession | None) -> None:
    """Clear the temporary workspace for a book owned by this process."""
    if session is None:
        return
    project_dir = Path(session.attribution_dir or session.project_dir)
    if project_dir.resolve().parent == Path(_WORKSPACE.name).resolve():
        shutil.rmtree(project_dir, ignore_errors=True)


def character_rows(session: BookSession) -> list[CharacterRow]:
    roster = session.roster
    narrator = roster.narrator if roster else None
    narrator_id = narrator.id if narrator else "NARRATOR"
    characters = roster.characters if roster else ()
    counts = Counter(session.attributions.speaker_of_each_line(session.book).values())
    lines_for_other_characters = sum(counts[character.id] for character in characters if character.id != narrator_id)
    narrator_hint = narrator.voice_hint if narrator else ""
    if not narrator_hint and roster is not None:
        narrator_hint = roster.narration.voice_hint

    return [
        CharacterRow(
            narrator_id,
            narrator.name if narrator else "Narrator",
            len(session.book.phrase_groups) - lines_for_other_characters,
            narrator_hint,
        ),
        *(
            CharacterRow(character.id, character.name, counts[character.id], character.voice_hint)
            for character in characters if not character.is_narrator and character.id != narrator_id
        ),
    ]


def identify_characters(
        session: BookSession,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        effort: str = DEFAULT_REASONING_EFFORT,
        endpoint: str = DEFAULT_LLM_ENDPOINT,
        settings_json: str = "",
) -> BookSession:
    api_key, prefs = _request_settings(api_key, model, effort, endpoint, settings_json)
    if not session.book.dialog_line_indices():
        raise WebBackendError("This book has no dialogue lines to identify.")

    llm = _make_session(prefs, api_key, ROSTER_SYSTEM_PROMPT)
    result = _send(llm, RosterUtil.build_message(session.book))
    if result.error is not None:
        raise WebBackendError(_request_error_message(result.error))
    if result.hit_output_limit:
        raise WebBackendError("The model reached its output limit. Try lower reasoning effort or a shorter book.")
    try:
        roster = RosterUtil.normalize_roster(AttributionReplyUtil.extract_json(result.reply)).roster
    except ValueError:
        raise WebBackendError("The model did not return a readable character roster. Try again.") from None
    if SpeakerAssignFileUtil.delete_attributions_file(session.attribution_dir or session.project_dir):
        raise WebBackendError("The previous dialogue assignments could not be cleared from the session workspace.")
    return replace(session, roster=roster, attributions=AttributionsFile(), attribution_notice="")


def attribute_dialog(
        session: BookSession,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        effort: str = DEFAULT_REASONING_EFFORT,
        endpoint: str = DEFAULT_LLM_ENDPOINT,
        settings_json: str = "",
        on_progress: Callable[[int, int], None] | None = None,
) -> BookSession:
    if session.roster is None:
        raise WebBackendError("Identify the characters before attributing dialogue.")
    api_key, prefs = _request_settings(api_key, model, effort, endpoint, settings_json)

    attribution_dir = session.attribution_dir or session.project_dir
    attributions = _read_attributions(attribution_dir)
    plan = AssignPlan.make(session.book, attributions)
    progress = _ProgressReporter(on_progress, plan)
    progress.report()
    last_request: LlmRequestResult | None = None

    def send(llm: LlmSession, prompt: str) -> LlmRequestResult:
        nonlocal last_request
        last_request = _send(llm, prompt)
        return last_request

    runner = AttributionRunner(
        book=session.book,
        roster=session.roster,
        attributions=attributions,
        project_dir=attribution_dir,
        prefs=prefs,
        make_session=lambda: _make_session(prefs, api_key, ATTRIBUTION_SYSTEM_PROMPT),
        send=send,
        listener=progress,
    )
    result = runner.run()
    saved_attributions = _read_attributions(attribution_dir)
    complete = AssignPlan.make(session.book, saved_attributions).done_count
    total = len(plan.chunks)
    if complete < total:
        if last_request is not None and last_request.error is not None:
            notice = _request_error_message(last_request.error)
        elif last_request is not None and last_request.hit_output_limit:
            notice = "OpenAI reached its output limit. Try lower reasoning effort."
        elif result.error or result.failures:
            notice = "One or more dialogue batches could not be completed. Check the model and reasoning effort and retry."
        else:
            notice = "Some dialogue remains to be attributed. Run attribution again to continue."
        if not complete:
            raise WebBackendError(notice)
        notice = f"{complete:,} of {total:,} dialogue batches complete. {notice}"
    else:
        notice = f"{complete:,} dialogue batches complete."
    return replace(session, attributions=saved_attributions, attribution_notice=notice)


class _ProgressReporter(AttributionRunListener):
    def __init__(self, callback: Callable[[int, int], None] | None, plan: AssignPlan) -> None:
        self.callback = callback
        self.completed = plan.done_count
        self.total = len(plan.chunks)

    def report(self) -> None:
        if self.callback is not None:
            self.callback(self.completed, self.total)

    def chunk_saved(self, chunk: PlannedChunk, attribution: RangeAttribution, elapsed: float) -> None:
        self.completed += 1
        self.report()


def _read_attributions(project_dir: str) -> AttributionsFile:
    saved = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    if isinstance(saved, str):
        raise WebBackendError("Dialogue assignments could not be read from the session workspace.")
    return saved


def _request_settings(api_key: str, model: str, effort: str, endpoint: str, settings_json: str) -> tuple[str, Prefs]:
    endpoint = (endpoint or "").strip()
    try:
        address = urlsplit(endpoint)
        if (address.scheme not in ("http", "https") or not address.hostname
                or address.username is not None or address.fragment
                or (address.hostname == "api.openai.com" and address.scheme != "https")):
            raise ValueError
    except ValueError:
        raise WebBackendError("Enter a full HTTP or HTTPS chat completions endpoint URL.") from None

    api_key = api_key.strip()
    if (address.hostname == "api.openai.com" or api_key) and not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", api_key):
        raise WebBackendError("Paste a valid OpenAI API key.")
    model = model.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", model) or (api_key and api_key in model):
        raise WebBackendError("Enter a model ID.")
    effort = effort.strip()
    if effort and effort not in _REASONING_EFFORTS:
        raise WebBackendError("Choose a supported reasoning effort.")
    try:
        extra_params = json.loads(settings_json.strip() or "{}")
        if not isinstance(extra_params, dict):
            raise ValueError
    except ValueError:
        raise WebBackendError("Enter additional request parameters as a JSON object.") from None
    if extra_params.keys() & {"messages", "model", "store", "stream"}:
        raise WebBackendError("The app sets messages, model, store and stream for character analysis.")
    if effort:
        extra_params.setdefault("reasoning_effort", effort)
    return api_key, Prefs(
        llm_url=endpoint,
        llm_model=model,
        llm_speaker_assign_extra_params=extra_params,
    )


def _make_session(prefs: Prefs, api_key: str, system_prompt: str) -> LlmSession:
    return _WebOpenAISession(
        api_endpoint_url=prefs.llm_url,
        token=api_key,
        model=prefs.llm_model,
        api_type="openai",
        system_prompt=system_prompt,
        extra_params=prefs.llm_speaker_assign_extra_params,
        timeout=SPEAKER_ASSIGN_LLM_TIMEOUT,
    )


def _send(llm: LlmSession, prompt: str) -> LlmRequestResult:
    started = monotonic()
    try:
        reply = llm.send(prompt)
    except (requests.RequestException, WebBackendError) as error:
        return LlmRequestResult("", "", monotonic() - started, error=error)
    return LlmRequestResult(reply, "", monotonic() - started, finish_reason=llm.last_finish_reason)


class _SanitizedHttpError(requests.HTTPError):
    """A provider status with no response text, request headers or raw JSON."""

    def __init__(self, status: int, code: str = "") -> None:
        response = requests.Response()
        response.status_code = status
        super().__init__(f"OpenAI request failed with HTTP {status}.", response=response)
        self.code = code


def _request_error_message(error: Exception) -> str:
    if isinstance(error, _SanitizedHttpError):
        if error.code == "context_length_exceeded":
            return "This text exceeds the OpenAI model’s context limit. Try a model with a larger context window."
        if error.code == "insufficient_quota":
            return "OpenAI reported insufficient API quota. Check billing and usage for the API key."
        if error.response is not None:
            status = error.response.status_code
            if status in (401, 403):
                return "OpenAI denied the request. Check the API key and its project permissions."
            if status == 429:
                return "OpenAI rate-limited the request. Wait briefly and retry."
            if status in (400, 404):
                return f"OpenAI rejected the request (HTTP {status}). Check the model and reasoning effort."
        return str(error)
    if isinstance(error, requests.Timeout):
        return "The OpenAI request timed out. Try again."
    if isinstance(error, requests.ConnectionError):
        return "Could not connect to OpenAI. Check the network connection and retry."
    return "OpenAI did not return a usable answer. Try again."


class _WebOpenAISession(LlmSession):
    """Use the upstream chat state with a transport scoped to a pasted key."""

    def call_openai(self, payload, on_chunk, interrupt_event=None, on_reasoning=None) -> str:
        if "temperature" not in self.extra_params:
            payload.pop("temperature", None)
        if urlsplit(self.api_endpoint_url).hostname == "api.openai.com":
            payload["store"] = False
        if self.token and any(self.token in message["content"] for message in payload["messages"]):
            raise WebBackendError("The request text contains the API key. Remove it from the text before retrying.")
        try:
            with requests.Session() as http:
                # Disable .netrc auth overrides and ambient proxy configuration.
                http.trust_env = False
                with http.post(
                    self.api_endpoint_url,
                    headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
                    json=payload,
                    timeout=self.timeout,
                    allow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        raise _SanitizedHttpError(response.status_code, _safe_error_code(response))
                    choice = response.json()["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason")
                    self.last_finish_reason = finish_reason if finish_reason in _FINISH_REASONS else None
        except _SanitizedHttpError:
            raise
        except requests.Timeout:
            raise requests.Timeout("OpenAI request timed out.") from None
        except requests.RequestException:
            raise requests.ConnectionError("Could not connect to OpenAI.") from None
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            raise WebBackendError("OpenAI returned an invalid response.") from None
        if content is None:
            return ""
        if not isinstance(content, str) or (self.token and self.token in content):
            raise WebBackendError("OpenAI returned an invalid response.")
        return content


def _safe_error_code(response: requests.Response) -> str:
    try:
        code = response.json()["error"]["code"]
    except (KeyError, TypeError, ValueError):
        return ""
    return code if code in ("context_length_exceeded", "insufficient_quota") else ""
