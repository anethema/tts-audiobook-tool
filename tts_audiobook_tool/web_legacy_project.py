"""Read-only import of locally known console projects into the browser."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from tts_audiobook_tool.app_support import path_norm
from tts_audiobook_tool.app_types import Book, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.book_serialization import (
    book_from_json_dict,
    book_from_project_text_json_dict,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import VOICE_CROP_SUBDIR, Voice
from tts_audiobook_tool.constants import (
    APP_USER_SUBDIR,
    PROJECT_JSON_FILE_NAME,
    PROJECT_SPEC_VERSION,
    PROJECT_TEXT_EPUB_FILE_NAME,
    PROJECT_TEXT_FILE_NAME,
    PROJECT_VOICE_SUBDIR,
)
from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_MAX
from tts_audiobook_tool.prefs import PREFS_FILE_NAME
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.voice_reference_migration import (
    VoiceReferenceMigrationRequired,
    prepare_project_voice_references,
)
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterFile
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_backend import BookSession, WebBackendError, close_book_session, new_book_session
from tts_audiobook_tool.web_render import RenderVoice


_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".m4a", ".aac", ".mp4", ".ogg", ".opus", ".aif", ".aiff", ".webm", ".wma"})


class WebLegacyProjectError(ValueError):
    """A console import error suitable for display in the browser."""


@dataclass(frozen=True)
class ConsoleProject:
    project_id: str
    name: str


@dataclass(frozen=True)
class ImportedConsoleProject:
    name: str
    session: BookSession
    narrator: RenderVoice
    voices: dict[str, RenderVoice]
    model_id: str
    source_epub: str | None
    validate_audio: bool = True
    batch_size: int = 1


def list_console_projects(*, prefs_path: str | os.PathLike[str] | None = None) -> list[ConsoleProject]:
    """Discover project directories recorded in the console preferences."""
    return [ConsoleProject(project_id, folder.name) for project_id, folder in _known_projects(prefs_path).items()]


def load_console_project(
        project_id: str,
        *,
        model_choices: Iterable[tuple[str, str]] = (),
        prefs_path: str | os.PathLike[str] | None = None,
) -> ImportedConsoleProject:
    """Copy selected project content into an independent, unsaved browser session."""
    folder = _known_projects(prefs_path).get(project_id)
    if folder is None:
        raise WebLegacyProjectError("Choose a project from the console project list.")

    settings_path = _contained_file(folder, PROJECT_JSON_FILE_NAME)
    try:
        if settings_path is None:
            raise OSError
        settings = _read_json(settings_path)
        if not isinstance(settings, dict):
            raise ValueError
    except (OSError, ValueError, RecursionError):
        raise WebLegacyProjectError("The console project settings could not be read.") from None

    if isinstance(settings.get("version"), int) and settings["version"] > PROJECT_SPEC_VERSION:
        raise WebLegacyProjectError("Update the app before importing this console project.")

    warnings = []
    raw_references = settings.get("voice_references")
    try:
        settings = prepare_project_voice_references(settings)
        if isinstance(raw_references, list) and len(raw_references) != len(settings["voice_references"]):
            settings["voice_references"] = [
                Voice.from_dict(raw, index=index) or Voice()
                for index, raw in enumerate(raw_references)
            ]
            warnings.append("Some console voice slots contain unavailable reference clips.")
    except VoiceReferenceMigrationRequired:
        warnings.append("Open this project in the console to choose its voice list before importing voices.")
        ProjectLoadUtil.remap_legacy_keys(settings)
        settings["voice_references"] = []
    except ValueError:
        raise WebLegacyProjectError("The console project voice settings could not be read.") from None

    try:
        book = _load_book(folder, settings)
    except (OSError, TypeError, ValueError, RecursionError):
        raise WebLegacyProjectError("The console project text could not be read.") from None
    if not isinstance(book, Book) or not book.phrase_groups:
        raise WebLegacyProjectError("This console project has no readable book text.")
    if not book.title:
        book.title = folder.name
    language = settings.get("language_code")
    if not book.segmentation_settings.language_code and isinstance(language, str):
        book.segmentation_settings = book.segmentation_settings._replace(language_code=language)

    roster, attributions, voice_map = _load_speakers(folder, book, warnings)
    tts_id = settings.get("tts_model_type")
    tts_id = tts_id if isinstance(tts_id, str) else ""
    model_id = _installed_model(tts_id, model_choices)
    if tts_id not in ("", "none") and not model_id:
        warnings.append("The console project’s speech model is unavailable in the browser. Choose an installed model.")

    try:
        session = new_book_session(book, roster=roster, attributions=attributions)
    except (OSError, WebBackendError):
        raise WebLegacyProjectError("Could not open a workspace for the imported project.") from None
    staged = Path(session.project_dir)
    references = settings["voice_references"]
    copied: dict[Path, str] = {}
    narrator_id = roster.narrator.id if roster and roster.narrator else None
    cast_slots = {
        speaker: index for speaker, index in voice_map.items()
        if roster and speaker in roster.ids and speaker != narrator_id and index < len(references)
    }
    try:
        narrator = RenderVoice()
        if references:
            narrator = _stage_voice(folder, staged, references[0], tts_id, copied, warnings)
        voices = {
            speaker: _stage_voice(folder, staged, references[index], tts_id, copied, warnings)
            for speaker, index in cast_slots.items()
        }
        source = _contained_file(folder, PROJECT_TEXT_EPUB_FILE_NAME)
        source_epub = None
        if source is not None:
            source_epub = str(staged / PROJECT_TEXT_EPUB_FILE_NAME)
            shutil.copyfile(source, source_epub)
    except OSError:
        close_book_session(session)
        raise WebLegacyProjectError("Could not copy the console project’s book or voice files.") from None

    speakers_by_line = attributions.speaker_of_each_line(book)
    if any(
        group.voice_index != 0 and cast_slots.get(speakers_by_line.get(index, ""), 0) != group.voice_index
        for index, group in enumerate(book.phrase_groups)
    ):
        warnings.append("The browser uses its character cast for voice assignment. Review the imported voices before generating.")

    return ImportedConsoleProject(
        folder.name, replace(session, warnings=tuple(dict.fromkeys(warnings))), narrator, voices, model_id, source_epub,
        batch_size=_batch_size(settings, tts_id),
    )


def _known_projects(prefs_path: str | os.PathLike[str] | None) -> dict[str, Path]:
    path = Path(prefs_path) if prefs_path is not None else Path.home() / APP_USER_SUBDIR / PREFS_FILE_NAME
    try:
        prefs = _read_json(path)
    except (OSError, ValueError, RecursionError):
        return {}
    if not isinstance(prefs, dict):
        return {}

    projects = {}
    for key in ("project_dir", "last_project_dir"):
        value = prefs.get(key)
        if not isinstance(value, str) or value.startswith(("//", "\\\\")) or path_norm.looks_foreign(value):
            continue
        try:
            path = Path(value).expanduser()
            if not path.is_absolute() or path.is_symlink():
                continue
            folder = path.resolve(strict=True)
            if _contained_file(folder, PROJECT_JSON_FILE_NAME) is None:
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        project_id = sha256(os.fsencode(folder)).hexdigest()
        projects[project_id] = folder
    return projects


def _load_book(folder: Path, settings: dict) -> Book | str:
    if "book" in settings:
        book = settings["book"]
        if isinstance(book, dict) and "book" in book:
            return book_from_project_text_json_dict(book)
        return book_from_json_dict(book)
    for field in ("text", "phrase_groups"):
        if field in settings:
            return book_from_project_text_json_dict(settings[field], _legacy_segmentation(settings))
    if "text_segments" in settings:
        phrases = Phrase.phrases_from_json_dicts(settings["text_segments"])
        if isinstance(phrases, str):
            return phrases
        groups = []
        for index, phrase in enumerate(phrases):
            reason = phrases[index + 1].reason if index + 1 < len(phrases) else Reason.UNDEFINED
            groups.append(PhraseGroup([Phrase(phrase.text, reason)]))
        return book_from_project_text_json_dict([group.to_json_dict() for group in groups], _legacy_segmentation(settings))

    path = _contained_file(folder, PROJECT_TEXT_FILE_NAME)
    if path is None:
        return "Missing project text."
    return book_from_project_text_json_dict(_read_json(path), _legacy_segmentation(settings))


def _legacy_segmentation(settings: dict) -> BookSegmentationSettings:
    language = settings.get("applied_language_code")
    max_words = settings.get("applied_max_words")
    strategy = SegmentationStrategy.from_id(settings.get("applied_strategy", ""))
    return BookSegmentationSettings(
        language_code=language if isinstance(language, str) else "",
        max_words_per_segment=max_words if type(max_words) is int else 0,
        strategy=strategy or BookSegmentationSettings().strategy,
        dialog_segmentation=settings.get("applied_dialog_segmentation") is True,
    )


def _load_speakers(folder: Path, book: Book, warnings: list[str]) -> tuple[Roster | None, AttributionsFile, dict[str, int]]:
    project_dir = str(folder)
    files = [
        (SpeakerAssignFileUtil.roster_path(project_dir), SpeakerAssignFileUtil.load_roster_file),
        (SpeakerAssignFileUtil.attributions_path(project_dir), SpeakerAssignFileUtil.load_attributions_file),
        (SpeakerAssignFileUtil.voices_path(project_dir), SpeakerAssignFileUtil.load_voices_file),
    ]
    loaded = [
        loader(project_dir) if _contained_file(folder, Path(path).relative_to(folder).as_posix()) is not None else None
        for path, loader in files
    ]
    roster_file, attributions_file, voices_file = loaded
    if any(isinstance(item, str) for item in loaded):
        warnings.append("Some console speaker assignment data could not be read.")
    if not isinstance(roster_file, RosterFile):
        return None, AttributionsFile(), {}
    attributions = AttributionsFile()
    if isinstance(attributions_file, AttributionsFile):
        # Settings provenance can contain custom service URLs or parameters.
        attributions = AttributionsFile(tuple(
            replace(saved, llm={}) for saved in attributions_file.valid_ranges(book)
        ))
    return roster_file.roster, attributions, voices_file.voices if isinstance(voices_file, VoiceMapFile) else {}


def _installed_model(tts_id: str, choices: Iterable[tuple[str, str]]) -> str:
    model = TtsModelType.get_by_id(tts_id)
    if model.id == "none":
        return ""
    label = model.value.ui["proper_name"]
    return next((value for choice_label, value in choices if choice_label == label), "")


def _batch_size(settings: dict, tts_id: str) -> int:
    binding = REGISTRY.orchestration_binding(tts_id)
    if binding is None:
        return 1
    value = settings.get("model_settings")
    for key in ("models", tts_id, binding.section, binding.name):
        value = value.get(key) if isinstance(value, dict) else None
    return min(value, PROJECT_BATCH_SIZE_MAX) if type(value) is int and value > 0 else 1


def _stage_voice(
        folder: Path, staged: Path, voice: Voice, tts_id: str, copied: dict[Path, str], warnings: list[str],
) -> RenderVoice:
    reference = None
    transcript = voice.transcript
    if voice.crop is not None:
        reference = _contained_file(folder, f"{PROJECT_VOICE_SUBDIR}/{VOICE_CROP_SUBDIR}/{voice.crop.file_name}")
        if reference is not None:
            transcript = voice.crop.transcript
    if reference is None and voice.file_name:
        for candidate in (f"{PROJECT_VOICE_SUBDIR}/{voice.file_name}", voice.file_name):
            reference = _contained_file(folder, candidate)
            if reference is not None and reference.suffix.lower() in _AUDIO_SUFFIXES:
                break
            reference = None
    instructions = voice.instructions.get(tts_id, "")
    if voice.presets.get(tts_id):
        warnings.append("Console preset voices require a new voice selection in the browser.")
    if reference is None:
        if voice.file_name or voice.crop is not None:
            warnings.append("Some console reference clips are unavailable inside the project folder.")
        return RenderVoice(instructions=instructions)
    if reference not in copied:
        destination = staged / f"console-voice-{len(copied)}{reference.suffix.lower()}"
        shutil.copyfile(reference, destination)
        copied[reference] = str(destination)
    return RenderVoice(copied[reference], transcript, instructions)


def _contained_file(folder: Path, relative: str) -> Path | None:
    if not relative or path_norm.is_absolute_any_grammar(relative) or path_norm.has_drive_or_unc(relative):
        return None
    parts = path_norm.split_relative(relative)
    if not parts or any(part == ".." or ":" in part or "\x00" in part for part in parts):
        return None
    path = folder.joinpath(*parts)
    try:
        resolved = path.resolve(strict=True)
        if resolved == path and resolved.is_relative_to(folder) and resolved.is_file():
            return resolved
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def _read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)