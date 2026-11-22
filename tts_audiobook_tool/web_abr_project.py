"""Build a new browser project from an audiobook's embedded settings."""

from dataclasses import dataclass
from pathlib import Path

from tts_audiobook_tool.app_types import Book, BookSegmentationSettings
from tts_audiobook_tool.app_types.app_metadata import AppMetadata
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.web_backend import BookSession, new_book_session
from tts_audiobook_tool.web_model_options import model_options_for, validate_model_options
from tts_audiobook_tool.web_model_setup import model_profiles
from tts_audiobook_tool.web_projects import WebProjectError
from tts_audiobook_tool.web_render import RenderVoice


@dataclass(frozen=True)
class AbrProject:
    session: BookSession
    narrator: RenderVoice
    model_id: str
    model_options: dict[str, dict[str, object]]
    batch_size: int
    strictness: str
    max_retries: int
    auto_concat: bool


def read_abr_project(path: str | None, upload_root: str | Path) -> AbrProject:
    if not path:
        raise WebProjectError("Upload an ABR M4B, M4A or FLAC file.")
    try:
        source = Path(path).resolve(strict=True)
        if (source.suffix.lower() not in {".flac", ".m4a", ".m4b"} or not source.is_file()
                or not source.is_relative_to(Path(upload_root).resolve())):
            raise OSError
    except (OSError, RuntimeError):
        raise WebProjectError("Upload an ABR M4B, M4A or FLAC file.") from None
    metadata = AppMetadata.load_from_file(str(source))
    if metadata is None or not metadata.project_snapshot:
        raise WebProjectError("This audiobook has no readable ABR project settings.")
    try:
        project = ProjectTransferUtil.validate_abr_snapshot(metadata.project_snapshot)
    except (TypeError, ValueError, RecursionError):
        raise WebProjectError("Could not read the ABR project settings.") from None
    try:
        profiles = {profile.catalog_id: profile.model_id for profile in model_profiles()}
        model_id = profiles.get(project.tts_model_type, "")
        model_options = {}
        for catalog_id, environment in profiles.items():
            overrides = {}
            for option in model_options_for(catalog_id):
                value = project.get_model_setting(catalog_id, option.key)
                if option.key == "lora_target" or value is None:
                    continue
                try:
                    validated = validate_model_options(catalog_id, {option.key: value})
                except ValueError:
                    continue
                if option.key in validated:
                    overrides[option.key] = validated[option.key]
            if overrides:
                model_options[environment] = overrides
        native_voice = project.voice_references[0] if project.voice_references else None
        narrator = RenderVoice(
            name=native_voice.name, pan=native_voice.pan, volume=native_voice.volume,
            room_reverb=native_voice.room_reverb.id, model_instructions=tuple(native_voice.instructions.items()),
            presets=tuple(native_voice.presets.items()),
        ) if native_voice else RenderVoice()
        session = new_book_session(
            Book([], title=metadata.title or source.stem, segmentation_settings=BookSegmentationSettings(
                project.language_code, project.max_words, project.segmentation_strategy, project.dialog_segmentation,
            )),
            project_language=project.language_code,
            word_substitutions={key: value for key, value in project.word_substitutions.items() if key and value},
        )
        return AbrProject(
            session, narrator, model_id, model_options,
            ProjectVoiceUtil.get_batch_size(project), project.strictness.id, project.max_retries, project.gen_auto_concat,
        )
    finally:
        project.kill()
        if project.sound_segments.observer.is_alive():
            project.sound_segments.observer.join()
