"""The AAC representation of a generated take."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tts_audiobook_tool.app_types import NormalizationSpecs, NormalizationType, Sound
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.sound.audio_decoder import AudioDecodeError, AudioDecoder
from tts_audiobook_tool.sound.audio_writers import AacWriter, aac_bit_rate


PROFILE_TAG = "tts_audiobook_take_v1"
FRAME_SAMPLES = 1024
PSEUDO_SILENCE_MAX_SECONDS = 1.0
PSEUDO_SILENCE_THRESHOLD_DB = -20.0
CHECKPOINT_NORMALIZATION = NormalizationType.DEFAULT.value
# Change the analysis version when altering the leveling or silence algorithms.
ANALYSIS_SIGNATURE = (
    f"1:{CHECKPOINT_NORMALIZATION.i}:{CHECKPOINT_NORMALIZATION.tp}:"
    f"{PSEUDO_SILENCE_MAX_SECONDS}:{PSEUDO_SILENCE_THRESHOLD_DB}"
)


@dataclass(frozen=True)
class AacTakeProfile:
    samples: int
    bitrate: int
    leading_silence: float
    trailing_silence: float
    # A constant gain is reversible under the same codec loss as a raw take.
    # A clip requiring a limiter is stored at the original level.
    gain_db: float | None
    measured_lufs: float | None
    gain_clamped: bool
    analysis_signature: str = ""

    @classmethod
    def from_tags(cls, tags: dict[str, str]) -> AacTakeProfile | None:
        if PROFILE_TAG not in tags:
            return None
        try:
            profile = cls(**json.loads(tags[PROFILE_TAG]))
            duration = profile.samples / APP_SAMPLE_RATE
            if (type(profile.samples) is int and profile.samples > 0
                    and type(profile.bitrate) is int and profile.bitrate > 0
                    and isinstance(profile.analysis_signature, str)
                    and 0 <= profile.leading_silence <= duration
                    and 0 <= profile.trailing_silence <= duration
                    and (profile.gain_db is None or -30 <= profile.gain_db <= 30)
                    and (profile.measured_lufs is None or math.isfinite(profile.measured_lufs))
                    and type(profile.gain_clamped) is bool):
                return profile
        except (ValueError, OverflowError, TypeError):
            pass
        return None

    def matches_normalization(self, loudness: NormalizationSpecs | None, volume_db: float) -> bool:
        if self.analysis_signature != ANALYSIS_SIGNATURE:
            return False
        if loudness is None:
            return self.gain_db in (None, 0.0)
        return loudness == CHECKPOINT_NORMALIZATION and volume_db == 0 and self.gain_db is not None


def save_take(sound: Sound, path: str, bitrate: str) -> None:
    # Local imports keep the ordinary FLAC and decoder paths independent
    # of resampling and loudness analysis.
    from tts_audiobook_tool.sound.loudness_leveling import LoudnessLeveling
    from tts_audiobook_tool.sound.silence_util import SilenceUtil
    from tts_audiobook_tool.sound.sound_util import SoundUtil

    sound = SoundUtil.resample_if_necessary(sound, APP_SAMPLE_RATE)
    if not sound.data.size:
        raise ValueError("Cannot encode an empty take")
    leveled = LoudnessLeveling.level(sound, CHECKPOINT_NORMALIZATION)
    gain_db = None
    if leveled.limiter_reduction_db == 0 and np.max(np.abs(leveled.sound.data)) <= 1:
        data = leveled.sound.data
        gain_db = leveled.gain_db
    else:
        data = sound.data

    start_silence = SilenceUtil.get_start_silence(
        sound,
        max_seconds=PSEUDO_SILENCE_MAX_SECONDS,
        threshold_db_relative_to_peak=PSEUDO_SILENCE_THRESHOLD_DB,
    )
    end_silence = SilenceUtil.get_end_silence(
        sound,
        max_seconds=PSEUDO_SILENCE_MAX_SECONDS,
        threshold_db_relative_to_peak=PSEUDO_SILENCE_THRESHOLD_DB,
    )
    profile = AacTakeProfile(
        samples=len(data),
        bitrate=aac_bit_rate(bitrate),
        analysis_signature=ANALYSIS_SIGNATURE,
        leading_silence=start_silence or 0.0,
        trailing_silence=max(0.0, sound.duration - end_silence) if end_silence is not None else 0.0,
        gain_db=gain_db,
        measured_lufs=leveled.measured_lufs,
        gain_clamped=leveled.clamped,
    )
    with AacWriter(path, APP_SAMPLE_RATE, 1, bitrate, {PROFILE_TAG: json.dumps(asdict(profile))},
                   checkpoint=True) as writer:
        writer.write(data)


def read_take(path: str, *, original_level: bool = True) -> tuple[np.ndarray, int]:
    with AudioDecoder(path, channels=1) as decoder:
        profile = AacTakeProfile.from_tags(decoder.metadata)
        if profile is None:
            return decoder.read(), decoder.sample_rate
        if (decoder.sample_rate != APP_SAMPLE_RATE or decoder.source_channels != 1
                or decoder.duration != profile.samples / APP_SAMPLE_RATE):
            raise AudioDecodeError("AAC take duration or format does not match its metadata")
        data = decoder.read(duration=profile.samples / APP_SAMPLE_RATE)
        if len(data) != profile.samples:
            raise AudioDecodeError("AAC take is incomplete")
        if original_level and profile.gain_db:
            data *= 10 ** (-profile.gain_db / 20)
        return data, decoder.sample_rate


def is_take_path(path: str) -> bool:
    return Path(path).suffix.lower() == ".m4a"
