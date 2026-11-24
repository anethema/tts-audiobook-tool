from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest

from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import HighShelfEq, NormalizationType, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.concat_util import ConcatRenderResult, ConcatUtil
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.aac_checkpoint import AacTakeProfile, FRAME_SAMPLES, read_take
from tts_audiobook_tool.sound.aac_packet_export import AacPacketExport
from tts_audiobook_tool.sound.audio_writers import Chapter, staging_path_for
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil


pytestmark = pytest.mark.no_ffmpeg_exe
SR = APP_SAMPLE_RATE
PAUSES = ReasonPauseTypes.NORMAL.value


def _take(directory: Path, index: int, samples: int = 9001) -> tuple[str, np.ndarray]:
    path = str(directory / f"take-{index} ü.m4a")
    x = np.asarray(0.18 * np.cos((index + 15) * np.arange(samples) / 127), dtype=np.float32)
    assert SoundFileUtil.save_aac(Sound(x, SR), path) == ""
    return path, x


def _packet_payloads(path: str) -> dict[int, bytes]:
    with av.open(path) as file:
        return {packet.pts: bytes(packet) for packet in file.demux(audio=0) if packet.size}


def _export(path: Path, pairs, **kwargs):
    result = ConcatUtil.concatenate_sound_segments(
        str(path), pairs, False, HighShelfEq.DISABLED, kwargs.pop("reason_pauses", PAUSES), False,
        loudness=NormalizationType.DEFAULT.value, **kwargs,
    )
    assert isinstance(result, ConcatRenderResult), result
    return result


def test_saved_take_has_original_level_and_exact_sample_count(tmp_path: Path):
    path, input_audio = _take(tmp_path, 0, 1025)
    with av.open(path) as file:
        profile = AacTakeProfile.from_tags(file.metadata)
        assert profile.samples == len(input_audio) == file.streams.audio[0].duration
        assert profile.gain_db == -3.0
    original = SoundFileUtil.load(path)
    leveled, rate = read_take(path, original_level=False)
    assert original.sr == rate == SR
    assert len(original.data) == len(leveled) == len(input_audio)
    np.testing.assert_allclose(original.data * 10 ** (-3 / 20), leveled, atol=1e-6)


def test_packet_copy_preserves_takes_and_updates_actual_timing(tmp_path: Path):
    takes = [_take(tmp_path, i, 8300 + 123 * i)[0] for i in range(12)]
    pairs = [(Phrase(f"Line {i}.", Reason.PHRASE), path) for i, path in enumerate(takes)]
    output = tmp_path / "book.m4b"
    chapter_index = 5
    result = _export(
        output, pairs, container_metadata={"title": "Book"},
        make_chapters=lambda durations: [
            Chapter(0, sum(durations[:chapter_index]), "Start"),
            Chapter(sum(durations[:chapter_index]), sum(durations), "Next"),
        ],
    )
    output_packets = _packet_payloads(str(output))
    planned_samples = output_samples = 0
    for i, (path, duration) in enumerate(zip(takes, result.durations)):
        assert abs(output_samples - planned_samples) <= FRAME_SAMPLES // 2
        for pts, payload in _packet_payloads(path).items():
            assert output_packets[output_samples + pts] == payload
        planned_samples += 8300 + 123 * i + round(PAUSES.get_pause_for(Reason.PHRASE) * SR)
        output_samples += round(duration * SR)

    with av.open(str(output)) as file:
        assert file.streams.audio[0].duration == output_samples == planned_samples
        assert file.metadata["title"] == "Book"
        chapters = file.chapters()
        assert chapters[1]["metadata"]["title"] == "Next"
        assert float(chapters[1]["start"] * chapters[1]["time_base"]) == pytest.approx(
            sum(result.durations[:chapter_index]), abs=1 / SR,
        )
        decoded = np.concatenate([frame.to_ndarray()[0] for frame in file.decode(audio=0)])

    start = 0
    for path, duration in zip(takes, result.durations):
        expected, _ = read_take(path, original_level=False)
        # The guard and retained priming packet reproduce the isolated take's
        # onset and overlap, including the first transform after each join.
        np.testing.assert_allclose(decoded[start:start + len(expected)], expected, atol=0.01)
        start += round(duration * SR)
    assert not Path(staging_path_for(str(output))).exists()


def test_short_pause_rebuilds_only_the_adjacent_takes(tmp_path: Path):
    samples = 9001
    paths = [_take(tmp_path, i, samples)[0] for i in range(4)]
    pairs = [
        (Phrase(f"Line {i}.", Reason.WORD if i == 1 else Reason.PHRASE), path)
        for i, path in enumerate(paths)
    ]
    pauses = replace(PAUSES, pauses={**PAUSES.pauses, Reason.WORD: 0.001})
    output = tmp_path / "short.m4b"
    before = _export(output, pairs, reason_pauses=pauses)
    packets = _packet_payloads(str(output))
    starts = [round(sum(before.durations[:i]) * SR) for i in range(len(pairs))]
    assert any(packets.get(starts[1] + pts) != payload for pts, payload in _packet_payloads(paths[1]).items())
    assert starts[2] == starts[1] + samples + 48
    for i in (0, 3):
        assert all(packets[starts[i] + pts] == payload for pts, payload in _packet_payloads(paths[i]).items())

    _take(tmp_path, 2, samples + 51)
    after = _export(output, pairs, reason_pauses=pauses)
    edited_packets = _packet_payloads(str(output))
    assert round((sum(after.durations) - sum(before.durations)) * SR) == 51
    for i in (0, 3):
        start = round(sum(after.durations[:i]) * SR)
        assert all(edited_packets[start + pts] == payload for pts, payload in _packet_payloads(paths[i]).items())


def test_cancelled_packet_export_preserves_prior_output(tmp_path: Path, monkeypatch):
    pairs = [(Phrase(f"Line {i}.", Reason.PHRASE), _take(tmp_path, i)[0]) for i in range(4)]
    output = tmp_path / "book.m4b"
    output.write_bytes(b"existing audiobook")
    read = AacPacketExport.read_take

    def read_then_cancel(self, path, loudness, volume_db):
        take = read(self, path, loudness, volume_db)
        if path == pairs[2][1]:
            Interrupts()._flag = True
        return take

    monkeypatch.setattr(AacPacketExport, "read_take", read_then_cancel)
    result = ConcatUtil.concatenate_sound_segments(
        str(output), pairs, False, HighShelfEq.DISABLED, PAUSES, False,
        loudness=NormalizationType.DEFAULT.value,
    )
    assert result == "Interrupted by user"
    assert output.read_bytes() == b"existing audiobook"
    assert not Path(staging_path_for(str(output))).exists()
