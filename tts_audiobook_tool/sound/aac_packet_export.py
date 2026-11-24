"""Assemble an AAC audiobook with a continuous, sample-counted media track."""

from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types import NormalizationSpecs

from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.sound.aac_checkpoint import AacTakeProfile, FRAME_SAMPLES
from tts_audiobook_tool.sound.audio_writers import (
    AacWriter, AudioWriteError, Chapter, _DiscardableFile, _delete_silently,
    aac_bit_rate, set_mp4_chapters, staging_path_for,
)


TIME_BASE = Fraction(1, APP_SAMPLE_RATE)
# Limit the lookahead when no pause can host an independent AAC join.
MAX_PENDING_SAMPLES = 30 * APP_SAMPLE_RATE


class PacketExportUnavailable(Exception):
    """This export requires one continuous waveform encode."""


def _audio_stream(container):
    stream = container.streams.best("audio")
    if (stream is not None and stream.time_base == TIME_BASE
            and stream.codec_context.name == "aac" and stream.codec_context.profile == "LC"
            and stream.codec_context.sample_rate == APP_SAMPLE_RATE
            and stream.codec_context.layout.name == "mono" and stream.codec_context.extradata):
        return stream
    return None


def _take_profile(
        container, stream, bitrate: int, loudness: NormalizationSpecs | None, volume_db: float,
) -> AacTakeProfile | None:
    profile = AacTakeProfile.from_tags(container.metadata)
    if (stream is not None and profile is not None and stream.start_time == 0
            and stream.duration == profile.samples and profile.bitrate == bitrate
            and profile.matches_normalization(loudness, volume_db)):
        return profile
    return None


def is_packet_copy_candidate(
        path: str, bitrate: str, loudness: NormalizationSpecs | None, volume_db: float,
) -> bool:
    import av

    try:
        with av.open(path) as container:
            return _take_profile(container, _audio_stream(container), aac_bit_rate(bitrate),
                                 loudness, volume_db) is not None
    except (OSError, av.error.FFmpegError):
        return False


@dataclass(frozen=True)
class PacketTake:
    profile: AacTakeProfile
    config: bytes
    packets: tuple[bytes, ...]

    def audio(self) -> np.ndarray:
        """Decode the leveled waveform if this take needs a local rebuild."""
        import av

        decoder = av.CodecContext.create("aac", "r")
        decoder.extradata = self.config
        try:
            frames = [
                frame.to_ndarray()[0] for payload in self.packets for frame in decoder.decode(av.Packet(payload))
            ]
        except av.error.FFmpegError as error:
            raise PacketExportUnavailable(f"AAC take could not be decoded: {error}") from error
        if not frames:
            raise PacketExportUnavailable("AAC take contains no decodable frames")
        data = np.concatenate(frames)[FRAME_SAMPLES:FRAME_SAMPLES + self.profile.samples]
        if len(data) < self.profile.samples:
            raise PacketExportUnavailable("AAC take could not be decoded to its declared duration")
        return data


def _read_take(
        path: str, config: bytes, bitrate: int, loudness: NormalizationSpecs | None, volume_db: float,
) -> PacketTake | None:
    import av

    try:
        with av.open(path) as container:
            stream = _audio_stream(container)
            profile = _take_profile(container, stream, bitrate, loudness, volume_db)
            if profile is None or stream.codec_context.extradata != config:
                return None
            packets = []
            pts = -FRAME_SAMPLES
            for packet in container.demux(stream):
                if not packet.size:
                    continue
                if (pts >= profile.samples or packet.is_corrupt or packet.pts != pts or packet.dts != pts
                        or packet.duration != min(FRAME_SAMPLES, profile.samples - pts)
                        or packet.time_base != TIME_BASE):
                    return None
                packets.append(bytes(packet))
                pts += FRAME_SAMPLES
            if pts < profile.samples:
                return None
            return PacketTake(profile, config, tuple(packets))
    except (OSError, av.error.FFmpegError):
        return None


def _encoder(bitrate: int):
    import av

    encoder = av.CodecContext.create("aac", "w")
    encoder.sample_rate = APP_SAMPLE_RATE
    encoder.layout = "mono"
    encoder.format = "fltp"
    encoder.time_base = TIME_BASE
    encoder.bit_rate = bitrate
    encoder.flags |= av.codec.context.Flags.global_header
    encoder.open()
    return encoder


def _encode(encoder, audio: np.ndarray):
    import av

    planar = AacWriter.to_planar(audio.reshape(-1, 1))
    for offset in range(0, planar.shape[1], AacWriter.MAX_FRAME_SAMPLES):
        frame = av.AudioFrame.from_ndarray(
            planar[:, offset:offset + AacWriter.MAX_FRAME_SAMPLES], format="fltp", layout="mono",
        )
        frame.sample_rate = APP_SAMPLE_RATE
        frame.time_base = TIME_BASE
        frame.pts = offset
        yield from encoder.encode(frame)
    yield from encoder.encode(None)


@dataclass
class _Part:
    audio: PacketTake | np.ndarray
    samples: int

    @property
    def content_samples(self) -> int:
        return self.audio.profile.samples if isinstance(self.audio, PacketTake) else len(self.audio)


class AacPacketExport:
    """Copy compatible packets and encode only the parts requiring waveform work.

    Each independent run begins at a 1024-sample boundary. A copied run
    retains its AAC priming packet, using the preceding silence for that
    packet and for one intervening all-zero packet. Partial final packets
    are padded inside the silence. To keep rounding from accumulating, each
    boundary is rounded against the original cumulative sample position.
    The final duration uses the original sample count.
    """

    def __init__(
            self, dest_path: str, template_path: str, bitrate: str, metadata: dict[str, str] | None,
    ) -> None:
        import av

        self._dest_path = dest_path
        self._staging_path = staging_path_for(dest_path)
        self._bitrate = aac_bit_rate(bitrate)
        self._parts: list[_Part] = []
        self._samples = 0
        self._group_start = 0
        self._planned_group_start = 0
        self._content_end = 0
        self._adjustments: dict[int, int] = {}
        self._open = False

        encoder = _encoder(self._bitrate)
        self._config = encoder.extradata
        if encoder.frame_size != FRAME_SAMPLES:
            raise PacketExportUnavailable("Unexpected AAC frame size")
        silence = list(_encode(encoder, np.zeros(FRAME_SAMPLES * 2, dtype=np.float32)))
        if len(silence) != 3 or bytes(silence[1]) != bytes(silence[2]):
            raise PacketExportUnavailable("AAC encoder did not produce reusable silence")
        self._silence_packet = bytes(silence[2])

        try:
            with av.open(template_path) as source:
                stream = _audio_stream(source)
                if stream is None or stream.codec_context.extradata != self._config:
                    raise PacketExportUnavailable("AAC take configuration is incompatible")
                self._file = _DiscardableFile(self._staging_path)
                self._open = True
                try:
                    self._container = av.open(
                        self._file, mode="w", format="ipod",
                        options={"movflags": "+faststart", "movie_timescale": str(APP_SAMPLE_RATE)},
                    )
                    self._stream = self._container.add_stream_from_template(stream)
                    self._stream.time_base = TIME_BASE
                    self._container.metadata.update(metadata or {})
                except BaseException:
                    self.abort()
                    raise
        except (OSError, av.error.FFmpegError) as error:
            raise PacketExportUnavailable(str(error)) from error

    def read_take(self, path: str, loudness: NormalizationSpecs | None, volume_db: float) -> PacketTake | None:
        return _read_take(path, self._config, self._bitrate, loudness, volume_db)

    def append(self, index: int, audio: PacketTake | np.ndarray, samples: int, *, final: bool) -> None:
        """Append a rendered part and commit at a safe silence boundary."""
        part = _Part(audio, samples)
        if part.content_samples > samples:
            raise ValueError("Rendered audio exceeds its duration")
        if part.content_samples:
            self._content_end = self._samples - self._planned_group_start + part.content_samples
        self._parts.append(part)
        self._samples += samples
        boundary = self._samples
        if final:
            if self._group_start + self._content_end > boundary:
                raise PacketExportUnavailable("Final silence cannot accommodate the last AAC join")
        else:
            boundary = ((self._samples + FRAME_SAMPLES // 2) // FRAME_SAMPLES) * FRAME_SAMPLES
            padded_end = ((self._content_end + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES
            if boundary < self._group_start + padded_end + 2 * FRAME_SAMPLES:
                if self._samples - self._planned_group_start >= MAX_PENDING_SAMPLES:
                    raise PacketExportUnavailable("No usable silence for a localized AAC join")
                return

        adjustment = boundary - self._samples
        pause_adjustment = adjustment - (self._group_start - self._planned_group_start)
        if part.samples + pause_adjustment < part.content_samples:
            raise PacketExportUnavailable("An AAC join would exceed the available trailing silence")
        self._write_group(boundary, final=final)
        self._adjustments[index] = pause_adjustment
        self._group_start = self._samples + adjustment
        self._planned_group_start = self._samples
        self._parts.clear()
        self._content_end = 0

    def _group_packets(self):
        if len(self._parts) == 1 and isinstance(self._parts[0].audio, PacketTake):
            yield from self._parts[0].audio.packets
            return

        data = np.zeros(max(self._content_end, 1), dtype=np.float32)
        offset = 0
        for part in self._parts:
            audio = part.audio.audio() if isinstance(part.audio, PacketTake) else part.audio
            data[offset:offset + len(audio)] = audio
            offset += part.samples
        encoder = _encoder(self._bitrate)
        yield from (bytes(packet) for packet in _encode(encoder, data))

    def _write_group(self, boundary: int, *, final: bool) -> None:
        import av

        stop = boundary if final else boundary - FRAME_SAMPLES
        pts = self._group_start - FRAME_SAMPLES
        try:
            for payload in self._group_packets():
                if pts >= stop:
                    raise PacketExportUnavailable("An AAC packet would extend past the join")
                self._mux(payload, pts, min(FRAME_SAMPLES, stop - pts))
                pts += FRAME_SAMPLES
            while pts < stop:
                self._mux(self._silence_packet, pts, min(FRAME_SAMPLES, stop - pts))
                pts += FRAME_SAMPLES
        except av.error.FFmpegError as error:
            raise AudioWriteError(f"Couldn't assemble audiobook: {error}") from error

    def _mux(self, payload: bytes, pts: int, duration: int) -> None:
        import av

        packet = av.Packet(payload)
        packet.pts = packet.dts = pts
        packet.time_base = TIME_BASE
        packet.duration = duration
        packet.stream = self._stream
        self._container.mux(packet)

    def sample_adjustments(self) -> dict[int, int]:
        return self._adjustments

    def close(self, chapters: Sequence[Chapter]) -> None:
        try:
            if chapters:
                set_mp4_chapters(self._container, chapters, APP_SAMPLE_RATE)
            self._container.close()
            self._file.close()
            os.replace(self._staging_path, self._dest_path)
        except Exception as error:
            self.abort()
            raise AudioWriteError(f"Couldn't finalize audiobook: {error}") from error
        self._open = False

    def abort(self) -> None:
        if self._open:
            self._open = False
            try:
                self._file.discard()
                container = getattr(self, "_container", None)
                if container is not None:
                    container.close()
            except Exception:
                pass
            _delete_silently(self._staging_path)
