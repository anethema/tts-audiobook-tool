"""Verify the end of a saved take before reusing it for review or resume."""

from os import SEEK_END
from pathlib import Path


def has_complete_audio(path: Path) -> bool:
    if path.suffix.lower() == ".m4a":
        import av

        try:
            with av.open(str(path)) as container:
                stream = container.streams.best("audio")
                if stream is None or not stream.duration:
                    return False
                end = (stream.start_time or 0) + stream.duration
                container.seek(end - 1, stream=stream)
                for packet in container.demux(stream):
                    if packet.pts is None or not packet.duration:
                        continue
                    frames = packet.decode()
                    if packet.pts + packet.duration >= end and frames:
                        return True
        except (FileNotFoundError, av.error.InvalidDataError, av.error.EOFError):
            pass
        return False

    import soundfile

    try:
        # Opening the path ourselves lets permission and I/O errors propagate
        # separately from decoder errors indicating damaged audio.
        with path.open("rb") as source, soundfile.SoundFile(source) as audio:
            audio.seek(-1, SEEK_END)
            return bool(audio.read(1).size)
    except (FileNotFoundError, ValueError, soundfile.LibsndfileError):
        return False
