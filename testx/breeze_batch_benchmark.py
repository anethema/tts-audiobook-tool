"""Compare warmed Breeze TTS 2 batch sizes using the audiobook generation loop.

Run from the repository root with the Breeze environment:
    ./venv-breeze/bin/python -m testx.breeze_batch_benchmark --text passage.txt
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, replace
from io import StringIO
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Event, Thread
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tts_audiobook_tool.state import State


MIB = 1024**2


def parse_args() -> argparse.Namespace:
    from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_MAX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, type=Path, help="UTF-8 TXT file")
    parser.add_argument("--voice", default="Narrator", help="Name in the browser voice library")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[10, 16, 20, 24, 28, 32])
    parser.add_argument("--limit", type=int, default=0, help="Sample this many groups across the text (0 means all)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", choices=["each", "once"], default="each",
                        help="Warm up each batch size, or warm up only the first size")
    parser.add_argument("--no-validation", action="store_true", help="Disable Whisper audio validation")
    parser.add_argument("--min-free-mib", type=int, default=0,
                        help="End the ascending sweep after a candidate whose sampled VRAM headroom is below this")
    parser.add_argument("--json", type=Path, help="Write a JSON report after each completed candidate")
    args = parser.parse_args()
    if any(not 1 <= size <= PROJECT_BATCH_SIZE_MAX for size in args.batch_sizes):
        parser.error(f"batch sizes must be between 1 and {PROJECT_BATCH_SIZE_MAX}")
    if args.limit < 0 or args.min_free_mib < 0:
        parser.error("group limit and VRAM headroom must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 4294967295")
    if args.json is not None and args.json.resolve() == args.text.resolve():
        parser.error("the text and JSON report must have different paths")
    args.batch_sizes = sorted(set(args.batch_sizes))
    return args


def sample_indices(total: int, limit: int) -> list[int]:
    if not limit or limit >= total:
        return list(range(total))
    # One midpoint per equal-width bucket across the complete source.
    return [(2 * index + 1) * total // (2 * limit) for index in range(limit)]


@dataclass
class VramPeak:
    used_mib: int | None = None
    total_mib: int | None = None

    def sample(self) -> None:
        from tts_audiobook_tool.app_support.app_memory import get_nv_vram

        info = get_nv_vram()
        if info is not None:
            used, total = (value // MIB for value in info)
            self.used_mib = max(self.used_mib or 0, used)
            self.total_mib = total


@contextmanager
def sample_vram():
    """Sample device-wide use, including Whisper and other GPU clients."""
    peak = VramPeak()
    stop = Event()

    def poll() -> None:
        while not stop.wait(0.2):
            peak.sample()

    peak.sample()
    sampler = Thread(target=poll, daemon=True)
    sampler.start()
    try:
        yield peak
    finally:
        stop.set()
        sampler.join()
        peak.sample()


@dataclass
class BatchResult:
    elapsed: float
    synthesis_seconds: float
    audio_seconds: float
    outcomes: Counter[str]
    error: str | None = None
    out_of_memory: bool = False


def generate_batch(state: State, indices: list[int]) -> BatchResult:
    import torch

    from tts_audiobook_tool.app_types.validation_result import SkippedResult, ValidationResult
    from tts_audiobook_tool.generate_util import GenerateUtil
    from tts_audiobook_tool.generation_events import GenerationEvents, GenerationStats
    from tts_audiobook_tool.util import is_oom_error_message

    stats = None

    def on_event(event):
        nonlocal stats
        if isinstance(event, GenerationStats):
            stats = event

    started = perf_counter()
    try:
        with GenerationEvents.using_sink(on_event), redirect_stdout(StringIO()):
            results = GenerateUtil.generate_and_validate_batch(
                state=state,
                indices=indices,
                phrase_groups=state.project.phrase_groups,
                stt_variant=state.prefs.stt_variant,
                stt_config=state.prefs.stt_config,
                force_random_seed=False,
                is_realtime=False,
                voice_selection_index=0,
            )
            torch.cuda.synchronize()
    except (RuntimeError, MemoryError) as error:
        if not (isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)) or is_oom_error_message(str(error))):
            raise
        return BatchResult(perf_counter() - started, 0.0, 0.0, Counter(), str(error), True)
    elapsed = perf_counter() - started

    errors = [GenerateUtil.get_error_result_message(result) for result in results
              if not isinstance(result, ValidationResult)]
    outcomes = Counter(
        "error" if not isinstance(result, ValidationResult)
        else "validation_skipped" if isinstance(result, SkippedResult)
        else "validation_failed" if result.is_fail
        else "validation_passed"
        for result in results
    )
    error = errors[0] if errors else None
    if stats is None and error is None:
        error = "The audiobook generator did not report complete synthesis statistics."
    return BatchResult(
        elapsed,
        stats.generation_seconds if stats else 0.0,
        stats.audio_seconds if stats else 0.0,
        outcomes,
        error,
        any(is_oom_error_message(message) for message in errors),
    )


def measure_candidate(state: State, indices: list[int], size: int, warmup: bool) -> dict:
    import torch

    from tts_audiobook_tool.generate_util import make_multi_voice_rounds, make_single_voice_rounds

    groups = state.project.phrase_groups
    items = [(index, 0) for index in indices]
    if size == 1:
        rounds = make_single_voice_rounds(items, size)
    else:
        rounds = make_multi_voice_rounds(items, groups, dict.fromkeys(indices, 0), size, sort_by_words=True)
    batches = [[index for index, _ in batch.items] for group in rounds for batch in group]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    totals = BatchResult(0.0, 0.0, 0.0, Counter())
    processed = 0
    warmup_seconds = 0.0
    with sample_vram() as peak:
        if warmup:
            print(f"Batch {size} warming up with {size} groups.", flush=True)
            shortest = sorted(indices, key=lambda index: groups[index].num_words)[:size]
            result = generate_batch(state, shortest)
            warmup_seconds = result.elapsed
            totals.error = result.error
            totals.out_of_memory = result.out_of_memory

        if totals.error is None:
            for number, batch in enumerate(batches, start=1):
                result = generate_batch(state, batch)
                totals.elapsed += result.elapsed
                totals.synthesis_seconds += result.synthesis_seconds
                totals.audio_seconds += result.audio_seconds
                totals.outcomes.update(result.outcomes)
                totals.error = result.error
                totals.out_of_memory = result.out_of_memory
                processed += len(batch)
                speed = totals.audio_seconds / totals.elapsed
                print(f"Batch {size} [{number}/{len(batches)}] {processed}/{len(indices)} groups, {speed:.2f}x", flush=True)
                if totals.error:
                    break

    return {
        "batch_size": size,
        "status": "out_of_memory" if totals.out_of_memory else "error" if totals.error else "complete",
        "error": totals.error,
        "groups_attempted": processed,
        "batch_calls": len(batches),
        "warmup_seconds": warmup_seconds,
        "synthesis_seconds": totals.synthesis_seconds,
        "generation_validation_seconds": totals.elapsed,
        "audio_seconds": totals.audio_seconds,
        "synthesis_realtime_multiplier": totals.audio_seconds / totals.synthesis_seconds if totals.synthesis_seconds else None,
        "generation_validation_realtime_multiplier": totals.audio_seconds / totals.elapsed if totals.audio_seconds else None,
        "outcomes": dict(totals.outcomes),
        "pytorch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
        "pytorch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
        "gpu_sampled_peak_used_mib": peak.used_mib,
        "gpu_sampled_min_free_mib": peak.total_mib - peak.used_mib if peak.total_mib is not None else None,
    }


def write_report(path: Path | None, report: dict) -> None:
    if path is None:
        return
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary_path = Path(file.name)
        try:
            json.dump(report, file, indent=2)
            file.write("\n")
            file.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    import torch

    from tts_audiobook_tool.app_support.app_memory import get_nv_vram
    from tts_audiobook_tool.app_types import SegmentationStrategy
    from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_DEFAULT
    from tts_audiobook_tool.l import L
    from tts_audiobook_tool.model_manager import ModelManager
    from tts_audiobook_tool.stt import Stt
    from tts_audiobook_tool.web_render import _detect_model, _project_state
    from tts_audiobook_tool.web_runtime_settings import load_runtime_settings
    from tts_audiobook_tool.web_text import import_book_text
    from tts_audiobook_tool.web_voices import list_voices, load_voice

    L.init("breeze-batch-benchmark", level=logging.WARNING)
    model = _detect_model()
    if model.id != "breeze_tts_2_local" or not torch.cuda.is_available():
        raise SystemExit("Run this benchmark in the Breeze TTS 2 environment with CUDA.")
    matching = [voice for voice in list_voices() if voice.name.casefold() == args.voice.casefold()]
    if len(matching) != 1:
        raise SystemExit(f"Expected exactly one saved voice named {args.voice!r}; found {len(matching)}.")
    voice = load_voice(matching[0].voice_id).voice
    session = import_book_text(
        "plain_text", None, str(args.text), None, "en", MAX_WORDS_PER_SEGMENT_DEFAULT,
        SegmentationStrategy.SENTENCE_PLUS.id, True,
    )
    total = len(session.book.phrase_groups)
    indices = sample_indices(total, args.limit)
    if args.batch_sizes[-1] > len(indices):
        raise SystemExit(f"The largest batch exceeds the {len(indices)} selected text groups.")

    device_baseline = get_nv_vram()
    report = {
        "text": str(args.text),
        "voice": voice.name,
        "model": model.id,
        "gpu": torch.cuda.get_device_name(),
        "gpu_baseline_used_mib": device_baseline[0] // MIB if device_baseline else None,
        "gpu_total_mib": device_baseline[1] // MIB if device_baseline else None,
        "source_groups": total,
        "selected_groups": [index + 1 for index in indices],
        "seed": args.seed,
        "warmup_policy": args.warmup,
        "batch_sizes": args.batch_sizes,
        "results": [],
    }
    runtime = replace(load_runtime_settings(), force_cpu=False)
    print(f"Breeze TTS 2 on {report['gpu']}, {voice.name}, {len(indices)}/{total} text groups.", flush=True)
    with TemporaryDirectory(prefix="breeze-batch-benchmark-") as project_dir:
        with _project_state(
            Path(project_dir), "en", [voice], session.book,
            model=model, model_options={"seed": args.seed}, validate_audio=not args.no_validation,
            runtime_settings=runtime, generation_indices=indices,
        ) as state:
            skip_reason = Stt.should_skip(state)
            if skip_reason and not args.no_validation:
                raise SystemExit(f"Whisper validation is unavailable: {skip_reason}")
            report["whisper_model"] = state.prefs.stt_variant.id
            report["whisper_device_setting"] = state.prefs.stt_config.id
            write_report(args.json, report)
            warmup = ModelManager.warm_up_models(state)
            if warmup.should_stop:
                raise SystemExit(warmup.error or "Model warmup interrupted.")
            for offset, size in enumerate(args.batch_sizes):
                result = measure_candidate(state, indices, size, warmup=args.warmup == "each" or offset == 0)
                report["results"].append(result)
                write_report(args.json, report)
                if result["status"] != "complete":
                    print(f"Batch {size} {result['status']}: {result['error']}", flush=True)
                    break
                print(
                    f"Batch {size} complete: {result['synthesis_realtime_multiplier']:.2f}x synthesis, "
                    f"{result['generation_validation_realtime_multiplier']:.2f}x including validation, "
                    f"GPU peak {result['gpu_sampled_peak_used_mib']} MiB.", flush=True,
                )
                headroom = result["gpu_sampled_min_free_mib"]
                if headroom is not None and headroom < args.min_free_mib:
                    print(f"Stopping with {headroom} MiB sampled VRAM headroom.", flush=True)
                    break


if __name__ == "__main__":
    main()
