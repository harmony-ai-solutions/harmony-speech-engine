"""Benchmark the pipeline hops that Phase 2 moved onto the artifact bus.

Two measurement modes:

``--mode stub`` (default)
    Measures the *serialization work itself* with representative payloads, so
    it needs no model weights and runs anywhere:

    * ``mel_hop`` — VoiceFixer restorer -> vocoder. Legacy: ``base64(JSON(list))``
      encode + decode of a ``[channels, frames, 128]`` float mel. Artifact: passing
      the ``MelArtifact`` reference (zero encode/decode).
    * ``wav_hop`` — synthesis -> tone converter. Legacy: WAV encode + base64 +
      base64 decode + WAV decode of a one-second waveform. Artifact: reference
      pass-through.

    ``--legacy-serialization`` keeps both representations in the report (it is
    always on in stub mode; the flag exists so the intent is explicit).

``--mode engine``
    Boots a real engine for one family (weights from the HuggingFace cache) and
    runs full requests, recording per-model-step wall time (via executor
    instrumentation), total request wall time, ``tracemalloc`` peak host memory
    and — on CUDA — ``torch.cuda.max_memory_allocated``. It also counts codec
    encode/decode calls to prove the artifact bus touches each payload once at
    the boundary instead of once per hop.

Results are appended to ``docs/benchmarks/artifact-bus.md`` (Markdown) so the
numbers live in the repository.

Usage::

    python scripts/bench_pipeline_hops.py --mode stub --label phase2-artifacts
    python scripts/bench_pipeline_hops.py --mode engine --family voicefixer --iterations 3

The measurement helpers (``measure``, ``summarize``, ``legacy_mel_roundtrip``,
``artifact_passthrough``, ``append_markdown``) are unit-tested with stubs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import json
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/bench_pipeline_hops.py`
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REPORT = REPO_ROOT / "docs" / "benchmarks" / "artifact-bus.md"


# ---------------------------------------------------------------------------
# Measurement plumbing (unit-tested)
# ---------------------------------------------------------------------------


def measure(fn: Callable[[], object], repeat: int = 5) -> list[float]:
    """Run ``fn`` ``repeat`` times, returning elapsed seconds per run.

    Each run is preceded by a ``gc.collect()`` so one iteration's garbage does
    not dominate the next (especially relevant for the multi-MB mel payloads).
    """
    samples: list[float] = []
    for _ in range(max(1, repeat)):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def summarize(samples: Iterable[float]) -> dict[str, float]:
    """Return mean/median/min/max in milliseconds for a sample list."""
    values = [float(value) * 1000.0 for value in samples]
    if not values:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def legacy_mel_encode(mel: np.ndarray) -> str:
    """Reproduce the VoiceFixer restorer's pre-Phase-2 mel serialization."""
    return base64.b64encode(json.dumps(mel.tolist()).encode("utf-8")).decode("utf-8")


def legacy_mel_decode(payload: str) -> np.ndarray:
    """Reproduce the VoiceFixer vocoder's pre-Phase-2 mel deserialization."""
    return np.asarray(json.loads(base64.b64decode(payload).decode("utf-8")), dtype=np.float32)


def legacy_mel_roundtrip(mel: np.ndarray) -> np.ndarray:
    return legacy_mel_decode(legacy_mel_encode(mel))


def legacy_wav_roundtrip(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Reproduce the synthesis -> tone-converter WAV encode/base64/decode hop."""
    import io

    import soundfile as sf

    from harmonyspeech.common import audio_codec

    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=sample_rate, format="wav")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return audio_codec.decode_wav_b64(encoded)[0]


def artifact_passthrough(artifact):
    """The artifact path: hand the same object to the next step (no encode)."""
    return artifact


def append_markdown(path: Path, rows: list[dict]) -> None:
    """Append a Markdown table to ``path`` (creating it with a header first)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["label", "mode", "family", "hop", "n", "mean_ms", "p50_ms", "min_ms", "max_ms", "peak_mib", "notes"]
    lines = []
    if not path.exists():
        lines.append("# Artifact bus — hop benchmarks\n")
        lines.append(
            "Generated by `scripts/bench_pipeline_hops.py`. `stub` rows measure the "
            "serialization work with representative payloads; `engine` rows are real "
            "requests against real weights.\n"
        )
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Stub mode: serialization cost of the two worst legacy hops
# ---------------------------------------------------------------------------


def _mel_payload(frames: int = 100, bins: int = 128, channels: int = 1) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.random((channels, frames, bins), dtype=np.float32)


def _waveform_payload(seconds: float = 1.0, sample_rate: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def run_stub_mode(args) -> list[dict]:
    from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact

    mel = _mel_payload()
    waveform = _waveform_payload()

    mel_artifact = MelArtifact.from_numpy(mel[0], sample_rate=44100, hop_length=441)
    wav_artifact = AudioArtifact.from_numpy(waveform, 24000)

    rows: list[dict] = []
    for hop, legacy_fn, artifact_fn in [
        (
            "voicefixer_mel_hop",
            lambda: legacy_mel_roundtrip(mel),
            lambda: artifact_passthrough(mel_artifact),
        ),
        (
            "synth_to_tone_converter",
            lambda: legacy_wav_roundtrip(waveform, 24000),
            lambda: artifact_passthrough(wav_artifact),
        ),
    ]:
        gc.collect()
        tracemalloc.start()
        legacy_samples = measure(legacy_fn, args.iterations)
        _, legacy_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        legacy = summarize(legacy_samples)
        rows.append(
            {
                "label": args.label,
                "mode": "stub",
                "family": "synthetic",
                "hop": hop,
                "n": legacy["n"],
                "mean_ms": legacy["mean_ms"],
                "p50_ms": legacy["p50_ms"],
                "min_ms": legacy["min_ms"],
                "max_ms": legacy["max_ms"],
                "peak_mib": legacy_peak / (1024 * 1024),
                "notes": "pre-Phase-2 serialized representation (encode+base64+decode)",
            }
        )

        gc.collect()
        tracemalloc.start()
        artifact_samples = measure(artifact_fn, args.iterations)
        _, artifact_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        artifact = summarize(artifact_samples)
        rows.append(
            {
                "label": args.label,
                "mode": "stub",
                "family": "synthetic",
                "hop": hop,
                "n": artifact["n"],
                "mean_ms": artifact["mean_ms"],
                "p50_ms": artifact["p50_ms"],
                "min_ms": artifact["min_ms"],
                "max_ms": artifact["max_ms"],
                "peak_mib": artifact_peak / (1024 * 1024),
                "notes": "artifact bus (reference pass-through, 0 encode/decode)",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Engine mode: real weights, real requests
# ---------------------------------------------------------------------------


def _family_configs(family: str):
    from harmonyspeech.common.config import DeviceConfig, ModelConfig

    device_config = DeviceConfig(device="cpu")
    if family == "voicefixer":
        return [
            ModelConfig(
                name="vf-restorer",
                model="jlmarrugom/voice_fixer",
                model_type="VoiceFixerRestorer",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
            ),
            ModelConfig(
                name="vf-vocoder",
                model="jlmarrugom/voice_fixer",
                model_type="VoiceFixerVocoder",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
            ),
        ]
    if family == "harmonyspeech":
        return [
            ModelConfig(
                name="hs-encoder",
                model="harmony-ai/harmony-speech-v1",
                model_type="HarmonySpeechEncoder",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
            ),
            ModelConfig(
                name="hs-synthesizer",
                model="harmony-ai/harmony-speech-v1",
                model_type="HarmonySpeechSynthesizer",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
            ),
            ModelConfig(
                name="hs-vocoder",
                model="harmony-ai/harmony-speech-v1",
                model_type="HarmonySpeechVocoder",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
            ),
        ]
    if family == "openvoice_v2":
        return [
            ModelConfig(
                name="faster-whisper",
                model="Systran/faster-whisper-tiny",
                model_type="FasterWhisper",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
            ),
            ModelConfig(
                name="melo-en",
                model="myshell-ai/MeloTTS-English-v3",
                model_type="MeloTTSSynthesizer",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
                language="EN",
                voices=["EN-Newest"],
            ),
            ModelConfig(
                name="ov2-tc",
                model="myshell-ai/openvoicev2",
                model_type="OpenVoiceV2ToneConverter",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
            ),
            ModelConfig(
                name="ov2-tce",
                model="myshell-ai/openvoicev2",
                model_type="OpenVoiceV2ToneConverterEncoder",
                max_batch_size=1,
                dtype="float32",
                device_config=device_config,
            ),
        ]
    raise SystemExit(f"unknown family {family!r}")


def _build_request(family: str, request_id: str, reference_b64: str):
    from harmonyspeech.common.inputs import AudioConversionRequestInput, TextToSpeechRequestInput

    if family == "voicefixer":
        return AudioConversionRequestInput(
            request_id=request_id,
            requested_model="voicefixer",
            model="",
            source_audio=reference_b64,
        )
    if family == "harmonyspeech":
        return TextToSpeechRequestInput(
            request_id=request_id,
            requested_model="harmonyspeech",
            model="",
            input_text="Hello from the artifact bus benchmark.",
            mode="voice_cloning",
            input_audio=reference_b64,
        )
    return TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="openvoice_v2",
        model="",
        input_text="Hello from the artifact bus benchmark.",
        mode="voice_cloning",
        language_id="EN",
        voice_id="EN-Newest",
        input_audio=reference_b64,
    )


class CodecCounter:
    """Count codec encode/decode calls without changing their behaviour."""

    def __init__(self):
        self.encode_calls = 0
        self.decode_calls = 0
        self._original = {}

    def __enter__(self):
        from harmonyspeech.common import audio_codec

        self._original = {
            "encode_audio": audio_codec.encode_audio,
            "decode_audio": audio_codec.decode_audio,
        }
        counter = self

        def encode_audio(*args, **kwargs):
            counter.encode_calls += 1
            return counter._original["encode_audio"](*args, **kwargs)

        def decode_audio(*args, **kwargs):
            counter.decode_calls += 1
            return counter._original["decode_audio"](*args, **kwargs)

        audio_codec.encode_audio = encode_audio
        audio_codec.decode_audio = decode_audio
        # Modules that imported the function directly keep their own reference;
        # patch those that participate in the measured path too.
        try:
            from harmonyspeech.common import inputs as inputs_module

            inputs_module.decode_audio = decode_audio
        except Exception:  # pragma: no cover - defensive
            pass
        return self

    def __exit__(self, *exc_info):
        from harmonyspeech.common import audio_codec

        audio_codec.encode_audio = self._original["encode_audio"]
        audio_codec.decode_audio = self._original["decode_audio"]
        try:
            from harmonyspeech.common import inputs as inputs_module

            inputs_module.decode_audio = self._original["decode_audio"]
        except Exception:  # pragma: no cover - defensive
            pass
        return False


def _instrument_executors(engine) -> dict[str, list[float]]:
    """Wrap each executor's async entry point and record step wall times."""
    timings: dict[str, list[float]] = {}
    for name, executor in engine.engine.model_executors.items():
        original = executor.execute_model_async
        timings[name] = []

        async def timed(model_requests, _original=original, _name=name):
            start = time.perf_counter()
            result = await _original(model_requests)
            timings[_name].append(time.perf_counter() - start)
            return result

        executor.execute_model_async = timed
    return timings


async def _run_engine_requests(args, reference_b64: str):
    from harmonyspeech.common.config import EngineConfig
    from harmonyspeech.engine.args_tools import AsyncEngineArgs
    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech

    model_configs = _family_configs(args.family)
    engine_config = EngineConfig(model_configs=model_configs)
    engine_args = AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True)
    engine = AsyncHarmonySpeech.from_engine_args_and_config(engine_args, engine_config)

    timings = _instrument_executors(engine)
    request_seconds: list[float] = []
    audio_artifacts = []
    peak_mib = 0.0

    with CodecCounter() as counter:
        for index in range(args.iterations):
            request_id = f"bench-{args.family}-{index}"
            request_data = _build_request(args.family, request_id, reference_b64)
            if index == 0:
                gc.collect()
                tracemalloc.start()
            start = time.perf_counter()
            last = None
            async for output in engine.generate(request_id=request_id, request_data=request_data):
                last = output
            request_seconds.append(time.perf_counter() - start)
            if index == 0:
                _, peak = tracemalloc.get_traced_memory()
                peak_mib = peak / (1024 * 1024)
                tracemalloc.stop()
            if last is not None and last.finish_reason == "error":
                raise SystemExit(f"request failed: {last.error}")
            artifact = getattr(last, "audio_artifact", None)
            if artifact is not None:
                audio_artifacts.append(artifact)

    # Hash/encode outside the counter window so encode_calls reflects only the
    # pipeline (the artifact bus performs zero internal encodes).
    import hashlib

    audio_digests: list[str] = []
    audio_stats: list[dict] = []
    for artifact in audio_artifacts:
        wav_bytes = artifact.to_wav_bytes()
        audio_digests.append(hashlib.sha256(wav_bytes).hexdigest())
        samples = artifact.numpy
        audio_stats.append(
            {
                "duration_s": artifact.duration_s,
                "rms": float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0,
            }
        )

    cuda_mib = ""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_mib = f"{torch.cuda.max_memory_allocated() / (1024 * 1024):.1f} MiB"
    except Exception:  # pragma: no cover - defensive
        pass

    return {
        "request_seconds": request_seconds,
        "step_timings": timings,
        "peak_mib": peak_mib,
        "encode_calls": counter.encode_calls,
        "decode_calls": counter.decode_calls,
        "cuda_mib": cuda_mib,
        "audio_digests": audio_digests,
        "audio_stats": audio_stats,
    }


def _reference_audio_b64(seconds: float = 1.0, sample_rate: int = 24000) -> str:
    """A deterministic tone as base64 WAV (fallback when no sample exists)."""
    import io

    import soundfile as sf

    waveform = _waveform_payload(seconds=seconds, sample_rate=sample_rate)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _resolve_reference_b64(args) -> str:
    """Reference clip for voice-cloning families: real speech when available."""
    path = args.reference_audio
    if path is None:
        candidate = REPO_ROOT / "tests" / "test-data" / "samples" / "wanda4.wav"
        path = candidate if candidate.exists() else None
    if path is not None:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return _reference_audio_b64(seconds=6.0)


def run_engine_mode(args) -> list[dict]:
    reference_b64 = _resolve_reference_b64(args)
    report = asyncio.run(_run_engine_requests(args, reference_b64))

    rows: list[dict] = []
    total = summarize(report["request_seconds"])
    rows.append(
        {
            "label": args.label,
            "mode": "engine",
            "family": args.family,
            "hop": "full_request",
            "n": total["n"],
            "mean_ms": total["mean_ms"],
            "p50_ms": total["p50_ms"],
            "min_ms": total["min_ms"],
            "max_ms": total["max_ms"],
            "peak_mib": report["peak_mib"],
            "notes": (
                f"encode_calls={report['encode_calls']} decode_calls={report['decode_calls']} "
                f"cuda={report['cuda_mib'] or 'n/a'}"
            ),
        }
    )
    for name, samples in report["step_timings"].items():
        stats = summarize(samples)
        rows.append(
            {
                "label": args.label,
                "mode": "engine",
                "family": args.family,
                "hop": name,
                "n": stats["n"],
                "mean_ms": stats["mean_ms"],
                "p50_ms": stats["p50_ms"],
                "min_ms": stats["min_ms"],
                "max_ms": stats["max_ms"],
                "peak_mib": "",
                "notes": "executor step wall time",
            }
        )

    if args.determinism and len(report["audio_stats"]) >= 2:
        first_stats = report["audio_stats"][0]
        second_stats = report["audio_stats"][1]
        same_bytes = report["audio_digests"][0] == report["audio_digests"][1]
        duration_delta = abs(first_stats["duration_s"] - second_stats["duration_s"]) / max(
            first_stats["duration_s"], 1e-9
        )
        rms_delta = abs(first_stats["rms"] - second_stats["rms"]) / max(first_stats["rms"], 1e-9)
        rows.append(
            {
                "label": args.label,
                "mode": "engine",
                "family": args.family,
                "hop": "determinism",
                "n": 2,
                "mean_ms": "",
                "p50_ms": "",
                "min_ms": "",
                "max_ms": "",
                "peak_mib": "",
                "notes": (
                    f"sha256_equal={same_bytes} duration_delta={duration_delta:.4%} rms_delta={rms_delta:.4%}"
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("stub", "engine"), default="stub", help="measurement mode")
    parser.add_argument(
        "--family",
        choices=("voicefixer", "harmonyspeech", "openvoice_v2"),
        default="voicefixer",
        help="engine-mode model family",
    )
    parser.add_argument("--iterations", type=int, default=5, help="number of measured runs")
    parser.add_argument("--label", default="artifact-bus", help="row label (e.g. phase2-artifacts)")
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=None,
        help="reference clip for voice-cloning families (default: tests/test-data/samples/wanda4.wav)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Markdown report to append to")
    parser.add_argument(
        "--legacy-serialization",
        action="store_true",
        help="explicitly include the pre-Phase-2 serialized representation (always on in stub mode)",
    )
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="engine mode: compare two runs' audio bytes/duration/RMS and append a determinism row",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = run_stub_mode(args) if args.mode == "stub" else run_engine_mode(args)
    append_markdown(args.output, rows)
    for row in rows:
        mean = row.get("mean_ms", "")
        mean_text = f"{mean:.3f}" if isinstance(mean, (int, float)) else "-"
        print(
            f"{row['label']} {row['hop']}: mean={mean_text} ms "
            f"peak={row.get('peak_mib', '')} MiB {row.get('notes', '')}"
        )
    print(f"appended {len(rows)} row(s) to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
