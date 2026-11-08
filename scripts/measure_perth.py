"""Measure the Perth watermark cost on the real Chatterbox watermarker.

Owner request 2026-11-08: how much does ``apply_watermark`` cost, standalone
across output durations, and what share of a Chatterbox request's wall time is
it under load? This entry point covers the standalone half (and can time a real
returned WAV with ``--audio``); the request-share half is derived by combining
these numbers with the per-request wall times the throughput harness records.

Usage::

    python scripts/measure_perth.py --model C:\\models\\chatterbox
    python scripts/measure_perth.py --audio out/perth/sample.wav

Engine code is untouched; this only loads the shipped model and times its
watermarker.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _load_watermarker(model_path: str):
    from harmonyspeech.common.config import DeviceConfig, ModelConfig
    from harmonyspeech.modeling.loader import get_model

    cfg = ModelConfig(
        name="chatterbox-perth",
        model=model_path,
        model_type="ChatterboxTTS",
        max_batch_size=1,
        device_config=DeviceConfig("cuda"),
        dtype="float16",
    )
    model = get_model(cfg, cfg.device_config)
    return model, getattr(model, "watermarker", None), int(getattr(model, "sr", 24000))


def _time_watermark(watermarker, wav: np.ndarray, sample_rate: int, reps: int) -> dict:
    device = None
    net = getattr(watermarker, "perth_net", None)
    if isinstance(net, torch.nn.Module):
        try:
            device = str(next(net.parameters()).device)
        except StopIteration:
            device = "no-params"
    durations = []
    for _ in range(2):
        watermarker.apply_watermark(wav, sample_rate=sample_rate)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(reps):
        started = time.perf_counter()
        watermarker.apply_watermark(wav, sample_rate=sample_rate)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "audio_seconds": round(len(wav) / sample_rate, 3),
        "device": device,
        "median_ms": round(statistics.median(durations), 2),
        "min_ms": round(min(durations), 2),
        "max_ms": round(max(durations), 2),
        "reps": reps,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure Perth apply_watermark cost")
    parser.add_argument("--model", default=os.environ.get("HSE_CHATTERBOX_PARITY_MODEL"))
    parser.add_argument("--audio", help="real WAV to time instead of synthetic durations")
    parser.add_argument("--durations", default="1,2,3,5,10,30")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    if not args.model:
        raise SystemExit("--model (or HSE_CHATTERBOX_PARITY_MODEL) is required")

    model, watermarker, sample_rate = _load_watermarker(args.model)
    if watermarker is None:
        raise SystemExit("the loaded model has no watermarker")

    rows = []
    if args.audio:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(args.audio, sampling_rate=sample_rate)
        wav = np.asarray(audio, dtype=np.float32)
        rows.append(_time_watermark(watermarker, wav, sample_rate, args.reps))
    else:
        for seconds in [float(x) for x in args.durations.split(",") if x.strip()]:
            wav = np.random.randn(int(sample_rate * seconds)).astype(np.float32)
            rows.append(_time_watermark(watermarker, wav, sample_rate, args.reps))

    report = {"model": args.model, "sample_rate": sample_rate, "rows": rows}
    print(json.dumps(report, indent=1))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
