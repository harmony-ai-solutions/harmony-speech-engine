"""S3Gen autocast experiment probe (Phase 11-2).

Measures the two questions the experiment asks, on real ChatterboxVC weights
through the HSE loader (so the seam under test is the shipped one):

``--mode timing``
    Per-hop wall time (decode / tokenize / flow / mel2wav / perth) and the
    end-to-end padded vocode, plus the torch activation peak of each GPU phase,
    for ``HSE_S3GEN_AUTOCAST`` = ``off`` | ``bfloat16`` | ``float16``.

``--mode audio``
    Renders the same seeded rows in each precision and scores them against the
    fp32 reference with the 4-2 harness metrics — separately for the **surgical**
    padded path (flow autocast, HiFT fp32) and the **blanket** solo path (the
    loader seam: flow + HiFT autocast).

Results feed ``docs/benchmarks/throughput.md`` §Phase 11-2. This is a
measurement tool, not a test: it needs the cached Chatterbox weights and a CUDA
device, and takes a couple of minutes.

    CUDA_VISIBLE_DEVICES=1 python scripts/bench_s3gen_autocast.py --batch 1,8
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_s3gen_autocast.py --mode audio
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harmonyspeech.modeling.models.chatterbox.s3gen_batch import (  # noqa: E402
    S3GenRow,
    _flow_batched,
    _generators,
    _hift_batched,
    inference_batched,
    tokenize_batched,
)
from harmonyspeech.modeling.models.chatterbox.s3gen_precision import (  # noqa: E402
    S3GEN_AUTOCAST_ENV,
)

MODES = ("off", "bfloat16", "float16")
_S3GEN_SR = 24000


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def median_seconds(fn, repeat: int) -> float:
    """Median wall time of ``fn`` over ``repeat`` synchronised runs."""
    samples = []
    for _ in range(max(1, repeat)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def peak_delta(fn, repeat: int = 3) -> tuple[float, int]:
    """(median seconds, peak activation bytes above the pre-phase allocation)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    elapsed = median_seconds(fn, repeat)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return elapsed, int(max(0, peak - baseline))


def _load_model(device: str):
    from harmonyspeech.modeling.models.chatterbox.chatterbox import ChatterboxVCModel
    from tests.audio_fixtures import REFERENCE_CLIPS

    model = ChatterboxVCModel.from_pretrained(device=device, dtype=torch.float32)
    model.set_target_voice(REFERENCE_CLIPS["repo_voice_b"])
    return model, dict(model.ref_dict), REFERENCE_CLIPS["repo_voice_a"]


def _source_audio(path: str) -> torch.Tensor:
    import librosa

    from chatterbox.models.s3tokenizer import S3_SR

    audio, _ = librosa.load(path, sr=S3_SR)
    return torch.from_numpy(audio).float()


# ---------------------------------------------------------------------------
# Timing mode
# ---------------------------------------------------------------------------
def run_timing(model, ref_dict: dict, source_path: str, batches, repeat: int) -> dict:
    import librosa

    from chatterbox.models.s3tokenizer import S3_SR

    device = "cuda"
    s3gen = model.s3gen
    flow, mel2wav = s3gen.flow, s3gen.mel2wav
    watermarker = model.watermarker
    sr = int(model.sr)
    decode_s = median_seconds(lambda: librosa.load(source_path, sr=S3_SR), repeat)
    audio_t = _source_audio(source_path)

    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "source_seconds": round(audio_t.numel() / S3_SR, 3),
        "decode_ms": round(decode_s * 1e3, 2),
        "modes": {},
    }
    for batch in batches:
        batch_report = {}
        for mode in MODES:
            os.environ[S3GEN_AUTOCAST_ENV] = mode
            clips = [audio_t] * batch
            with torch.inference_mode():
                tokenize_s = median_seconds(lambda: tokenize_batched(s3gen.tokenizer, clips), repeat)
                tokens = tokenize_batched(s3gen.tokenizer, clips)
                rows = [
                    S3GenRow(tokens=tokens[i].clone(), ref_dict=dict(ref_dict), seed=1000 + i)
                    for i in range(batch)
                ]

                def run_flow():
                    return _flow_batched(
                        flow, rows, _generators(rows, device), n_timesteps=10
                    )

                flow_s, flow_peak = peak_delta(run_flow)
                mels = run_flow()

                def run_hift():
                    return _hift_batched(mel2wav, mels, _generators(rows, device))

                hift_s, hift_peak = peak_delta(run_hift)
                waveforms = run_hift()
                wav_np = waveforms[0].detach().cpu().numpy()
                perth_s = median_seconds(
                    lambda: watermarker.apply_watermark(wav_np, sample_rate=sr), repeat
                )
                e2e_s, e2e_peak = peak_delta(lambda: inference_batched(s3gen, rows))

                solo_s = None
                if batch == 1:
                    solo_row = S3GenRow(
                        tokens=tokens[0].clone(), ref_dict=dict(ref_dict), seed=None
                    )
                    solo_s = median_seconds(
                        lambda: inference_batched(s3gen, [solo_row]), repeat
                    )

            batch_report[mode] = {
                "tokenize_ms": round(tokenize_s * 1e3, 2),
                "flow_ms": round(flow_s * 1e3, 2),
                "flow_peak_mib": round(flow_peak / 2**20, 1),
                "mel2wav_ms": round(hift_s * 1e3, 2),
                "mel2wav_peak_mib": round(hift_peak / 2**20, 1),
                "perth_ms": round(perth_s * 1e3, 2),
                "e2e_ms": round(e2e_s * 1e3, 2),
                "e2e_peak_mib": round(e2e_peak / 2**20, 1),
                "solo_upstream_ms": None if solo_s is None else round(solo_s * 1e3, 2),
                "audio_seconds": round(sum(int(w.numel()) for w in waveforms) / sr, 3),
            }
            print(
                f"[timing] B={batch} {mode:8s} tokenize {tokenize_s * 1e3:7.1f}  "
                f"flow {flow_s * 1e3:7.1f}  mel2wav {hift_s * 1e3:7.1f}  "
                f"perth {perth_s * 1e3:6.1f}  e2e {e2e_s * 1e3:7.1f} ms | "
                f"peaks flow {flow_peak / 2**20:6.1f} + hift {hift_peak / 2**20:6.1f} MiB",
                flush=True,
            )
        report["modes"][str(batch)] = batch_report
    os.environ[S3GEN_AUTOCAST_ENV] = "off"
    return report


# ---------------------------------------------------------------------------
# Audio mode
# ---------------------------------------------------------------------------
def _harness():
    from scripts import audio_regression

    return audio_regression


def run_audio(model, ref_dict: dict, source_path: str, out_dir: str) -> dict:
    """Seeded renders scored against the fp32 reference (surgical vs blanket)."""
    import soundfile as sf

    ar = _harness()
    os.makedirs(out_dir, exist_ok=True)
    s3gen = model.s3gen
    audio_t = _source_audio(source_path)
    report: dict = {"sr": _S3GEN_SR, "cases": {}}

    def render(batch: int, mode: str, *, solo: bool = False):
        os.environ[S3GEN_AUTOCAST_ENV] = mode
        torch.manual_seed(0)  # the solo/blanket path draws from the global RNG
        with torch.inference_mode():
            tokens = s3gen.tokenizer(audio_t[None].to("cuda"))[0]
            if solo:
                rows = [S3GenRow(tokens=tokens[0].clone(), ref_dict=dict(ref_dict), seed=None)]
            else:
                rows = [
                    S3GenRow(tokens=tokens[0].clone(), ref_dict=dict(ref_dict), seed=7000 + i)
                    for i in range(batch)
                ]
            wavs = inference_batched(s3gen, rows)
        return [wav.detach().cpu().numpy().astype(np.float32) for wav in wavs]

    for label, batch, solo in (
        ("surgical_b1", 1, False),
        ("surgical_b8", 8, False),
        ("blanket_solo", 1, True),
    ):
        paths = {}
        for mode in ("off", *MODES[1:]):
            waveform = render(batch, mode, solo=solo)[0]
            path = os.path.join(out_dir, f"{label}_{mode}.wav")
            sf.write(path, waveform, _S3GEN_SR, subtype="PCM_16")
            paths[mode] = path
        for mode in MODES[1:]:
            metrics = ar.compute_metrics(paths["off"], paths[mode], "deterministic")
            report["cases"][f"{label}_{mode}"] = metrics
            print(
                f"[audio] {label:13s} {mode:8s} logmel_mse {metrics['logmel_mse']:.6f}  "
                f"corr {metrics['corr']:.4f}  rms {metrics['rms_pct']:.3f}%  "
                f"duration {metrics['duration_pct']:.3f}%  "
                f"max|d| {metrics['max_abs_diff']:.5f}",
                flush=True,
            )
    os.environ[S3GEN_AUTOCAST_ENV] = "off"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S3Gen autocast experiment probe (Phase 11-2)")
    parser.add_argument("--mode", choices=("timing", "audio", "both"), default="both")
    parser.add_argument("--batch", default="1,8", help="comma-separated batch sizes (timing mode)")
    parser.add_argument("--repeat", type=int, default=5, help="median-of-N per measurement")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="out/bench/autocast-audio")
    parser.add_argument("--json-out", default="out/bench/s3gen-autocast-probe.json")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a CUDA device (autocast is a CUDA-only experiment)")
    print(f"[probe] device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    model, ref_dict, source_path = _load_model(args.device)
    report: dict = {"modes": args.mode}
    if args.mode in ("timing", "both"):
        batches = [int(value) for value in args.batch.split(",") if value.strip()]
        report["timing"] = run_timing(model, ref_dict, source_path, batches, args.repeat)
    if args.mode in ("audio", "both"):
        report["audio"] = run_audio(model, ref_dict, source_path, args.out_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[probe] wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
