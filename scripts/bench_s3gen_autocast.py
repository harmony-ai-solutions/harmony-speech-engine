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

``--mode listening`` (Phase 11-3)
    Writes the owner's listening-pass pairs: seeded ``<case>_fp32.wav`` /
    ``<case>_fp16.wav`` for each Chatterbox family whose vocode uses S3Gen
    (bundled voice, clone voice, VC speech→B, VC speech→synth, multilingual
    en + de-accent, turbo bundled) plus a one-screen ``README.md`` index. The
    fp16 side is driven by ``HSE_S3GEN_AUTOCAST=float16`` (the documented env
    override of the fp32 config default) so no shipped config is flipped.

Results feed ``docs/benchmarks/throughput.md`` §Phase 11-2 and the 11-3
listening pass. This is a measurement tool, not a test: it needs the cached
Chatterbox weights and a CUDA device, and takes a couple of minutes.

    CUDA_VISIBLE_DEVICES=1 python scripts/bench_s3gen_autocast.py --batch 1,8
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_s3gen_autocast.py --mode audio
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_s3gen_autocast.py --mode listening
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

#: Phase 11-3 listening pass: one seeded pair per row, grouped by the wrapper
#: that renders it. Each case names its text (tests/audio_fixtures.TEXTS) and,
#: for clone/VC cases, the fixture clips. The 11-2 gate column is the published
#: seeded verdict for that family (docs/benchmarks/throughput.md §Phase 11-2).
_LISTENING_GROUPS = (
    (
        "chatterbox",
        "chatterbox (en TTS): 11-2 fp16 PASS, worst 0.099 dB², corr ≥ 0.9978, RMS ≤ 0.11 %",
        (
            {"name": "chatterbox_bundled", "text": "en_short", "label": "bundled voice, plain text"},
            {
                "name": "chatterbox_clone",
                "text": "en_numbers",
                "reference": "voice_a",
                "label": "synthetic reference: timbre porting",
            },
        ),
    ),
    (
        "chatterbox_vc",
        "chatterbox VC: 11-2 fp16 PASS; batched c=8 1.51× (flow 2.45× at B=8)",
        (
            {
                "name": "vc_speech_to_b",
                "source": "repo_voice_a",
                "target": "repo_voice_b",
                "label": "real speech -> real speech timbre",
            },
            {
                "name": "vc_speech_to_synth",
                "source": "repo_voice_a",
                "target": "voice_b",
                "label": "real speech -> synthetic vowel timbre",
            },
        ),
    ),
    (
        "chatterbox_multilingual",
        "chatterbox_multilingual: 11-2 fp16 PASS, 0.009–0.013 dB²",
        (
            {"name": "multilingual_en", "text": "en_short", "language": "en", "label": "English text"},
            {
                "name": "multilingual_de_accent",
                "text": "de_short",
                "language": "de",
                "label": "German text + bundled English conditionals (accent transfer)",
            },
        ),
    ),
    (
        "chatterbox_turbo",
        "chatterbox_turbo (meanflow): 11-2 fp16 PASS, 0.020 dB²",
        (
            {"name": "turbo_bundled", "text": "en_short", "label": "bundled voice, plain text"},
        ),
    ),
)

#: Seed used for both sides of every listening pair (same tokens + noise draw).
_LISTENING_SEED = 20261108


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


# ---------------------------------------------------------------------------
# Listening mode (Phase 11-3)
# ---------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    """Seed every RNG the TTS/VC paths read, so both sides draw identically."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _render_wav(model, group: str, case: dict, device: str):
    """Render one case with the loaded wrapper and return (samples, rate)."""
    from tests.audio_fixtures import REFERENCE_CLIPS, TEXTS

    if group == "chatterbox":
        kwargs = {}
        if case.get("reference"):
            kwargs["audio_prompt_path"] = REFERENCE_CLIPS[case["reference"]]
        wav = model.generate(TEXTS[case["text"]], **kwargs)
    elif group == "chatterbox_multilingual":
        wav = model.generate(TEXTS[case["text"]], language_id=case["language"])
    elif group == "chatterbox_turbo":
        wav = model.generate(TEXTS[case["text"]])
    elif group == "chatterbox_vc":
        model.set_target_voice(REFERENCE_CLIPS[case["target"]])
        wav = model.generate(audio=REFERENCE_CLIPS[case["source"]])
    else:  # pragma: no cover - guarded by the group table
        raise ValueError(f"unknown listening group {group!r}")

    samples = wav.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return samples, int(model.sr)


def _load_listening_model(group: str, device: str):
    from harmonyspeech.modeling.models.chatterbox.chatterbox import (
        ChatterboxMultilingualTTSModel,
        ChatterboxTTSModel,
        ChatterboxTurboTTSModel,
        ChatterboxVCModel,
    )

    wrappers = {
        "chatterbox": ChatterboxTTSModel,
        "chatterbox_vc": ChatterboxVCModel,
        "chatterbox_multilingual": ChatterboxMultilingualTTSModel,
        "chatterbox_turbo": ChatterboxTurboTTSModel,
    }
    return wrappers[group].from_pretrained(device=device, dtype=torch.float32)


def run_listening(out_dir: str, device: str) -> dict:
    """Write seeded fp32/fp16 pairs per family + a one-screen README index."""
    import gc

    import soundfile as sf

    ar = _harness()
    os.makedirs(out_dir, exist_ok=True)
    report: dict = {
        "seed": _LISTENING_SEED,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "out_dir": out_dir,
        "groups": [],
    }

    for group, gate_note, cases in _LISTENING_GROUPS:
        print(f"[listening] loading {group}…", flush=True)
        model = _load_listening_model(group, device)
        rendered = []
        try:
            for case in cases:
                paths = {}
                seconds = {}
                for side, env_value in (("fp32", "off"), ("fp16", "float16")):
                    os.environ[S3GEN_AUTOCAST_ENV] = env_value
                    _seed_everything(_LISTENING_SEED)
                    with torch.inference_mode():
                        samples, rate = _render_wav(model, group, case, device)
                    path = os.path.join(out_dir, f"{case['name']}_{side}.wav")
                    sf.write(path, samples, rate, subtype="PCM_16")
                    paths[side] = path
                    seconds[side] = round(float(samples.size) / rate, 3)
                    print(
                        f"[listening] {case['name']:22s} {side} {seconds[side]:.2f}s "
                        f"-> {path}",
                        flush=True,
                    )
                metrics = ar.compute_metrics(paths["fp32"], paths["fp16"], "stochastic")
                rendered.append(
                    {
                        "case": case["name"],
                        "label": case["label"],
                        "files": {"fp32": paths["fp32"], "fp16": paths["fp16"]},
                        "seconds": seconds,
                        "metrics": metrics,
                    }
                )
                print(
                    f"[listening] {case['name']:22s} logmel_mse {metrics['logmel_mse']:.6f}  "
                    f"corr {metrics['corr']:.4f}  rms {metrics['rms_pct']:.3f}%",
                    flush=True,
                )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["groups"].append({"group": group, "gate_note": gate_note, "cases": rendered})

    os.environ[S3GEN_AUTOCAST_ENV] = "off"
    _write_listening_readme(out_dir, report)
    return report


def _write_listening_readme(out_dir: str, report: dict) -> str:
    """Render the one-screen listening index (case → files → 11-2 gate)."""
    lines = [
        "# S3Gen fp32 vs fp16 listening pass (Phase 11-3)",
        "",
        f"Host: {report['device']} · torch {report['torch']} · seed {report['seed']} · "
        f"generated 2026-11-08",
        "",
        "Each case is one seeded request rendered twice: `*_fp32.wav` (config "
        "default, no autocast) and `*_fp16.wav` (`HSE_S3GEN_AUTOCAST=float16`, the "
        "documented env override). The AR token stream is identical by "
        "construction (autocast wraps S3Gen only); listen for vocode texture, "
        "not wording. `logmel MSE` / `corr` / `RMS` are the live 4-2 harness "
        "metrics for this pair, beside the published 11-2 seeded gate for the "
        "family.",
        "",
        "| case | family | fp32 | fp16 | sec | logmel MSE (dB²) | corr | RMS Δ | 11-2 gate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group in report["groups"]:
        for case in group["cases"]:
            m = case["metrics"]
            lines.append(
                "| {case} | {group} | [{a}]({a}) | [{b}]({b}) | {sec} | {mse:.6f} | "
                "{corr:.4f} | {rms:.3f} % | {gate} |".format(
                    case=case["case"],
                    group=group["group"],
                    a=os.path.basename(case["files"]["fp32"]),
                    b=os.path.basename(case["files"]["fp16"]),
                    sec=case["seconds"]["fp32"],
                    mse=m["logmel_mse"],
                    corr=m["corr"],
                    rms=m["rms_pct"],
                    gate=group["gate_note"],
                )
            )
    lines += [
        "",
        "The 11-2 verdict: **fp16 PASS on all 7 seeded cases; bf16 rejected.** "
        "Throughput win is vocode-bound: batched VC c=8 1.51× (flow 2.45× at B=8); "
        "AR-dominated en/turbo floods within noise. See "
        "[`docs/models.md` §Precision](../../../docs/models.md#precision-dtype-and-shared-backbones) "
        "and [`docs/benchmarks/throughput.md` §Phase 11-2](../../../docs/benchmarks/throughput.md#phase-11-2--s3gen-autocast).",
        "",
        "This folder is untracked measurement scaffolding (`out/bench/`); never commit it.",
        "",
    ]
    path = os.path.join(out_dir, "README.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"[listening] wrote {path}", flush=True)
    return path


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
    parser = argparse.ArgumentParser(description="S3Gen autocast experiment probe (Phase 11-2/11-3)")
    parser.add_argument(
        "--mode", choices=("timing", "audio", "listening", "both"), default="both"
    )
    parser.add_argument("--batch", default="1,8", help="comma-separated batch sizes (timing mode)")
    parser.add_argument("--repeat", type=int, default=5, help="median-of-N per measurement")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="out/bench/autocast-audio")
    parser.add_argument(
        "--listening-out-dir",
        default="out/bench/listening-11-3",
        help="output folder for the Phase 11-3 fp32/fp16 listening pairs",
    )
    parser.add_argument("--json-out", default="out/bench/s3gen-autocast-probe.json")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a CUDA device (autocast is a CUDA-only experiment)")
    print(f"[probe] device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    report: dict = {"modes": args.mode}
    if args.mode == "listening":
        report["listening"] = run_listening(args.listening_out_dir, args.device)
    else:
        model, ref_dict, source_path = _load_model(args.device)
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
