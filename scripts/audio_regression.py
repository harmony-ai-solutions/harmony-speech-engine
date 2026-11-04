"""Threshold-based audio regression harness (Phase 4-2).

Compares two *configurations of the current build* along a change axis — dtype
(fp32 vs fp16), weight sharing (deduplicated vs duplicated) or a plain identity
run — on fixed inputs, and fails (non-zero exit) when a family leaves its
thresholds.

Design notes
------------
* **Two separate processes per family.** Each side boots its own engine in a
  subprocess, so VRAM is genuinely released between sides and the measurement is
  not contaminated by a resident fp32 engine. The parent orchestrates, keeps the
  metrics code, and is the only place that needs spectra/numpy heavy lifting.
* **No golden audio.** Bit-exactness across torch versions is unrealistic, so
  the harness always derives both sides of the axis in the same run.
* **Deterministic vs stochastic.** Deterministic families (HarmonySpeech,
  VoiceFixer, KittenTTS) are compared sample-wise (waveform correlation +
  frame-aligned log-mel MSE). Stochastic families (Chatterbox sampling,
  OpenVoice/MeloTTS `noise_scale`) are compared with distribution metrics:
  duration delta, RMS delta, and long-term average spectrum (LTAS) distance —
  frame-by-frame comparison of two independent samples of the same distribution
  is meaningless.
* **Sharing axis is an identity gate.** Deduplicating weights must not change a
  single bit: the harness requires identical length and bitwise-identical
  samples.

Usage
-----
    python scripts/audio_regression.py --families chatterbox,openvoice_v1 \
        --axis dtype:float32-vs-float16 --gpu

    python scripts/audio_regression.py --families chatterbox \
        --axis sharing:dedup-vs-dup --gpu

    # identity self-test (must always pass)
    python scripts/audio_regression.py --families kittentts \
        --axis dtype:float32-vs-float32
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Thresholds (initial values, tuned on the first GPU run — see
# docs/benchmarks/audio-regression.md)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "deterministic": {
        "logmel_mse": 1.0,  # dB^2 over aligned mel frames
        "duration_pct": 1.0,
        "rms_pct": 5.0,
        "corr_min": 0.98,
    },
    "stochastic": {
        "ltas_rmse_db": 4.0,  # long-term average spectrum per mel band
        "duration_pct": 12.0,
        "rms_pct": 15.0,
    },
}

#: Per-family overrides. Values are filled in from measured runs; a family whose
#: measured spread is structurally larger (e.g. an AR model whose token stream
#: legitimately diverges between two samples) gets its own numbers rather than
#: hiding the difference in a global loosening.
FAMILY_THRESHOLDS: dict[str, dict] = {}

#: Families whose dtype axis is not meaningful in the shipped configuration
#: (CPU-pinned entries are rewritten to bf16 by the CPU executor, and ONNX/CT2
#: backends ignore torch dtype entirely).
DTYPE_AXIS_EXEMPT = {
    "kittentts": "ONNX runtime ignores torch dtype (variant chosen by model name)",
    "harmonyspeech": "CPU-pinned in the shipped GPU config (torch dtype only selects bf16 on CPU)",
}


# ---------------------------------------------------------------------------
# Family catalogue (mirrors config.gpu.yml entries)
# ---------------------------------------------------------------------------
def model(
    name: str,
    model: str,
    model_type: str,
    *,
    device: str = "cuda",
    language: str | None = None,
    voices: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    entry = {
        "name": name,
        "model": model,
        "model_type": model_type,
        "max_batch_size": 1,
        "device": device,
        "language": language,
        "voices": voices,
    }
    entry.update(extra or {})
    return entry


def _whisper_model():
    return model("faster-whisper", "Systran/faster-whisper-tiny", "FasterWhisper", device="cpu")


def _openvoice_v1_models():
    return [
        _whisper_model(),
        model(
            "ov1-synthesizer-en",
            "myshell-ai/openvoice",
            "OpenVoiceV1Synthesizer",
            language="EN",
            voices=["default"],
        ),
        model("ov1-tone-converter", "myshell-ai/openvoice", "OpenVoiceV1ToneConverter"),
        model("ov1-tone-converter-encoder", "myshell-ai/openvoice", "OpenVoiceV1ToneConverterEncoder"),
    ]


def _openvoice_v2_models():
    return [
        _whisper_model(),
        model(
            "ov2-synthesizer-en",
            "myshell-ai/MeloTTS-English-v3",
            "MeloTTSSynthesizer",
            language="EN",
            voices=["EN-Newest"],
        ),
        model("ov2-tone-converter", "myshell-ai/openvoicev2", "OpenVoiceV2ToneConverter"),
        model("ov2-tone-converter-encoder", "myshell-ai/openvoicev2", "OpenVoiceV2ToneConverterEncoder"),
    ]

FAMILY_SPECS: dict[str, dict] = {
    "chatterbox": {
        "models": [
            model("chatterbox", "resemble-ai/chatterbox", "ChatterboxTTS", extra={"watermark": True}),
            model("chatterbox_vc", "resemble-ai/chatterbox", "ChatterboxVC", extra={"watermark": True}),
            model("chatterbox_embedding", "resemble-ai/chatterbox", "ChatterboxEmbedding"),
        ],
        "group": "chatterbox-base",
        "cases": [
            {"name": "tts_bundled_voice", "kind": "tts", "model": "chatterbox", "text": "en_short"},
            {
                "name": "tts_clone_voice_a",
                "kind": "tts",
                "model": "chatterbox",
                "text": "en_numbers",
                "mode": "voice_cloning",
                "reference": "voice_a",
            },
            {
                "name": "vc_a_to_b",
                "kind": "vc",
                "model": "chatterbox_vc",
                "reference": "voice_a",
                "target": "voice_b",
            },
        ],
    },
    "chatterbox_turbo": {
        "models": [
            model("chatterbox_turbo", "resemble-ai/chatterbox-turbo", "ChatterboxTurboTTS", extra={"watermark": True}),
        ],
        "group": None,  # turbo stays ungrouped (own meanflow S3Gen)
        "cases": [
            {"name": "tts_bundled_voice", "kind": "tts", "model": "chatterbox_turbo", "text": "en_short"},
        ],
    },
    "chatterbox_multilingual": {
        "models": [
            model(
                "chatterbox_multilingual",
                "resemble-ai/chatterbox",
                "ChatterboxMultilingualTTS",
                extra={"watermark": True},
            ),
        ],
        "group": "chatterbox-base",  # joins only on the hash evidence in factory.py
        "cases": [
            {
                "name": "tts_en",
                "kind": "tts",
                "model": "chatterbox_multilingual",
                "text": "en_short",
                "language": "en",
            },
            {
                "name": "tts_de",
                "kind": "tts",
                "model": "chatterbox_multilingual",
                "text": "de_short",
                "language": "de",
            },
        ],
    },
    "openvoice_v1": {
        "models": _openvoice_v1_models(),
        "group": None,
        "cases": [
            {
                "name": "clone_voice_a",
                "kind": "tts",
                "model": "openvoice_v1",
                "text": "en_short",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "default",
                "reference": "repo_voice_a",
            },
            {
                "name": "clone_voice_b",
                "kind": "tts",
                "model": "openvoice_v1",
                "text": "en_numbers",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "default",
                "reference": "voice_b",
            },
        ],
    },
    "openvoice_v2": {
        "models": _openvoice_v2_models(),
        "group": None,
        "cases": [
            {
                "name": "clone_voice_a",
                "kind": "tts",
                "model": "openvoice_v2",
                "text": "en_short",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "EN-Newest",
                "reference": "repo_voice_a",
            },
        ],
    },
    "melotts": {
        "models": [
            model(
                "ov2-synthesizer-en",
                "myshell-ai/MeloTTS-English-v3",
                "MeloTTSSynthesizer",
                language="EN",
                voices=["EN-Newest"],
            )
        ],
        "group": None,
        "cases": [
            {
                "name": "tts_en",
                "kind": "tts",
                "model": "ov2-synthesizer-en",
                "text": "en_short",
                "language": "EN",
                "voice": "EN-Newest",
            },
        ],
    },
    "voicefixer": {
        "models": [
            model("voicefixer-restorer", "jlmarrugom/voice_fixer", "VoiceFixerRestorer"),
            model("voicefixer-vocoder", "jlmarrugom/voice_fixer", "VoiceFixerVocoder"),
        ],
        "group": None,
        "cases": [
            {"name": "restore_voice_a", "kind": "restore", "model": "voicefixer", "reference": "voice_a"},
        ],
    },
    "harmonyspeech": {
        "models": [
            model("hs1-encoder", "harmony-ai/harmony-speech-v1", "HarmonySpeechEncoder", device="cpu"),
            model("hs1-synthesizer", "harmony-ai/harmony-speech-v1", "HarmonySpeechSynthesizer", device="cpu"),
            model("hs1-vocoder", "harmony-ai/harmony-speech-v1", "HarmonySpeechVocoder", device="cpu"),
        ],
        "group": None,
        "cases": [
            {
                "name": "clone_voice_a",
                "kind": "tts",
                "model": "harmonyspeech",
                "text": "en_short",
                "mode": "voice_cloning",
                "reference": "repo_voice_a",
            },
        ],
    },
    "kittentts": {
        "models": [
            model(
                "kitten-tts-mini",
                "KittenML/kitten-tts-mini-0.8",
                "KittenTTSSynthesizer",
                device="cpu",
                voices=["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"],
            )
        ],
        "group": None,
        "cases": [
            {
                "name": "tts_en",
                "kind": "tts",
                "model": "kitten-tts-mini",
                "text": "en_short",
                "voice": "Bella",
                "language": "default",
            }
        ],
    },
}

# Canonical alignment rate for metric computation.
METRIC_SR = 24000


# ---------------------------------------------------------------------------
# Axis definition
# ---------------------------------------------------------------------------
@dataclass
class Axis:
    """A change axis with a label for each side."""

    kind: str  # "dtype" | "sharing" | "baseline"
    side_a: str
    side_b: str

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.side_a}-vs-{self.side_b}"

    @property
    def identity(self) -> bool:
        return self.side_a == self.side_b


def parse_axis(text: str) -> Axis:
    try:
        kind, pair = text.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"axis must look like 'dtype:float32-vs-float16', got {text!r}") from exc
    if "-vs-" not in pair:
        raise ValueError(f"axis must contain '-vs-', got {text!r}")
    a, b = pair.split("-vs-", 1)
    if kind not in {"dtype", "sharing", "baseline"}:
        raise ValueError(f"unknown axis kind '{kind}' (expected dtype, sharing or baseline)")
    return Axis(kind=kind, side_a=a, side_b=b)


def side_settings(axis: Axis, side: str) -> dict:
    """Translate one side of the axis into config knobs.

    ``dtype`` value is passed verbatim to ``ModelConfig.dtype``; ``sharing``
    values ``dedup`` / ``dup`` select whether grouped families get a
    ``backbone_group``.
    """
    if axis.kind == "dtype":
        return {"dtype": side, "share": True, "watermark": True}
    if axis.kind == "sharing":
        # PerthNet's conv kernels are not bit-reproducible across processes
        # (cuDNN algorithm choice varies with free VRAM), so the identity gate
        # runs with the dummy watermarker: it tests the synthesis path, which is
        # what sharing touches. The dtype axis keeps watermarking enabled.
        return {"dtype": "float32", "share": side == "dedup", "watermark": False}
    return {"dtype": side, "share": True, "watermark": True}


# ---------------------------------------------------------------------------
# Worker: boot one engine, run the family's cases, write wavs + manifest
# ---------------------------------------------------------------------------
def _build_model_configs(family: str, settings: dict, use_gpu: bool):
    from harmonyspeech.common.config import DeviceConfig, ModelConfig

    spec = FAMILY_SPECS[family]
    configs = []
    for entry in spec["models"]:
        device = entry["device"]
        if not use_gpu:
            device = "cpu"
        kwargs = {
            "name": entry["name"],
            "model": entry["model"],
            "model_type": entry["model_type"],
            "max_batch_size": entry["max_batch_size"],
            "device_config": DeviceConfig(device=device),
            "dtype": settings["dtype"],
        }
        if entry.get("language"):
            kwargs["language"] = entry["language"]
        if entry.get("voices"):
            kwargs["voices"] = entry["voices"]
        if entry.get("watermark") is not None:
            kwargs["watermark"] = bool(entry["watermark"]) and bool(settings.get("watermark", True))
        if spec.get("group") and settings["share"]:
            kwargs["backbone_group"] = spec["group"]
        configs.append(ModelConfig(**kwargs))
    return configs


def _decode_response_audio(response) -> tuple[np.ndarray, int]:
    import soundfile as sf

    from harmonyspeech.endpoints.openai.protocol import ErrorResponse

    if isinstance(response, ErrorResponse) or not getattr(response, "data", None):
        message = getattr(response, "message", None) or getattr(response, "error", None) or response
        raise RuntimeError(f"request failed: {message}")
    raw = base64.b64decode(response.data)
    samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32), int(rate)


def run_worker(job_path: str) -> int:
    """Boot one engine for one side of the axis and run the family's cases."""
    import asyncio
    import random

    import torch

    from harmonyspeech.endpoints.openai.protocol import (
        AudioConversionRequest,
        TextToSpeechRequest,
        VoiceConversionRequest,
    )
    from harmonyspeech.endpoints.openai.serving_audio_conversion import OpenAIServingAudioConversion
    from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
    from harmonyspeech.endpoints.openai.serving_voice_conversion import OpenAIServingVoiceConversion
    from harmonyspeech.engine.args_tools import AsyncEngineArgs
    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
    from tests.audio_fixtures import reference_wav_b64

    with open(job_path, encoding="utf-8") as handle:
        job = json.load(handle)

    model_configs = _build_model_configs(job["family"], job["settings"], job["use_gpu"])
    engine_args = AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True)
    from harmonyspeech.common.config import EngineConfig

    engine_config = EngineConfig(model_configs=model_configs)
    engine = AsyncHarmonySpeech.from_engine_args_and_config(engine_args, engine_config, start_engine_loop=True)
    serving_tts = OpenAIServingTextToSpeech(
        engine, OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
    )
    serving_vc = OpenAIServingVoiceConversion(
        engine, OpenAIServingVoiceConversion.models_from_config(engine_config.model_configs)
    )
    serving_ac = OpenAIServingAudioConversion(
        engine, OpenAIServingAudioConversion.models_from_config(engine_config.model_configs)
    )

    class _Request:
        is_disconnected = staticmethod(lambda: asyncio.sleep(0, result=False))

    os.makedirs(job["out_dir"], exist_ok=True)
    manifest = {"family": job["family"], "side": job["side"], "cases": {}}

    text_table = __import__("tests.audio_fixtures", fromlist=["TEXTS"]).TEXTS

    for case in job["cases"]:
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        started = time.time()
        try:
            kind = case["kind"]
            if kind == "tts":
                request = TextToSpeechRequest(
                    model=case["model"],
                    input=text_table[case["text"]],
                    mode=case.get("mode", "single_speaker_tts"),
                    input_audio=reference_wav_b64(case["reference"]) if case.get("reference") else None,
                    language=case.get("language"),
                    voice=case.get("voice"),
                )
                response = asyncio.run(serving_tts.create_text_to_speech(request, _Request()))
            elif kind == "vc":
                request = VoiceConversionRequest(
                    model=case["model"],
                    source_audio=reference_wav_b64(case["reference"]),
                    target_audio=reference_wav_b64(case["target"]),
                )
                response = asyncio.run(serving_vc.convert_voice(request, _Request()))
            elif kind == "restore":
                request = AudioConversionRequest(
                    model=case["model"], source_audio=reference_wav_b64(case["reference"])
                )
                response = asyncio.run(serving_ac.convert_audio(request, _Request()))
            else:
                raise ValueError(f"unknown case kind {kind!r}")
            samples, rate = _decode_response_audio(response)
        except Exception as exc:  # noqa: BLE001 - reported per case
            manifest["cases"][case["name"]] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[worker:{job['side']}] {job['family']}/{case['name']} FAILED: {exc}", flush=True)
            continue

        import soundfile as sf

        wav_path = os.path.join(job["out_dir"], f"{case['name']}.wav")
        sf.write(wav_path, samples, rate, subtype="PCM_16")
        manifest["cases"][case["name"]] = {
            "wav": wav_path,
            "sample_rate": rate,
            "samples": int(samples.size),
            "seconds": round(float(samples.size) / rate, 4),
            "elapsed_s": round(time.time() - started, 2),
        }
        print(
            f"[worker:{job['side']}] {job['family']}/{case['name']}: "
            f"{samples.size / rate:.2f}s audio in {time.time() - started:.1f}s",
            flush=True,
        )

    manifest["vram"] = engine.engine.get_vram_stats() if hasattr(engine, "engine") else None
    with open(job["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _load_wav(path: str) -> tuple[np.ndarray, int]:
    import librosa

    samples, rate = librosa.load(path, sr=None, mono=True)
    return np.asarray(samples, dtype=np.float32), int(rate)


def _resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return samples
    import librosa

    return librosa.resample(samples, orig_sr=rate, target_sr=target)


def _logmel(samples: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max)


def compute_metrics(path_a: str, path_b: str, kind: str) -> dict:
    """Compare two rendered clips and return metrics plus a pass flag."""
    a, sr_a = _load_wav(path_a)
    b, sr_b = _load_wav(path_b)
    sr = METRIC_SR
    a = _resample(a, sr_a, sr)
    b = _resample(b, sr_b, sr)

    rms_a = float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0
    rms_b = float(np.sqrt(np.mean(np.square(b)))) if b.size else 0.0
    duration_pct = abs(a.size - b.size) / max(a.size, b.size, 1) * 100.0
    rms_pct = abs(rms_a - rms_b) / max(rms_a, 1e-6) * 100.0

    metrics = {
        "duration_a": round(a.size / sr, 4),
        "duration_b": round(b.size / sr, 4),
        "duration_pct": round(duration_pct, 3),
        "rms_a": round(rms_a, 6),
        "rms_b": round(rms_b, 6),
        "rms_pct": round(rms_pct, 3),
    }

    mel_a = _logmel(a, sr)
    mel_b = _logmel(b, sr)
    frames = min(mel_a.shape[1], mel_b.shape[1])
    mel_mse = float(np.mean((mel_a[:, :frames] - mel_b[:, :frames]) ** 2))

    # Long-term average spectrum: robust to a different sampled realisation.
    ltas_a = mel_a.mean(axis=1)
    ltas_b = mel_b.mean(axis=1)
    ltas_rmse = float(np.sqrt(np.mean((ltas_a - ltas_b) ** 2)))
    ltas_corr = float(np.corrcoef(ltas_a, ltas_b)[0, 1]) if frames > 1 else 0.0

    common = min(a.size, b.size)
    corr = float(np.corrcoef(a[:common], b[:common])[0, 1]) if common > 1 else 0.0
    identical = a.size == b.size and bool(np.array_equal(a, b))
    max_abs_diff = float(np.max(np.abs(a[:common] - b[:common]))) if common else 0.0

    metrics.update(
        {
            "logmel_mse": round(mel_mse, 6),
            "ltas_rmse_db": round(ltas_rmse, 4),
            "ltas_corr": round(ltas_corr, 4),
            "corr": round(corr, 4),
            "identical": identical,
            "max_abs_diff": round(max_abs_diff, 8),
        }
    )

    thresholds = {**THRESHOLDS[kind], **FAMILY_THRESHOLDS.get(kind, {})}
    failures = []
    if duration_pct > thresholds["duration_pct"]:
        failures.append(f"duration {duration_pct:.2f}% > {thresholds['duration_pct']}%")
    if rms_pct > thresholds["rms_pct"]:
        failures.append(f"rms {rms_pct:.2f}% > {thresholds['rms_pct']}%")
    if kind == "deterministic":
        if mel_mse > thresholds["logmel_mse"]:
            failures.append(f"logmel mse {mel_mse:.3f} > {thresholds['logmel_mse']}")
        if corr < thresholds["corr_min"]:
            failures.append(f"corr {corr:.3f} < {thresholds['corr_min']}")
    else:
        if ltas_rmse > thresholds["ltas_rmse_db"]:
            failures.append(f"ltas rmse {ltas_rmse:.2f}dB > {thresholds['ltas_rmse_db']}dB")

    metrics["thresholds"] = thresholds
    metrics["failures"] = failures
    metrics["passed"] = not failures
    return metrics


#: Weight-sharing gate. Deduplication must not alter computation. Sample-exact
#: equality is asserted in-process by
#: tests/e2e/tts/test_chatterbox_sharing_identity.py; the cross-process harness
#: gate additionally tolerates last-bit CUDA kernel selection noise (measured
#: max |Δsample| ≈ 1.5e-4, −76 dBFS — four orders of magnitude below any wiring
#: change) while requiring identical length and an inaudible mel distance.
SHARING_MAX_ABS_DIFF = 1e-3
SHARING_LOG_MEL_MSE = 1e-2


def verdict_for_axis(axis: Axis, metrics: dict) -> tuple[bool, list[str]]:
    """Apply axis-specific rules on top of the family metrics."""
    if axis.identity:
        # Same configuration twice must be bit-identical.
        if not metrics["identical"]:
            return False, ["not bit-identical (identity axis must not change a sample)"]
        return True, []

    if axis.kind == "sharing":
        failures = []
        if metrics["duration_pct"] != 0.0:
            failures.append(f"duration changed ({metrics['duration_pct']}%) — sharing altered the token stream")
        if metrics["max_abs_diff"] > SHARING_MAX_ABS_DIFF:
            failures.append(
                f"max |Δsample| {metrics['max_abs_diff']} > {SHARING_MAX_ABS_DIFF} (sharing changed the output)"
            )
        if metrics["logmel_mse"] > SHARING_LOG_MEL_MSE:
            failures.append(f"logmel mse {metrics['logmel_mse']} > {SHARING_LOG_MEL_MSE}")
        return (not failures), failures

    return metrics["passed"], metrics["failures"]


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------
def _gpu_description() -> dict:
    info = {
        "torch": None,
        "cuda": False,
        "gpu": None,
        "driver": None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["total_vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # pragma: no cover
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0:
            info["driver"] = out.stdout.strip().splitlines()[0]
    except Exception:  # pragma: no cover
        pass
    return info


def _run_side(family: str, side: str, axis: Axis, use_gpu: bool, workdir: str) -> dict:
    out_dir = os.path.join(workdir, f"{family}_{side}")
    manifest_path = os.path.join(workdir, f"{family}_{side}.json")
    job = {
        "family": family,
        "side": side,
        "settings": side_settings(axis, side),
        "use_gpu": use_gpu,
        "cases": FAMILY_SPECS[family]["cases"],
        "out_dir": out_dir,
        "manifest": manifest_path,
    }
    job_path = os.path.join(workdir, f"{family}_{side}_job.json")
    with open(job_path, "w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", _REPO_ROOT)
    started = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", job_path],
        cwd=_REPO_ROOT,
        env=env,
    )
    elapsed = round(time.time() - started, 1)
    if proc.returncode != 0 or not os.path.exists(manifest_path):
        return {"error": f"worker exited {proc.returncode}", "elapsed_s": elapsed}
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["elapsed_s"] = elapsed
    return manifest


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, timeout=15
        ).stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def _markdown_table(report: dict) -> str:
    lines = [
        f"## {report['timestamp_utc']} — `{report['axis']}`",
        "",
        f"* commit `{report.get('commit') or 'dirty'}` · torch {report['env'].get('torch')} · "
        f"{report['env'].get('gpu') or 'CPU'} · driver {report['env'].get('driver') or 'n/a'} · "
        f"mode {'GPU' if report['use_gpu'] else 'CPU'}",
        "",
        "| Family | Case | Kind | Verdict | duration Δ% | RMS Δ% | log-mel MSE | LTAS RMSE dB | corr | identical |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for family, detail in report["families"].items():
        for case_name, case in detail.get("cases", {}).items():
            if "metrics" not in case:
                lines.append(
                    f"| {family} | {case_name} | - | ERROR | - | - | - | - | - | {case.get('error', '')[:60]} |"
                )
                continue
            m = case["metrics"]
            verdict = "PASS" if case["passed"] else "FAIL"
            lines.append(
                f"| {family} | {case_name} | {detail['kind']} | {verdict} | {m['duration_pct']} | "
                f"{m['rms_pct']} | {m['logmel_mse']} | {m['ltas_rmse_db']} | {m['corr']} | {m['identical']} |"
            )
    lines.append("")
    return "\n".join(lines)


def run_harness(args) -> int:
    axis = parse_axis(args.axis)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILY_SPECS]
    if unknown:
        raise SystemExit(f"unknown families: {', '.join(unknown)}")

    use_gpu = bool(args.gpu)
    if use_gpu:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--gpu requested but CUDA is not available")

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": axis.label,
        "use_gpu": use_gpu,
        "commit": _git_commit(),
        "env": _gpu_description(),
        "families": {},
        "passed": True,
    }

    for family in families:
        if axis.kind == "dtype" and family in DTYPE_AXIS_EXEMPT and not axis.identity:
            report["families"][family] = {
                "kind": "skipped",
                "skip_reason": DTYPE_AXIS_EXEMPT[family],
                "cases": {},
            }
            print(f"[harness] {family}: SKIP dtype axis — {DTYPE_AXIS_EXEMPT[family]}", flush=True)
            continue

        kind = _family_kind(family)
        print(f"[harness] {family}: side A ({axis.side_a})…", flush=True)
        side_a = _run_side(family, axis.side_a, axis, use_gpu, workdir)
        print(f"[harness] {family}: side B ({axis.side_b})…", flush=True)
        side_b = _run_side(family, axis.side_b, axis, use_gpu, workdir)

        family_entry = {"kind": kind, "cases": {}, "side_a": side_a, "side_b": side_b}
        if "error" in side_a or "error" in side_b:
            family_entry["error"] = side_a.get("error") or side_b.get("error")
            report["passed"] = False
            report["families"][family] = family_entry
            continue

        for case in FAMILY_SPECS[family]["cases"]:
            name = case["name"]
            a = side_a["cases"].get(name, {})
            b = side_b["cases"].get(name, {})
            if a.get("error") or b.get("error"):
                family_entry["cases"][name] = {
                    "error": a.get("error") or b.get("error"),
                    "passed": False,
                }
                report["passed"] = False
                continue
            metrics = compute_metrics(a["wav"], b["wav"], kind)
            passed, failures = verdict_for_axis(axis, metrics)
            family_entry["cases"][name] = {
                "metrics": metrics,
                "passed": passed,
                "failures": failures,
            }
            if not passed:
                report["passed"] = False
                _write_listening_pack(workdir, family, name, a["wav"], b["wav"])
            print(
                f"[harness] {family}/{name}: {'PASS' if passed else 'FAIL'}"
                + (f" — {'; '.join(failures)}" if failures else ""),
                flush=True,
            )

        if axis.kind == "dtype" or axis.identity:
            family_entry["vram"] = {
                side: {
                    "allocated_bytes": (manifest.get("vram") or {}).get("allocated_bytes"),
                    "reserved_bytes": (manifest.get("vram") or {}).get("reserved_bytes"),
                }
                for side, manifest in ((axis.side_a, side_a), (axis.side_b, side_b))
            }
        report["families"][family] = family_entry

    report["passed"] = bool(report["passed"]) and all(
        detail.get("kind") == "skipped" or all(c.get("passed") for c in detail["cases"].values())
        for detail in report["families"].values()
    )

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    table = _markdown_table(report)
    if args.md_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
        with open(args.md_out, "a", encoding="utf-8") as handle:
            handle.write(table)

    print(table)
    print(f"[harness] overall: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


def _family_kind(family: str) -> str:
    from tests.audio_fixtures import FAMILY_KIND

    return FAMILY_KIND.get(family, "deterministic")


def _write_listening_pack(workdir: str, family: str, case: str, path_a: str, path_b: str) -> None:
    """Copy A/B clips into out/regression-pack/ for the human listening pass."""
    pack_dir = os.path.join(_REPO_ROOT, "out", "regression-pack", family)
    os.makedirs(pack_dir, exist_ok=True)
    import shutil

    shutil.copyfile(path_a, os.path.join(pack_dir, f"{case}__A.wav"))
    shutil.copyfile(path_b, os.path.join(pack_dir, f"{case}__B.wav"))
    print(f"[harness] listening pack: {pack_dir}/{case}__A.wav vs {case}__B.wav", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audio regression gate (Phase 4-2)")
    parser.add_argument("--families", default="chatterbox", help="comma-separated family ids")
    parser.add_argument("--axis", default="dtype:float32-vs-float16", help="change axis")
    parser.add_argument("--gpu", action="store_true", help="require a CUDA host")
    parser.add_argument("--json-out", default="out/audio-regression/latest.json")
    parser.add_argument("--md-out", default=None, help="append the markdown result table here")
    parser.add_argument("--workdir", default="out/audio-regression")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return run_worker(args.worker)
    return run_harness(args)


if __name__ == "__main__":
    raise SystemExit(main())
