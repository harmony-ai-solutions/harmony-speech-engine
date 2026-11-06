"""Throughput / latency benchmark against a *running* engine (Phase 5-3).

Measures the honest per-family concurrency of the shipped engine over HTTP:
p50/p95 per-request latency and requests/second at concurrency {1, 2, 4, 8}.
Run it twice — once against a config with the old batch values (or
``batched_inference: false``) and once against the shipped config — and the
delta is the phase's evidence.

Families
--------
``melo``
    A MeloTTS one-shot synthesizer (padded-batch forward, ``max_batch_size: 8``).
``openvoice_v2``
    The full OpenVoice V2 clone toolchain (Whisper VAD → tone-converter encoder
    → MeloTTS synthesis), i.e. the end-to-end request a client actually sends.
``chatterbox``
    Chatterbox TTS, the honesty check: autoregressive, ``max_batch_size: 1``, so
    extra concurrency must *queue*, not speed up.
``whisper``
    Faster-Whisper STT (``BatchedInferencePipeline``, ``max_batch_size: 16``).

Usage
-----
    # terminal 1 (idle GPU)
    CUDA_VISIBLE_DEVICES=1 python -m harmonyspeech.endpoints.openai.api_server \
        --config-file-path out/bench/config-post.yml --port 2242

    # terminal 2
    python scripts/bench_throughput.py --label post \
        --md-out docs/benchmarks/throughput.md

Reference clips come from ``tests/audio_fixtures`` (synthetic + repository
samples), so a run needs no external assets.

Clone flood (Phase 7)
---------------------
``--clone-reference same`` sends the ``--reference`` clip on every request (a
worker flood: the same voice cloned concurrently). ``--clone-reference distinct``
derives a deterministic variant per request, so every request has a different
``content_hash`` and therefore a guaranteed embedding-cache miss — the worst
case for the cross-request cache. Run each mode twice, once against a server
booted with the cache enabled and once with ``--no-embedding-cache``, and record
the four combinations in ``docs/benchmarks/throughput.md``:

    python scripts/bench_throughput.py --families openvoice_v2 --concurrency 1,4,20 \\
        --requests 20 --clone-reference same --label phase7-same-cache-on \\
        --config-note "cache on (default)" --md-out docs/benchmarks/throughput.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#: Per-family request payload templates. ``model`` may be overridden from the
#: command line when the deployment names its entries differently.
FAMILY_SPECS: dict[str, dict] = {
    "melo": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "ov2-synthesizer-en",
        "payload": {
            "mode": "single_speaker_tts",
            "language": "EN",
            "voice": "EN-Newest",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
        },
    },
    "openvoice_v2": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "openvoice_v2",
        "payload": {
            "mode": "voice_cloning",
            "language": "EN",
            "voice": "EN-Newest",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
            # filled in with the reference clip at run time
            "input_audio": "__REFERENCE_AUDIO__",
        },
    },
    "chatterbox": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "chatterbox",
        "payload": {
            "mode": "single_speaker_tts",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
        },
    },
    "whisper": {
        "path": "/v1/audio/transcriptions",
        "kind": "stt",
        "model": "faster-whisper-tiny",
        "payload": {"input_audio": "__REFERENCE_AUDIO__"},
    },
}


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    failures: int
    wall_s: float
    latencies_s: list[float] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        return self.requests / self.wall_s if self.wall_s > 0 else 0.0

    def percentile(self, fraction: float) -> float:
        if not self.latencies_s:
            return 0.0
        ordered = sorted(self.latencies_s)
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def as_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "requests": self.requests,
            "failures": self.failures,
            "wall_s": round(self.wall_s, 3),
            "throughput_rps": round(self.throughput, 2),
            "p50_ms": round(self.percentile(0.50) * 1000, 1),
            "p95_ms": round(self.percentile(0.95) * 1000, 1),
            "mean_ms": round(statistics.fmean(self.latencies_s) * 1000, 1) if self.latencies_s else 0.0,
        }


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=_REPO_ROOT,
                timeout=15,
            ).stdout.strip()
            or None
        )
    except Exception:  # pragma: no cover
        return None


def _gpu_description() -> dict:
    info = {"torch": None, "gpu": None, "driver": None, "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    except Exception:  # pragma: no cover
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            info["driver"] = result.stdout.strip().splitlines()[0]
    except Exception:  # pragma: no cover
        pass
    return info


def _build_payload(family: str, reference_audio_b64: str, text_override: str | None = None) -> dict:
    spec = FAMILY_SPECS[family]
    payload = {"model": spec["model"], **spec["payload"]}
    if text_override and "input" in payload:
        payload["input"] = text_override
    for key, value in list(payload.items()):
        if value == "__REFERENCE_AUDIO__":
            payload[key] = reference_audio_b64
    return payload


#: ``--clone-reference`` modes for the Phase 7 clone-flood scenario.
CLONE_REFERENCE_MODES = ("same", "distinct")

#: Samples the per-request distinct variants are rotated by. Any non-zero value
#: works; 37 is prime relative to typical clip lengths so the rotations do not
#: line up between requests.
_DISTINCT_SHIFT_SAMPLES = 37


def _variant_waveform(waveform, index: int):
    """Return a deterministic, speech-length variant of ``waveform`` for request ``index``.

    Index 0 is the untouched base clip (so a ``same`` run and the first
    ``distinct`` request measure the same audio). Later requests are rotated by a
    deterministic sample offset plus a small gain change: the clip keeps its
    duration, sample rate and speech shape (VAD still segments it, the embedder
    still works) but every request decodes to a different
    :attr:`~harmonyspeech.common.artifacts.AudioArtifact.content_hash` — i.e. a
    guaranteed cache miss.
    """
    if index == 0:
        return waveform
    import numpy as np

    shift = (index * _DISTINCT_SHIFT_SAMPLES) % max(1, len(waveform))
    variant = np.roll(waveform, shift)
    variant = variant * np.float32(1.0 + 0.001 * index)
    peak = float(np.max(np.abs(variant))) or 1.0
    if peak > 0.99:
        variant = variant * np.float32(0.99 / peak)
    return variant.astype(waveform.dtype, copy=False)


def distinct_reference_variants(reference_audio_b64: str, count: int) -> list[str]:
    """Return ``count`` base64 WAVs, each decoding to a unique ``content_hash``.

    Used by ``--clone-reference distinct`` to model the cache-miss worst case:
    every concurrent request clones a different reference voice. Deterministic
    per (clip, index) so a repeated run measures the same audio.
    """
    import base64

    from harmonyspeech.common import audio_codec

    if count <= 0:
        return []
    raw = base64.b64decode(reference_audio_b64)
    # decode_audio normalises to mono float32, so variants are independent of the
    # source container (wav/flac/mp3) exactly like the engine's admission decode.
    waveform, sample_rate = audio_codec.decode_audio(raw)
    variants: list[str] = []
    for index in range(count):
        encoded = audio_codec.encode_audio(_variant_waveform(waveform, index), sample_rate, "wav")
        variants.append(base64.b64encode(encoded).decode("ascii"))
    return variants


def _reference_payloads(
    family: str,
    reference_audio_b64: str,
    clone_reference: str = "same",
    count: int = 1,
    text_override: str | None = None,
) -> list[dict]:
    """Build one request payload per slot (``count`` of them) for a family.

    ``same`` repeats the single ``--reference`` clip (the worker-flood pattern);
    ``distinct`` gives every slot its own deterministic variant (cache-miss path).
    Families without reference audio ignore the mode.
    """
    if clone_reference not in CLONE_REFERENCE_MODES:
        raise ValueError(f"unknown clone-reference mode: {clone_reference!r}")
    uses_reference = any(value == "__REFERENCE_AUDIO__" for value in FAMILY_SPECS[family]["payload"].values())
    if not uses_reference:
        # Nothing to vary: one payload is reused for every request.
        return [_build_payload(family, reference_audio_b64, text_override)]
    if clone_reference == "distinct":
        references = distinct_reference_variants(reference_audio_b64, max(1, count))
    else:
        references = [reference_audio_b64] * max(1, count)
    return [_build_payload(family, reference, text_override) for reference in references]


async def _one_request(client, base_url: str, family: str, payload: dict, timeout: float):
    started = time.perf_counter()
    response = await client.post(base_url + FAMILY_SPECS[family]["path"], json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    ok = response.status_code == 200
    if ok:
        body = response.json()
        ok = bool(body.get("data") or body.get("text"))
    return elapsed, ok, response.status_code


async def _run_level(client, base_url, family, payloads, concurrency: int, total: int, timeout: float):
    queue: asyncio.Queue = asyncio.Queue()
    for index in range(total):
        queue.put_nowait(index)
    latencies: list[float] = []
    failures = 0
    lock = asyncio.Lock()
    payload_count = max(1, len(payloads))

    async def worker():
        nonlocal failures
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            payload = payloads[index % payload_count]
            elapsed, ok, _status = await _one_request(client, base_url, family, payload, timeout)
            async with lock:
                if ok:
                    latencies.append(elapsed)
                else:
                    failures += 1

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - started
    return LevelResult(
        concurrency=concurrency,
        requests=total,
        failures=failures,
        wall_s=wall,
        latencies_s=latencies,
    )


async def _discover_models(client, base_url: str) -> set[str]:
    names: set[str] = set()
    for path in ("/v1/audio/speech/models", "/v1/audio/transcriptions/models"):
        try:
            response = await client.get(base_url + path, timeout=30)
            if response.status_code != 200:
                continue
            for card in response.json().get("data", []):
                names.add(card.get("id"))
                for member in card.get("models") or []:
                    names.add(member)
        except Exception:  # pragma: no cover - discovery is best effort
            continue
    return names


async def run_benchmark(args) -> dict:
    import httpx

    from tests.audio_fixtures import reference_wav_b64

    reference_audio = reference_wav_b64(args.reference)
    families = [name.strip() for name in args.families.split(",") if name.strip()]
    unknown = [name for name in families if name not in FAMILY_SPECS]
    if unknown:
        raise SystemExit(f"unknown families: {', '.join(unknown)}")

    levels = [int(value) for value in args.concurrency.split(",")]
    overtime: dict[str, list[LevelResult]] = {}
    skipped: dict[str, str] = {}

    async with httpx.AsyncClient() as client:
        available = await _discover_models(client, args.base_url) if not args.force else set()
        if available:
            print(f"[bench] advertised models: {', '.join(sorted(available))}", flush=True)

        for family in families:
            spec = FAMILY_SPECS[family]
            model = args.model.get(family) or spec["model"]
            if available and model not in available:
                skipped[family] = f"model '{model}' is not configured on the target engine"
                print(f"[bench] {family}: SKIP — {skipped[family]}", flush=True)
                continue
            payloads = _reference_payloads(
                family,
                reference_audio,
                clone_reference=getattr(args, "clone_reference", "same"),
                count=args.requests,
                text_override=args.text,
            )
            for payload in payloads:
                payload["model"] = model

            for _ in range(args.warmup):
                await _one_request(client, args.base_url, family, payloads[0], args.timeout)

            results = []
            for level in levels:
                result = await _run_level(
                    client, args.base_url, family, payloads, level, args.requests, args.timeout
                )
                results.append(result)
                print(
                    f"[bench] {family} c={level}: {result.throughput:.2f} req/s "
                    f"p50={result.percentile(0.5) * 1000:.0f}ms p95={result.percentile(0.95) * 1000:.0f}ms "
                    f"failures={result.failures}",
                    flush=True,
                )
            overtime[family] = results

    return {
        "label": args.label,
        "config": args.config_note,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_commit(),
        "env": _gpu_description(),
        "base_url": args.base_url,
        "requests_per_level": args.requests,
        "clone_reference": getattr(args, "clone_reference", "same"),
        "families": {
            family: [result.as_dict() for result in results] for family, results in overtime.items()
        },
        "skipped": skipped,
    }


def _markdown(report: dict) -> str:
    env = report["env"]
    lines = [
        f"### {report['label']} — {report['timestamp_utc']}",
        "",
        f"* commit `{report.get('commit') or 'dirty'}` · torch {env.get('torch')} · "
        f"{env.get('gpu') or 'CPU'} · driver {env.get('driver') or 'n/a'} · "
        f"CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices') or 'all'}",
        f"* config: {report.get('config') or 'n/a'}",
        f"* clone reference: {report.get('clone_reference') or 'same'}",
        f"* {report['requests_per_level']} requests per concurrency level, HTTP against {report['base_url']}",
        "",
        "| Family | Concurrency | req/s | p50 (ms) | p95 (ms) | mean (ms) | failures |",
        "|---|---|---|---|---|---|---|",
    ]
    for family, levels in report["families"].items():
        for level in levels:
            lines.append(
                f"| {family} | {level['concurrency']} | {level['throughput_rps']} | {level['p50_ms']} | "
                f"{level['p95_ms']} | {level['mean_ms']} | {level['failures']} |"
            )
    if report.get("skipped"):
        lines.append("")
        for family, reason in report["skipped"].items():
            lines.append(f"* {family}: skipped — {reason}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Throughput benchmark against a running engine (Phase 5-3 / 7-3)")
    parser.add_argument("--base-url", default="http://127.0.0.1:2242")
    parser.add_argument("--families", default="melo,openvoice_v2,chatterbox,whisper")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=24, help="requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--reference", default="repo_voice_a", help="reference clip name from tests/audio_fixtures")
    parser.add_argument(
        "--clone-reference",
        choices=CLONE_REFERENCE_MODES,
        default="same",
        help=(
            "clone-flood reference mode: 'same' sends the --reference clip on every request (worker flood); "
            "'distinct' gives every request its own deterministic variant (embedding-cache miss worst case)"
        ),
    )
    parser.add_argument(
        "--text",
        default=None,
        help="override the text input for TTS families (use to check length sensitivity of the batching gain)",
    )
    parser.add_argument("--label", default="run", help="label for the recorded block (pre/post)")
    parser.add_argument("--config-note", default=None, help="free-form config description recorded in the table")
    parser.add_argument("--model", action="append", default=[], metavar="FAMILY=MODEL", help="override a family's model id")
    parser.add_argument("--force", action="store_true", help="run every family even if it is not advertised")
    parser.add_argument("--json-out", default="out/bench/throughput-latest.json")
    parser.add_argument("--md-out", default=None, help="append the markdown table to this file")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.model = dict(item.split("=", 1) for item in args.model)

    report = asyncio.run(run_benchmark(args))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    table = _markdown(report)
    if args.md_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
        with open(args.md_out, "a", encoding="utf-8") as handle:
            handle.write(table)
    print(table)

    failures = sum(level["failures"] for levels in report["families"].values() for level in levels)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
