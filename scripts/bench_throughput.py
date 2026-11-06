"""Throughput / latency benchmark against a *running* engine (Phase 5-3 / 7-3 / 8-6).

Measures the honest per-family concurrency of the shipped engine over HTTP:
p50/p95 per-request latency and requests/second at a list of concurrency levels.
Run it twice — once against a config with the old batch values (or
``batched_inference: false``) and once against the shipped config — and the
delta is the phase's evidence.

Families
--------
``melo``
    A MeloTTS one-shot synthesizer (padded-batch forward, ``max_batch_size: 8``).
``openvoice_v2`` / ``openvoice_v1``
    The full OpenVoice clone toolchain (Whisper VAD → tone-converter encoder →
    synthesis → tone converter), i.e. the end-to-end request a client sends.
``harmonyspeech``
    The Harmony Speech clone toolchain (encoder → synthesizer → vocoder).
``voicefixer``
    VoiceFixer restoration (restorer → vocoder) over ``/v1/audio/convert``.
``kitten``
    KittenTTS (one ONNX session, ``max_batch_size`` = lane count).
``chatterbox``
    Chatterbox TTS, the honesty check: autoregressive, ``max_batch_size: 1``, so
    extra concurrency must *queue*, not speed up.
``whisper``
    Faster-Whisper STT (``BatchedInferencePipeline``, ``max_batch_size: 16``).

Usage
-----
    # terminal 1 (idle GPU, DEBUG so the per-step composition is logged)
    CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=DEBUG \
        python -m harmonyspeech.endpoints.openai.api_server \
        --config-file-path out/bench/config-post.yml --port 2242 \
        > out/bench/engine.log 2>&1

    # terminal 2
    python scripts/bench_throughput.py --label post \
        --concurrency 1,4,20 --engine-log out/bench/engine.log \
        --md-out docs/benchmarks/throughput.md

Reference clips come from ``tests/audio_fixtures`` (synthetic + repository
samples), so a run needs no external assets.

Clone flood (Phase 7 / 8)
-------------------------
``--clone-reference same`` sends the ``--reference`` clip on every request (a
worker flood: the same voice cloned concurrently). ``--clone-reference
distinct`` derives a deterministic variant per request, so every request has a
different ``content_hash`` and therefore a guaranteed embedding-cache miss — the
worst case for the cross-request cache.

The cache toggle is **server-side**: this script measures an already-running
engine, so ``{same,distinct} × {cache on, cache off}`` is four runs against two
server boots (``--no-embedding-cache`` for the off rows). ``--embedding-cache
{auto,on,off}`` verifies the server's actual state against ``/health`` and aborts
on a mismatch instead of mislabelling a run; ``--print-scenarios`` prints the
whole recipe. ``scripts/bench_throughput.py --families openvoice_v2
--concurrency 1,4,20 --requests 20 --clone-reference same \\
--embedding-cache on --label phase8-same-cache-on \\
--config-note "cache on (default)" --md-out docs/benchmarks/throughput.md``

``--concurrency`` takes a comma list (``1,4,20``), and the HTTP client's
connection pool is sized from the highest level, because ``httpx``'s default of
100 connections silently queues requests at concurrency 200 and would look like
an engine ceiling. Pass ``--http-max-connections`` to probe a lower client
ceiling deliberately.

Per-step composition (``--engine-log``) parses the engine's DEBUG sub-batch
lines (Phase 5-2 / 8-2..8-5 handlers log ``... forwarded in N padded
sub-batch(es)``) for the lines emitted during each concurrency level and reports
how many padded forwards one engine step needed, next to end-to-end p50/p95 and
req/s.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
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
    "openvoice_v1": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "openvoice_v1",
        "payload": {
            "mode": "voice_cloning",
            "language": "EN",
            "voice": "default",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
            "input_audio": "__REFERENCE_AUDIO__",
        },
    },
    "harmonyspeech": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "harmonyspeech",
        "payload": {
            "mode": "voice_cloning",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
            "input_audio": "__REFERENCE_AUDIO__",
        },
    },
    "voicefixer": {
        "path": "/v1/audio/convert",
        "kind": "restore",
        "model": "voicefixer",
        "payload": {
            # restoration takes the clip to restore, not a reference voice
            "source_audio": "__REFERENCE_AUDIO__",
        },
    },
    "kitten": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "kitten-tts-mini",
        "payload": {
            "mode": "single_speaker_tts",
            "language": "default",
            "voice": "Jasper",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
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
    #: Per-step sub-batch composition parsed from the engine DEBUG log for this
    #: level (Phase 8-6). Empty when no ``--engine-log`` was given.
    steps: dict[str, dict] = field(default_factory=dict)

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
            "steps": self.steps,
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


#: Connection-pool ceiling for the load client. ``httpx`` defaults to 100
#: connections, so a c=200 run would queue client-side and be reported as an
#: engine ceiling; the pool is sized from the highest requested level instead.
def http_client_limits(levels: list[int], max_connections: int | None = None) -> tuple[int, int]:
    """Return ``(max_connections, max_keepalive_connections)`` for the load client."""
    peak = max(levels) if levels else 1
    connections = int(max_connections) if max_connections else peak
    connections = max(1, connections)
    return connections, connections


def parse_concurrency_levels(value: str) -> list[int]:
    """Parse ``--concurrency 1,4,20`` into positive, de-duplicated levels.

    Also accepts a single integer (``--concurrency 8``), so the Phase 5-3
    invocation keeps working.
    """
    levels: list[int] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            level = int(chunk)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"concurrency level {chunk!r} is not an integer") from error
        if level < 1:
            raise argparse.ArgumentTypeError(f"concurrency level {level} must be >= 1")
        if level not in levels:
            levels.append(level)
    if not levels:
        raise argparse.ArgumentTypeError("at least one concurrency level is required")
    return levels


#: The engine states' ``/health`` embedding-cache modes the bench can verify.
EMBEDDING_CACHE_MODES = ("auto", "on", "off")


def cache_state_matches(requested: str, observed: bool | None) -> bool | None:
    """Whether the server's reported cache state matches ``--embedding-cache``.

    ``None`` means "cannot verify" — the request is ``auto`` or the server did
    not report an ``embedding_cache`` block (old build / health error).
    """
    if requested not in EMBEDDING_CACHE_MODES:
        raise ValueError(f"unknown embedding-cache mode: {requested!r}")
    if requested == "auto" or observed is None:
        return None
    return observed == (requested == "on")


def clone_scenario_matrix(reference: str = "repo_voice_a") -> list[dict]:
    """The four clone-flood cells of the Phase 8 scenario matrix.

    ``{same,distinct} × {cache on, cache off}``. Each cell is one bench run; the
    cache half is a *server* flag, so the four cells span two engine boots
    (``--no-embedding-cache`` for the ``off`` rows).
    """
    return [
        {
            "label": f"phase8-{mode}-cache-{cache}",
            "clone_reference": mode,
            "cache": cache,
            "reference": reference,
        }
        for mode in CLONE_REFERENCE_MODES
        for cache in ("on", "off")
    ]


#: Step label for every engine DEBUG line that reports padded sub-batching.
STEP_LOG_LABELS = {
    "HarmonySpeech encoder": "embed",
    "OpenVoice tone-converter encoder": "embed",
    "OpenVoice tone converter": "tc",
    "OpenVoice V1": "synth",
    "MeloTTS": "synth",
    "HarmonySpeech vocoder": "vocoder",
    "VoiceFixer restorer": "restore",
    "VoiceFixer vocoder": "vocoder",
}

_STEP_LOG_RE = re.compile(
    r"(?P<step>"
    + "|".join(re.escape(name) for name in sorted(STEP_LOG_LABELS, key=len, reverse=True))
    + r"): (?:(?P<rows>\d+) (?:segment|sentence|partial|slice)\(s\) from )?"
    r"(?P<requests>\d+) request\(s\) forwarded in (?P<groups>\d+) padded sub-batch\(es\)"
)
_KITTEN_LANES_RE = re.compile(
    r"KittenTTS: (?P<requests>\d+) request\(s\) synthesised in (?P<lanes>\d+) lane\(s\)"
)


def parse_step_composition(lines) -> dict[str, dict]:
    """Tally per-step sub-batch composition from engine DEBUG log lines.

    One bucket per step label (``embed``, ``tc``, ``synth``, ``vocoder``,
    ``restore``; ``lanes`` for KittenTTS): how many engine steps logged a forward,
    how many padded sub-batches those steps needed in total, the rows they
    covered and the requests they served. ``avg_sub_batches_per_step`` is the
    "embed: 2 sub-batches of ≤8" figure — the number of padded forwards one
    engine step needed.
    """
    steps: dict[str, dict] = {}

    def _bucket(label: str) -> dict:
        return steps.setdefault(
            label, {"steps": 0, "sub_batches": 0, "rows": 0, "requests": 0}
        )

    for line in lines:
        match = _STEP_LOG_RE.search(line)
        if match:
            bucket = _bucket(STEP_LOG_LABELS[match.group("step")])
            bucket["steps"] += 1
            bucket["sub_batches"] += int(match.group("groups"))
            bucket["rows"] += int(match.group("rows") or 0)
            bucket["requests"] += int(match.group("requests"))
            continue
        match = _KITTEN_LANES_RE.search(line)
        if match:
            bucket = _bucket("lanes")
            bucket["steps"] += 1
            bucket["requests"] += int(match.group("requests"))
            bucket["lanes"] = max(int(bucket.get("lanes", 0)), int(match.group("lanes")))

    for bucket in steps.values():
        bucket["avg_sub_batches_per_step"] = (
            round(bucket["sub_batches"] / bucket["steps"], 2) if bucket["steps"] else 0.0
        )
    return steps


class EngineLogTail:
    """Incremental reader for the engine's DEBUG log (the server is another process)."""

    def __init__(self, path: str):
        self.path = path
        self._offset = 0

    def _size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def drain(self) -> list[str]:
        """Return the lines appended since the previous drain (possibly none)."""
        try:
            with open(self.path, encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError:
            return []
        return [line for line in chunk.splitlines() if line.strip()]

    def flush(self) -> None:
        """Skip everything written so far, e.g. before a level's measurement."""
        self._offset = self._size()


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


async def _run_level(
    client, base_url, family, payloads, concurrency: int, total: int, timeout: float, log_tail=None
):
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

    if log_tail is not None:
        # Warmup and previous levels must not pollute this level's composition.
        log_tail.flush()
    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - started
    steps = parse_step_composition(log_tail.drain()) if log_tail is not None else {}
    return LevelResult(
        concurrency=concurrency,
        requests=total,
        failures=failures,
        wall_s=wall,
        latencies_s=latencies,
        steps=steps,
    )


async def _discover_models(client, base_url: str) -> set[str]:
    names: set[str] = set()
    for path in (
        "/v1/audio/speech/models",
        "/v1/audio/transcriptions/models",
        "/v1/audio/convert/models",
        "/v1/embed/models",
    ):
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


async def _discover_cache_state(client, base_url: str) -> bool | None:
    """Read ``embedding_cache.enabled`` from the engine's ``/health`` block."""
    try:
        response = await client.get(base_url + "/health", timeout=30)
        if response.status_code != 200:
            return None
        cache = (response.json() or {}).get("embedding_cache") or {}
        enabled = cache.get("enabled")
        return None if enabled is None else bool(enabled)
    except Exception:  # pragma: no cover - verification is best effort
        return None


async def run_benchmark(args) -> dict:
    import httpx

    from tests.audio_fixtures import reference_wav_b64

    reference_audio = reference_wav_b64(args.reference)
    families = [name.strip() for name in args.families.split(",") if name.strip()]
    unknown = [name for name in families if name not in FAMILY_SPECS]
    if unknown:
        raise SystemExit(f"unknown families: {', '.join(unknown)}")

    levels = list(args.concurrency)
    max_connections, max_keepalive = http_client_limits(levels, args.http_max_connections)
    log_tail = EngineLogTail(args.engine_log) if args.engine_log else None
    cache_requested = getattr(args, "embedding_cache", "auto")
    overtime: dict[str, list[LevelResult]] = {}
    skipped: dict[str, str] = {}

    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
    async with httpx.AsyncClient(limits=limits) as client:
        cache_observed = await _discover_cache_state(client, args.base_url)
        verdict = cache_state_matches(cache_requested, cache_observed)
        if verdict is False:
            expected = "without" if cache_requested == "on" else "with"
            raise SystemExit(
                f"--embedding-cache {cache_requested} does not match the running engine "
                f"(cache is {'on' if cache_observed else 'off'}). Restart the server {expected} "
                f"the cache flag (--no-embedding-cache disables it) and rerun."
            )
        if cache_requested != "auto" and verdict is None:
            print(
                "[bench] the engine did not report an embedding_cache block on /health; "
                "recording the requested state without verification",
                flush=True,
            )
        if cache_observed is not None:
            print(f"[bench] embedding cache: {'on' if cache_observed else 'off'}", flush=True)
        if max_connections < max(levels):
            print(
                f"[bench] client ceiling: {max_connections} connection(s) for c={max(levels)} "
                "— requests queue client-side",
                flush=True,
            )

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
                    client,
                    args.base_url,
                    family,
                    payloads,
                    level,
                    args.requests,
                    args.timeout,
                    log_tail=log_tail,
                )
                results.append(result)
                print(
                    f"[bench] {family} c={level}: {result.throughput:.2f} req/s "
                    f"p50={result.percentile(0.5) * 1000:.0f}ms p95={result.percentile(0.95) * 1000:.0f}ms "
                    f"failures={result.failures}",
                    flush=True,
                )
                for step, stats in sorted(result.steps.items()):
                    print(
                        f"[bench]   {step}: {stats['steps']} step(s), "
                        f"{stats['sub_batches']} sub-batch(es), "
                        f"avg {stats['avg_sub_batches_per_step']}/step",
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
        "concurrency_levels": levels,
        "clone_reference": getattr(args, "clone_reference", "same"),
        "embedding_cache": {"requested": cache_requested, "observed": cache_observed},
        "http_client": {
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive,
        },
        "engine_log": args.engine_log,
        "families": {
            family: [result.as_dict() for result in results] for family, results in overtime.items()
        },
        "skipped": skipped,
    }


def _markdown(report: dict) -> str:
    env = report["env"]
    cache = report.get("embedding_cache") or {}
    cache_state = cache.get("observed")
    if cache_state is None:
        cache_text = f"{cache.get('requested') or 'auto'} (not reported by the engine)"
    else:
        cache_text = "on" if cache_state else "off"
        if cache.get("requested") not in (None, "auto"):
            cache_text += f" (verified against --embedding-cache {cache['requested']})"
    pool = report.get("http_client") or {}
    levels = report.get("concurrency_levels") or []
    lines = [
        f"### {report['label']} — {report['timestamp_utc']}",
        "",
        f"* commit `{report.get('commit') or 'dirty'}` · torch {env.get('torch')} · "
        f"{env.get('gpu') or 'CPU'} · driver {env.get('driver') or 'n/a'} · "
        f"CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices') or 'all'}",
        f"* config: {report.get('config') or 'n/a'}",
        f"* clone reference: {report.get('clone_reference') or 'same'} · embedding cache: {cache_text}",
        f"* concurrency: {', '.join(str(level) for level in levels) or 'n/a'} · "
        f"HTTP client pool: {pool.get('max_connections', 'n/a')} connection(s)",
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

    composition = [
        (family, level["concurrency"], step, stats)
        for family, levels in report["families"].items()
        for level in levels
        for step, stats in sorted((level.get("steps") or {}).items())
    ]
    if composition:
        lines += [
            "",
            "Per-step sub-batch composition (engine DEBUG log):",
            "",
            "| Family | Concurrency | Step | Steps logged | Sub-batches | avg/step | Rows | Requests | Lanes |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for family, concurrency, step, stats in composition:
            lines.append(
                f"| {family} | {concurrency} | {step} | {stats['steps']} | {stats['sub_batches']} | "
                f"{stats['avg_sub_batches_per_step']} | {stats['rows']} | {stats['requests']} | "
                f"{stats.get('lanes', '—')} |"
            )
    if report.get("skipped"):
        lines.append("")
        for family, reason in report["skipped"].items():
            lines.append(f"* {family}: skipped — {reason}")
    lines.append("")
    return "\n".join(lines)


def scenario_recipe(args) -> str:
    """Shell recipe for the Phase 8 clone-flood matrix (two server boots)."""
    port = args.base_url.rsplit(":", 1)[-1]
    server_config = args.server_config
    common = (
        f"python scripts/bench_throughput.py --families openvoice_v2 "
        f"--concurrency {','.join(str(level) for level in args.concurrency)} "
        f"--requests {args.requests} --reference {args.reference}"
    )
    engine_log = args.engine_log or "out/bench/engine.log"
    log_flag = f" --engine-log {engine_log}"
    lines = [
        "# Phase 8 clone-flood matrix: {same,distinct} x {cache on, cache off}.",
        "# The cache toggle is a SERVER flag: restart the engine between the two",
        "# halves. Boxes must run one at a time on an idle GPU.",
        "",
        "# terminal 1 - cache ON (the shipped default)",
        "CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=DEBUG \\",
        f"    python -m harmonyspeech.endpoints.openai.api_server --config-file-path {server_config} \\",
        f"    --port {port} > {engine_log} 2>&1",
        "",
        "# terminal 2 - cache-on rows",
    ]
    for scenario in clone_scenario_matrix(args.reference):
        if scenario["cache"] != "on":
            continue
        lines.append(
            f"{common} --clone-reference {scenario['clone_reference']} --embedding-cache on "
            f"--label {scenario['label']}{log_flag} --md-out docs/benchmarks/throughput.md"
        )
    lines += [
        "",
        "# restart terminal 1 with --no-embedding-cache, then run the cache-off rows:",
    ]
    for scenario in clone_scenario_matrix(args.reference):
        if scenario["cache"] != "off":
            continue
        lines.append(
            f"{common} --clone-reference {scenario['clone_reference']} --embedding-cache off "
            f"--label {scenario['label']}{log_flag} --md-out docs/benchmarks/throughput.md"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Throughput benchmark against a running engine (Phase 5-3 / 7-3 / 8-6)"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:2242")
    parser.add_argument(
        "--families",
        default="melo,openvoice_v2,openvoice_v1,harmonyspeech,voicefixer,kitten,chatterbox,whisper",
    )
    parser.add_argument(
        "--concurrency",
        type=parse_concurrency_levels,
        default=parse_concurrency_levels("1,2,4,8"),
        help="comma-separated concurrency levels, e.g. 1,4,20 (a single int still works)",
    )
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
        "--embedding-cache",
        choices=EMBEDDING_CACHE_MODES,
        default="auto",
        help=(
            "server-side cache state this run assumes: 'on'/'off' is verified against /health and the run "
            "aborts on a mismatch (restart the engine with --no-embedding-cache for 'off'); 'auto' trusts "
            "whatever the engine reports"
        ),
    )
    parser.add_argument(
        "--engine-log",
        default=None,
        help=(
            "path of the engine's DEBUG log; per-level lines are parsed for the per-step sub-batch "
            "composition (run the server with HARMONYSPEECH_LOG_LEVEL=DEBUG)"
        ),
    )
    parser.add_argument(
        "--http-max-connections",
        type=int,
        default=None,
        help="cap the load client's connection pool (default: the highest concurrency level)",
    )
    parser.add_argument(
        "--server-config",
        default="out/bench/config-phase8.yml",
        help="config path printed by --print-scenarios for the engine boots",
    )
    parser.add_argument(
        "--print-scenarios",
        action="store_true",
        help="print the Phase 8 clone-flood recipe (two server boots x two reference modes) and exit",
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

    if args.print_scenarios:
        print(scenario_recipe(args))
        return 0

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
