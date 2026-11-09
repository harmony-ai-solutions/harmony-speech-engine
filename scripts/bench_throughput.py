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

Text fidelity (``--scenario longform --fidelity``, added 2026-11-09)
--------------------------------------------------------------------
The throughput matrix measures duration, not whether the audio says the text —
the gap that let Chatterbox Turbo's word-dropping ship unnoticed. Fidelity mode
renders each family's fixture once per variant (default ON at the server's
per-model chunk budget; ``--longform-split both`` also scores OFF), saves the
WAVs under ``out/bench/fidelity-12/``, transcribes them with faster-whisper
``large-v3-turbo`` (``--gpu`` or CPU int8, language ``en``, beam 5) and scores
word edit distance plus words heard / reference. The transcription noise floor
on this fixture is ~2–6 edit distance ("Karina/Carina", "weaves/leaves").
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
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

#: The default family list for the ordinary throughput matrix; ``--scenario
#: longform`` substitutes :data:`LONGFORM_FAMILIES` when the user leaves it at
#: this default.
DEFAULT_FAMILIES = "melo,openvoice_v2,openvoice_v1,harmonyspeech,voicefixer,kitten,chatterbox,whisper"

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
    "chatterbox_turbo": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "chatterbox_turbo",
        "payload": {
            "mode": "single_speaker_tts",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
        },
    },
    "chatterbox_multilingual": {
        # Phase 12-4: the loop-class control for the long-form matrix. It is a
        # real TTS family but keeps `max_batch_size: 1`, so chunk sub-requests
        # queue instead of co-batching — the honest contrast to the AR/padded
        # families.
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "chatterbox_multilingual",
        "payload": {
            "mode": "single_speaker_tts",
            "language": "en",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
        },
    },
    # Phase 11-2: clone-flood variants of the two AR families. Same model/path as
    # the entries above, but every request carries a reference clip in
    # `voice_cloning` mode, so the clone pipeline (embed step + AR decode +
    # S3Gen vocode) is what the concurrency sweep measures. Additive: the
    # existing bundled-voice families and their published tables are unchanged.
    "chatterbox_clone": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "chatterbox",
        "payload": {
            "mode": "voice_cloning",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
            "input_audio": "__REFERENCE_AUDIO__",
        },
    },
    "chatterbox_turbo_clone": {
        "path": "/v1/audio/speech",
        "kind": "tts",
        "model": "chatterbox_turbo",
        "payload": {
            "mode": "voice_cloning",
            "input": "The quick brown fox jumps over the lazy dog near the river bank.",
            "input_audio": "__REFERENCE_AUDIO__",
        },
    },
    "chatterbox_vc": {
        # Phase 11-1: VC has no text; both fields take the reference clip. The
        # source is the clip to convert, the target the voice to convert to
        # (the same clip here — the throughput shape, not a quality test).
        "path": "/v1/voice/convert",
        "kind": "vc",
        "model": "chatterbox_vc",
        "payload": {
            "source_audio": "__REFERENCE_AUDIO__",
            "target_audio": "__REFERENCE_AUDIO__",
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
    #: ``/metrics`` AR pool + decode samples taken during this level (Phase 9-6).
    ar_trace: list[dict] = field(default_factory=list)
    #: Duration of the first successful response at this level (Phase 12-4).
    #: ``None`` when the response carried no decodable audio.
    audio_seconds: float | None = None

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
            "audio_seconds": self.audio_seconds,
            "steps": self.steps,
            "ar_trace": self.ar_trace,
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


# ---------------------------------------------------------------------------
# Phase 10-5 — Whisper exhaustion / fairness / budget-trip scenarios
# ---------------------------------------------------------------------------

#: Response formats the heterogeneous flood rotates through (Phase 10-4 wire
#: surface). ``json`` is the byte-identical default; the other three are text.
WHISPER_RESPONSE_FORMATS: tuple[str, ...] = ("json", "text", "srt", "vtt")

#: Source-language hints the flood rotates through. The reference audio is
#: reused (no external assets); the point is per-request prompt/tokenizer
#: diversity across a batch, not translation quality.
WHISPER_LANGUAGES: tuple[str, ...] = ("en", "de", "fr", "es", "ja")

#: Owner-approved tight caps for the exhaustion scenario (2026-11-07): M=2
#: workers, window batch=4, admission/queue depth=16. The engine must be booted
#: with these values; the script records them and refuses to claim a run it
#: cannot verify when ``/health`` exposes the configured budget.
WHISPER_FLOOD_WORKERS = 2
WHISPER_FLOOD_WINDOW_BATCH = 4
WHISPER_FLOOD_QUEUE = 16
WHISPER_FLOOD_REQUESTS = 100
WHISPER_FLOOD_MAX_SECONDS = 480.0  # 8 minutes

#: The fairness proof's pinned long-first request (owner: "one ≥10-min file
#: first, then short clips").
WHISPER_FAIRNESS_LONG_SECONDS = 600.0


def whisper_flood_durations(count: int = WHISPER_FLOOD_REQUESTS, max_seconds: float = WHISPER_FLOOD_MAX_SECONDS) -> list[float]:
    """Deterministic 2 s–``max_seconds`` duration spread for the flood.

    Most requests are short (the workload Whisper serves most), a middle band
    exercises the 30 s window boundary, and a long tail (including one
    ``max_seconds`` request) exercises multi-window collection and stitching.
    Deterministic so a rerun measures the same load shape.
    """
    count = max(1, int(count))
    max_seconds = max(2.0, float(max_seconds))
    durations: list[float] = []
    # 70 % short (2–25 s), 20 % medium (30–90 s), 10 % long (120–max_seconds).
    short_count = max(1, round(count * 0.70))
    medium_count = max(1, round(count * 0.20))
    long_count = max(1, count - short_count - medium_count)
    for index in range(short_count):
        span = 25.0 - 2.0
        durations.append(2.0 + span * (index / max(1, short_count - 1)))
    for index in range(medium_count):
        span = 90.0 - 30.0
        durations.append(30.0 + span * (index / max(1, medium_count - 1)))
    for index in range(long_count):
        span = max_seconds - 120.0
        durations.append(120.0 + span * (index / max(1, long_count - 1)))
    # Exactly `count` rows; the last one is the 8-minute multi-window case.
    durations = durations[:count]
    durations[-1] = max_seconds
    return [round(value, 3) for value in durations]


def whisper_flood_scenario(
    count: int = WHISPER_FLOOD_REQUESTS,
    max_seconds: float = WHISPER_FLOOD_MAX_SECONDS,
) -> list[dict]:
    """The 100-request heterogeneous exhaustion scenario (owner, 2026-11-07).

    Mixed durations (2 s–8 min), mixed tasks (transcription + translation),
    mixed source-language hints, and all four response formats. Each row is one
    HTTP request; the engine is expected to run it against deliberately tight
    caps (M=2 workers, window batch=4, queue 16).
    """
    specs = []
    for index, seconds in enumerate(whisper_flood_durations(count, max_seconds)):
        translate = index % 7 == 0
        specs.append(
            {
                "index": index,
                "duration_s": seconds,
                "task": "translate" if translate else "transcribe",
                "path": "/v1/audio/translations" if translate else "/v1/audio/transcriptions",
                "response_format": WHISPER_RESPONSE_FORMATS[index % len(WHISPER_RESPONSE_FORMATS)],
                "language": WHISPER_LANGUAGES[index % len(WHISPER_LANGUAGES)],
                "kind": "flood",
            }
        )
    return specs


def whisper_fairness_scenario(
    long_seconds: float = WHISPER_FAIRNESS_LONG_SECONDS,
    short_count: int = 20,
    short_seconds: float = 3.0,
) -> list[dict]:
    """The head-of-line fairness proof: one ≥10-min file **first**, then shorts.

    Short-clip p50/p95 under this queue is the pile-up answer; the sequential
    baseline is the Phase 5-3 table in ``docs/benchmarks/throughput.md``.
    """
    specs = [
        {
            "index": 0,
            "duration_s": float(long_seconds),
            "task": "transcribe",
            "path": "/v1/audio/transcriptions",
            "response_format": "json",
            "language": None,
            "kind": "long",
        }
    ]
    for index in range(max(1, int(short_count))):
        specs.append(
            {
                "index": index + 1,
                "duration_s": float(short_seconds),
                "task": "transcribe",
                "path": "/v1/audio/transcriptions",
                "response_format": WHISPER_RESPONSE_FORMATS[index % len(WHISPER_RESPONSE_FORMATS)],
                "language": WHISPER_LANGUAGES[index % len(WHISPER_LANGUAGES)],
                "kind": "short",
            }
        )
    return specs


def duration_audio_b64(reference_audio_b64: str, seconds: float, cache: dict | None = None) -> str:
    """Return base64 WAV of exactly ``seconds`` built by tiling the reference clip.

    No external assets are needed: the decoded reference speech is repeated (and
    trimmed) to the requested duration, so VAD/windowing see real speech at every
    length. Results are cached per (rounded) duration — the flood reuses many
    durations across its 100 requests.
    """
    import base64

    import numpy as np

    from harmonyspeech.common import audio_codec

    seconds = max(0.01, float(seconds))
    key = round(seconds, 3)
    cache = cache if cache is not None else {}
    if key in cache:
        return cache[key]

    raw = base64.b64decode(reference_audio_b64)
    waveform, sample_rate = audio_codec.decode_audio(raw)
    target = max(1, int(round(seconds * sample_rate)))
    if len(waveform) == 0:
        raise ValueError("reference audio decoded to zero samples")
    if len(waveform) >= target:
        trimmed = waveform[:target]
    else:
        repeats = int(np.ceil(target / len(waveform)))
        trimmed = np.tile(waveform, repeats)[:target]
    encoded = audio_codec.encode_audio(trimmed, sample_rate, "wav")
    value = base64.b64encode(encoded).decode("ascii")
    cache[key] = value
    return value


def build_whisper_payload(spec: dict, audio_b64: str, model: str) -> dict:
    """Build one Whisper request payload from a scenario row."""
    payload = {
        "model": model,
        "input_audio": audio_b64,
        "response_format": spec["response_format"],
    }
    if spec.get("language"):
        payload["language"] = spec["language"]
    return payload


def _whisper_response_ok(spec: dict, response) -> tuple[bool, str | None]:
    """Whether one flood response is a valid, non-lossy answer for its format."""
    if response.status_code != 200:
        return False, f"http_{response.status_code}"
    fmt = spec["response_format"]
    if fmt == "json":
        try:
            body = response.json()
        except Exception as error:  # noqa: BLE001 - malformed JSON is a format error
            return False, f"invalid_json: {error}"
        if not isinstance(body, dict) or "text" not in body:
            return False, "missing_text_field"
        return True, None
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("text/plain"):
        return False, f"wrong_content_type: {content_type}"
    # An empty document is valid (silence); a subtitle document that is not
    # empty must carry its format marker.
    text = response.text or ""
    if fmt == "srt" and text.strip() and "-->" not in text:
        return False, "invalid_srt"
    if fmt == "vtt" and text.strip() and not text.lstrip().startswith("WEBVTT"):
        return False, "invalid_vtt"
    return True, None


_WHISPER_GAUGE_RE = re.compile(
    r'^hse_whisper_(workers|in_flight|lane_short_depth|lane_long_depth|batch_fallbacks_total)'
    r'\{model="([^"]*)"\}\s+([0-9.eE+-]+)\s*$'
)
_WHISPER_CT2_RE = re.compile(
    r'^hse_whisper_ct2_(queued|active)_batches\{model="([^"]*)"\}\s+([0-9.eE+-]+)\s*$'
)
_WHISPER_QUEUE_WAIT_RE = re.compile(
    r'^hse_whisper_queue_wait_seconds_(sum|count)\{model="([^"]*)"\}\s+([0-9.eE+-]+)\s*$'
)


def parse_whisper_metrics(text: str) -> dict:
    """Parse the Whisper lane gauges and device-level VRAM from ``/metrics``.

    Returns ``{"models": {name: {...}}, "device": {...}}``. Pure so the load
    shape (and the lane-depth trace) is unit-testable without a server.
    """
    models: dict[str, dict] = {}
    device: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("hse_vram_device_used_bytes "):
            with contextlib.suppress(IndexError, ValueError):
                device["device_used_bytes"] = int(float(line.split()[1]))
            continue
        if line.startswith("hse_vram_headroom_bytes "):
            with contextlib.suppress(IndexError, ValueError):
                device["headroom_bytes"] = int(float(line.split()[1]))
            continue
        match = _WHISPER_GAUGE_RE.match(line)
        if match:
            key, model, value = match.groups()
            models.setdefault(model, {})[key] = int(float(value))
            continue
        match = _WHISPER_CT2_RE.match(line)
        if match:
            key, model, value = match.groups()
            entry = models.setdefault(model, {})
            entry[f"ct2_{key}"] = int(float(value))
            continue
        match = _WHISPER_QUEUE_WAIT_RE.match(line)
        if match:
            key, model, value = match.groups()
            models.setdefault(model, {})[f"queue_wait_{key}"] = float(value)
    return {"models": models, "device": device}


class MetricsSampler:
    """Sample ``/metrics`` while a flood runs (lane depths over time)."""

    def __init__(self, client, base_url: str, interval: float = 0.5) -> None:
        self.client = client
        self.base_url = base_url
        self.interval = max(0.05, float(interval))
        self.samples: list[dict] = []
        self._task = None
        self._started = 0.0

    async def _sample_once(self) -> None:
        try:
            response = await self.client.get(self.base_url + "/metrics", timeout=10)
        except Exception:  # pragma: no cover - sampling is best effort
            return
        if response.status_code != 200:
            return
        parsed = parse_whisper_metrics(response.text)
        parsed["t_s"] = round(time.monotonic() - self._started, 3)
        self.samples.append(parsed)

    async def _loop(self) -> None:
        while True:
            await self._sample_once()
            await asyncio.sleep(self.interval)

    async def start(self) -> None:
        self._started = time.monotonic()
        await self._sample_once()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> list[dict]:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._sample_once()
        return self.samples


async def _run_whisper_scenario(
    client,
    base_url: str,
    specs: list[dict],
    payloads: list[dict],
    concurrency: int,
    timeout: float,
    sampler: MetricsSampler | None = None,
) -> tuple[list[dict], float, list[dict]]:
    """Fire every request at ``concurrency`` clients, recording per-request traces.

    Every submitted request produces exactly one record — a client exception is
    recorded as a failed request, never dropped — so ``len(records) == len(specs)``
    is the request-loss check.
    """
    queue: asyncio.Queue = asyncio.Queue()
    for index in range(len(specs)):
        queue.put_nowait(index)
    records: list[dict] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            spec = specs[index]
            payload = payloads[index]
            started = time.perf_counter()
            ok = False
            status = None
            error = None
            try:
                response = await client.post(
                    base_url + spec["path"], json=payload, timeout=timeout
                )
                elapsed = time.perf_counter() - started
                status = response.status_code
                ok, error = _whisper_response_ok(spec, response)
            except Exception as exc:  # noqa: BLE001 - a client timeout/failure is data
                elapsed = time.perf_counter() - started
                error = f"{type(exc).__name__}: {exc}"
            async with lock:
                records.append(
                    {
                        "index": index,
                        "kind": spec.get("kind", "flood"),
                        "duration_s": spec["duration_s"],
                        "task": spec["task"],
                        "response_format": spec["response_format"],
                        "language": spec.get("language"),
                        "status": status,
                        "wall_ms": round(elapsed * 1000, 1),
                        "ok": ok,
                        "error": error,
                    }
                )

    if sampler is not None:
        await sampler.start()
    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(max(1, int(concurrency)))))
    wall = time.perf_counter() - started
    samples = await sampler.stop() if sampler is not None else []
    return records, wall, samples


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize_whisper_scenario(specs: list[dict], records: list[dict], wall_s: float, samples: list[dict]) -> dict:
    """Aggregate the exhaustion/fairness evidence the owner asked for."""
    completed = [record for record in records if record["ok"]]
    failed = [record for record in records if not record["ok"]]
    timeouts = [
        record
        for record in failed
        if "timeout" in (record.get("error") or "").lower()
        or record.get("error") == "http_504"
    ]
    format_errors = [
        record
        for record in failed
        if record.get("error")
        and (
            record["error"].startswith("invalid_")
            or record["error"].startswith("wrong_content_type")
            or record["error"] == "missing_text_field"
        )
    ]
    latencies = [record["wall_ms"] / 1000.0 for record in completed]
    short_latencies = [
        record["wall_ms"] / 1000.0 for record in completed if record.get("kind") == "short"
    ]

    lane_trace = []
    queue_wait_start: dict[str, tuple[float, int]] = {}
    queue_wait_end: dict[str, tuple[float, int]] = {}
    max_device_used = 0
    for sample in samples:
        entry = {"t_s": sample.get("t_s"), "models": {}}
        for model, detail in (sample.get("models") or {}).items():
            entry["models"][model] = {
                "in_flight": detail.get("in_flight", 0),
                "lane_short_depth": detail.get("lane_short_depth", 0),
                "lane_long_depth": detail.get("lane_long_depth", 0),
                "ct2_queued": detail.get("ct2_queued", 0),
                "ct2_active": detail.get("ct2_active", 0),
            }
            pair = (
                float(detail.get("queue_wait_sum", 0.0)),
                int(detail.get("queue_wait_count", 0)),
            )
            queue_wait_start.setdefault(model, pair)
            queue_wait_end[model] = pair
        device_used = (sample.get("device") or {}).get("device_used_bytes")
        if device_used is not None:
            max_device_used = max(max_device_used, int(device_used))
        lane_trace.append(entry)

    # The engine's histogram is cumulative since boot; report this run's delta.
    queue_wait = {
        "sum_s": round(
            sum(
                max(0.0, queue_wait_end[model][0] - queue_wait_start[model][0])
                for model in queue_wait_end
            ),
            3,
        ),
        "count": sum(
            max(0, queue_wait_end[model][1] - queue_wait_start[model][1])
            for model in queue_wait_end
        ),
    }

    return {
        "requests": len(specs),
        "records": len(records),
        "request_loss": len(specs) - len(records),
        "completed": len(completed),
        "failures": len(failed),
        "timeouts": len(timeouts),
        "format_errors": len(format_errors),
        "wall_s": round(wall_s, 3),
        "throughput_rps": round(len(completed) / wall_s, 2) if wall_s > 0 else 0.0,
        "p50_ms": round(_percentile(latencies, 0.50) * 1000, 1),
        "p95_ms": round(_percentile(latencies, 0.95) * 1000, 1),
        "short_p50_ms": round(_percentile(short_latencies, 0.50) * 1000, 1) if short_latencies else None,
        "short_p95_ms": round(_percentile(short_latencies, 0.95) * 1000, 1) if short_latencies else None,
        "queue_wait_sum_s": round(queue_wait["sum_s"], 3),
        "queue_wait_count": queue_wait["count"],
        "max_device_used_bytes": max_device_used or None,
        "lane_trace": lane_trace,
        "records_detail": records,
    }


def whisper_scenario_matrix(
    concurrencies: tuple[int, ...] = (4, 8, 20),
    turbo_workers: tuple[int, ...] = (1, 2, 4),
    tiny_workers: tuple[int, ...] = (1, 4, 6),
    window_batches: tuple[int, ...] = (4, 8),
) -> list[dict]:
    """The Phase 10-5 benchmark cells (config-side, so the script prints them).

    Each cell is one server boot + one bench run; ``lanes`` means
    ``batched_inference: false`` (per-request lane pool), ``lanes+windows``
    means ``batched_inference: true``.
    """
    cells = []
    for concurrency in concurrencies:
        for mode, flag in (("lanes", False), ("lanes+windows", True)):
            cells.append(
                {
                    "label": f"phase10-{mode.replace('+', '-')}-c{concurrency}",
                    "scenario": "throughput",
                    "concurrency": concurrency,
                    "batched_inference": flag,
                    "config_note": f"Whisper {mode} at c={concurrency}",
                }
            )
    cells.append(
        {
            "label": "phase10-c200",
            "scenario": "throughput",
            "concurrency": 200,
            "batched_inference": True,
            "config_note": "Whisper lanes+windows, 200-connection observation",
        }
    )
    cells.append(
        {
            "label": "phase10-fairness",
            "scenario": "whisper-fairness",
            "concurrency": 21,
            "batched_inference": True,
            "config_note": "one 10-min file first, then 20 short clips",
        }
    )
    for workers in turbo_workers:
        cells.append(
            {
                "label": f"phase10-turbo-m{workers}",
                "scenario": "throughput",
                "concurrency": 8,
                "batched_inference": False,
                "config_note": f"large-v3-turbo num_workers={workers}",
            }
        )
    for workers in tiny_workers:
        cells.append(
            {
                "label": f"phase10-tiny-m{workers}",
                "scenario": "throughput",
                "concurrency": 8,
                "batched_inference": False,
                "config_note": f"tiny num_workers={workers}",
            }
        )
    for batch in window_batches:
        cells.append(
            {
                "label": f"phase10-window-b{batch}",
                "scenario": "throughput",
                "concurrency": 8,
                "batched_inference": True,
                "config_note": f"window batch cap={batch}",
            }
        )
    cells.append(
        {
            "label": "phase10-exhaustion",
            "scenario": "whisper-flood",
            "concurrency": WHISPER_FLOOD_REQUESTS,
            "batched_inference": True,
            "config_note": "M=2, window batch=4, queue 16, 100 heterogeneous requests",
        }
    )
    cells.append(
        {
            "label": "phase10-budget-trip",
            "scenario": "whisper-budget-trip",
            "concurrency": WHISPER_FLOOD_REQUESTS,
            "batched_inference": True,
            "config_note": "whisper_vram_budget_bytes below demand; admission scales",
        }
    )
    return cells



# ---------------------------------------------------------------------------
# Phase 9-6 — Chatterbox AR decode scenarios (three-way + exhaustion)
# ---------------------------------------------------------------------------

#: Mixed text lengths for the AR flood. Long and short rows decode for different
#: numbers of tokens, so rows drop out of a batch at different steps (the
#: "mixed-length rows" sanity check the owner asked for).
CHATTERBOX_TEXTS: tuple[str, ...] = (
    "Hi.",
    "The quick brown fox jumps over the lazy dog.",
    "Yes.",
    "This is a deliberately longer sentence so that the autoregressive decoder emits a different number of speech tokens than the short rows around it.",
    "Okay, thanks.",
)

#: Owner-approved tight-pool exhaustion shape (2026-11-07): 100 distinct-voice
#: requests against a pool sized for ~8 slots. The pool budget is set on the
#: engine boot (`ar_inference.kv_cache_budget_bytes`); the script records it.
CHATTERBOX_FLOOD_REQUESTS = 100
CHATTERBOX_FLOOD_POOL_SLOTS = 8

_AR_CACHE_RE = re.compile(
    r"^hse_ar_cache_(enabled|pool_bytes|allocated_bytes|slots|slots_active|slots_free|grants_total|refusals_total)\s+"
    r"([0-9.eE+-]+)\s*$"
)
_AR_DECODE_RE = re.compile(
    r"^hse_ar_decode_(active_rows|slots|slots_active|slots_free|tokens_total|steps_total|tokens_per_step)"
    r'\{model="([^"]*)"\}\s+([0-9.eE+-]+)\s*$'
)


def parse_ar_metrics(text: str) -> dict:
    """Parse the Phase 9 ``hse_ar_cache_*`` and ``hse_ar_decode_*`` series.

    Returns ``{"cache": {...}, "models": {name: {...}}}``. Pure, so the trace
    shape is unit-testable without a server or a pool.
    """
    cache: dict[str, float] = {}
    models: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        match = _AR_CACHE_RE.match(line)
        if match:
            key, value = match.groups()
            cache[key] = float(value)
            continue
        match = _AR_DECODE_RE.match(line)
        if match:
            key, model, value = match.groups()
            models.setdefault(model, {})[key] = float(value)
    return {"cache": cache, "models": models}


class ARMetricsSampler:
    """Sample ``/metrics`` AR pool/decode gauges while a Chatterbox run executes."""

    def __init__(self, client, base_url: str, interval: float = 0.5) -> None:
        self.client = client
        self.base_url = base_url
        self.interval = max(0.05, float(interval))
        self.samples: list[dict] = []
        self._task = None
        self._started = 0.0

    async def _sample_once(self) -> None:
        try:
            response = await self.client.get(self.base_url + "/metrics", timeout=10)
        except Exception:  # pragma: no cover - sampling is best effort
            return
        if response.status_code != 200:
            return
        parsed = parse_ar_metrics(response.text)
        parsed["t_s"] = round(time.monotonic() - self._started, 3)
        self.samples.append(parsed)

    async def _loop(self) -> None:
        while True:
            await self._sample_once()
            await asyncio.sleep(self.interval)

    async def start(self) -> None:
        self._started = time.monotonic()
        await self._sample_once()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> list[dict]:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._sample_once()
        return self.samples


def chatterbox_flood_scenario(count: int = CHATTERBOX_FLOOD_REQUESTS) -> list[dict]:
    """The 100-request distinct-voice exhaustion shape.

    Distinct voices are produced by ``--clone-reference distinct`` (each request
    carries a different reference clip, so the embedding cache cannot help); the
    texts rotate through :data:`CHATTERBOX_TEXTS` so rows are mixed-length.
    """
    count = max(1, int(count))
    return [
        {
            "index": index,
            "path": "/v1/audio/speech",
            "kind": "tts",
            "text": CHATTERBOX_TEXTS[index % len(CHATTERBOX_TEXTS)],
            "distinct_voice": True,
        }
        for index in range(count)
    ]


def build_chatterbox_payload(spec: dict, model: str, reference_audio_b64: str | None = None) -> dict:
    """One Chatterbox TTS payload from a flood row (reference audio optional).

    A reference clip is only valid in ``voice_cloning`` mode; the engine rejects
    ``single_speaker_tts`` + ``input_audio`` with HTTP 400, which made every
    distinct-voice flood row fail before it reached the AR pool (fixed
    2026-11-08 while filling the 9-6 exhaustion table).
    """
    mode = "voice_cloning" if reference_audio_b64 else "single_speaker_tts"
    payload = {"model": model, "mode": mode, "input": spec["text"]}
    if reference_audio_b64:
        payload["input_audio"] = reference_audio_b64
    return payload


def _tts_response_ok(response) -> tuple[bool, str | None]:
    """Whether a Chatterbox TTS response carries audio (and is not an error)."""
    if response.status_code != 200:
        return False, f"http_{response.status_code}"
    try:
        body = response.json()
    except Exception as error:  # noqa: BLE001 - malformed JSON is a format error
        return False, f"invalid_json: {error}"
    if not isinstance(body, dict) or not (body.get("data") or body.get("text")):
        return False, "missing_audio"
    return True, None


async def _run_ar_scenario(
    client,
    base_url: str,
    specs: list[dict],
    payloads: list[dict],
    concurrency: int,
    timeout: float,
    sampler: ARMetricsSampler | None = None,
) -> tuple[list[dict], float, list[dict]]:
    """Fire every AR request at ``concurrency`` clients, recording each outcome."""
    queue: asyncio.Queue = asyncio.Queue()
    for index in range(len(specs)):
        queue.put_nowait(index)
    records: list[dict] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            spec = specs[index]
            started = time.perf_counter()
            ok = False
            status = None
            error = None
            try:
                response = await client.post(
                    base_url + spec["path"], json=payloads[index], timeout=timeout
                )
                elapsed = time.perf_counter() - started
                status = response.status_code
                ok, error = _tts_response_ok(response)
            except Exception as exc:  # noqa: BLE001 - a client timeout/failure is data
                elapsed = time.perf_counter() - started
                error = f"{type(exc).__name__}: {exc}"
            async with lock:
                records.append(
                    {
                        "index": index,
                        "kind": spec.get("kind", "tts"),
                        "text": spec.get("text"),
                        "status": status,
                        "wall_ms": round(elapsed * 1000, 1),
                        "ok": ok,
                        "error": error,
                    }
                )

    if sampler is not None:
        await sampler.start()
    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(max(1, int(concurrency)))))
    wall = time.perf_counter() - started
    samples = await sampler.stop() if sampler is not None else []
    return records, wall, samples


def summarize_ar_scenario(specs: list[dict], records: list[dict], wall_s: float, samples: list[dict]) -> dict:
    """Aggregate loss, latency and the AR pool/decode trace for one run."""
    completed = [record for record in records if record["ok"]]
    failed = [record for record in records if not record["ok"]]
    timeouts = [record for record in failed if "timeout" in (record.get("error") or "").lower()]
    latencies = [record["wall_ms"] / 1000.0 for record in completed]

    first_cache: dict = {}
    last_cache: dict = {}
    max_slots_active = 0
    max_tokens_per_step = 0.0
    for sample in samples:
        cache = sample.get("cache") or {}
        if cache:
            if not first_cache:
                first_cache = dict(cache)
            last_cache = dict(cache)
            max_slots_active = max(max_slots_active, int(cache.get("slots_active", 0) or 0))
        for detail in (sample.get("models") or {}).values():
            max_tokens_per_step = max(max_tokens_per_step, float(detail.get("tokens_per_step", 0.0) or 0.0))

    def delta(key: str) -> int:
        return max(0, int(last_cache.get(key, 0) or 0) - int(first_cache.get(key, 0) or 0))

    return {
        "requests": len(specs),
        "records": len(records),
        "request_loss": len(specs) - len(records),
        "completed": len(completed),
        "failures": len(failed),
        "timeouts": len(timeouts),
        "wall_s": round(wall_s, 3),
        "throughput_rps": round(len(completed) / wall_s, 2) if wall_s > 0 else 0.0,
        "p50_ms": round(_percentile(latencies, 0.50) * 1000, 1),
        "p95_ms": round(_percentile(latencies, 0.95) * 1000, 1),
        "max_slots_active": max_slots_active,
        "grants_delta": delta("grants_total"),
        "refusals_delta": delta("refusals_total"),
        "max_tokens_per_step": round(max_tokens_per_step, 3),
        "ar_trace": samples,
        "records_detail": records,
    }


def ar_scenario_matrix(
    concurrencies: tuple[int, ...] = (4, 8, 20),
    families: tuple[str, ...] = ("chatterbox", "chatterbox_turbo"),
) -> list[dict]:
    """The Phase 9-6 benchmark cells (config-side, so the script prints them).

    Three-way comparison per family and concurrency: sequential loop
    (``batched_inference: false``), AR-0 (the cleaned loop: same sequential path
    with the Phase 9-1 seam, measured against the parent commit) and the full AR
    runner (the shipped width). Then the cache-composition matrix and the
    tight-pool exhaustion trace.
    """
    cells: list[dict] = []
    for family in families:
        for concurrency in concurrencies:
            for mode, flag in (("sequential", False), ("ar0", False), ("ar-runner", None)):
                cells.append(
                    {
                        "label": f"phase9-{family}-{mode}-c{concurrency}",
                        "scenario": "chatterbox-ar",
                        "families": family,
                        "concurrency": concurrency,
                        "batched_inference": flag,
                        "config_note": f"{family} {mode} at c={concurrency}",
                    }
                )
    for mode in CLONE_REFERENCE_MODES:
        for cache in ("on", "off"):
            cells.append(
                {
                    "label": f"phase9-cache-{mode}-{cache}",
                    "scenario": "chatterbox-ar",
                    "families": "chatterbox",
                    "concurrency": 8,
                    "batched_inference": None,
                    "clone_reference": mode,
                    "embedding_cache": cache,
                    "config_note": f"Chatterbox AR runner, {mode} voice, cache {cache}",
                }
            )
    cells.append(
        {
            "label": "phase9-exhaustion",
            "scenario": "chatterbox-exhaustion",
            "families": "chatterbox",
            "concurrency": CHATTERBOX_FLOOD_REQUESTS,
            "batched_inference": None,
            "config_note": (
                f"{CHATTERBOX_FLOOD_REQUESTS} distinct-voice requests, "
                f"pool sized for ~{CHATTERBOX_FLOOD_POOL_SLOTS} slots"
            ),
        }
    )
    return cells


# ---------------------------------------------------------------------------
# Phase 12-4 — long-form scenarios (chunking on/off, single request + flood)
# ---------------------------------------------------------------------------

#: Historical seed of the Phase 12-4 generator. The fixture is now a fixed
#: literal (see :data:`LONGFORM_FIXTURE_TEXT`), so the seed no longer influences
#: the text; it stays in reports for continuity with the earlier runs.
LONGFORM_SEED = 20261109

#: Global character budget used when a long-form scenario turns chunking on
#: without naming a budget. Mirrors
#: ``serving_text_to_speech.DEFAULT_CHUNK_SIZE`` (300) without importing the
#: serving module (which pulls in torch). Phase 12-7/12-8 restored the unified
#: 300-char default for every family; the old Turbo 150 / Multilingual 75
#: workarounds are gone. The unit test asserts the splitter yields more than one
#: chunk for every family fixture.
LONGFORM_DEFAULT_CHUNK_SIZE = 300


def longform_default_chunk_size(family: str | None) -> int:
    """Return the serving layer's default chunk budget for ``family``.

    Every family (including ``None``/unknown) uses the global 300-char default.
    """
    return LONGFORM_DEFAULT_CHUNK_SIZE

#: Requests per concurrency level for the long-form flood. Eight keeps a
#: c={1,4,8,20} sweep bounded on the benchmark host while still filling c=8;
#: pass ``--longform-requests`` to change it.
LONGFORM_REQUESTS = 8

#: The concurrency sweep the Phase 12-4 spec names.
LONGFORM_CONCURRENCY: tuple[int, ...] = (1, 4, 8, 20)

#: Execution-model classes under measurement. ``chatterbox_multilingual`` is
#: the loop-class control; ``melo`` is the padded-batch path; ``openvoice_v2``
#: is the clone pipeline; ``kitten`` is the lanes path.
LONGFORM_FAMILIES: tuple[str, ...] = (
    "chatterbox",
    "chatterbox_turbo",
    "chatterbox_multilingual",
    "melo",
    "openvoice_v2",
    "kitten",
)

LONGFORM_CLASS: dict[str, str] = {
    "chatterbox": "AR decode, admission width 4",
    "chatterbox_turbo": "AR decode, admission width 4",
    "chatterbox_multilingual": "AR decode, admission width 4",
    "melo": "padded batch",
    "openvoice_v2": "clone pipeline (VAD -> embed -> synth -> tone convert)",
    "kitten": "lanes",
}

#: Real-prose long-form fixture (owner-provided, 2026-11-09 listening verdict).
#:
#: The Phase 12-4 generator emitted word-index gibberish ("word-salad fixtures
#: are unlistenable"); this is the owner's extract from *The Villainess Malfoy*,
#: hard line-wraps normalised to single spaces and em-dashes/quotes/apostrophes
#: kept exactly. It exercises multi-sentence structure and the em-dash narrative
#: rule (450de15). A literal is even more deterministic than arithmetic
#: generation — identical bytes on every host and Python version — and every
#: family shares it, so ON/OFF and family comparisons differ only in the engine.
LONGFORM_FIXTURE_TEXT = (
    "Always destined to stand out, and always destined to shine the brightest—her name is "
    "Carina Malfoy. One could easily call her the true protagonist of today's Hogwarts. Even "
    "the Gryffindors, who normally bare their teeth at the mere sight of Slytherin, "
    "momentarily retract their fangs whenever she appears. When she offers an angelic smile, "
    "even the most notoriously difficult professors bend over backward to accommodate her. "
    "She doesn't even need to lift a finger. Hogwarts is crawling with fools desperate to "
    "touch so much as the tip of a Malfoy's sleeve. Her sycophants flock around her like "
    "decorative leaves adorn a flower, while ordinary pests hover about, hoping to catch even "
    "a passing whiff of her scent. Tiresome chores are handed off to the pests, and filthy "
    "matters are swept away by the leaves."
)

#: Provenance label carried into the benchmark report.
LONGFORM_FIXTURE_SOURCE = (
    "real prose (owner-provided extract from The Villainess Malfoy, 2026-11-09)"
)


def longform_fixture(family: str) -> str:  # noqa: ARG001 - shared fixture
    """Return the fixed real-prose paragraph shared by every family.

    807 characters and 8 sentences. At the serving layer's unified 300-character
    default it is **4 chunks** for every family (the unit tests pin the SHA-256
    and the per-family chunk counts). Deterministic by construction: a literal,
    not a generator.
    """
    return LONGFORM_FIXTURE_TEXT


def longform_chunk_count(
    text: str,
    chunk_size: int | None = None,
    language: str | None = None,
    family: str | None = None,
) -> int:
    """Number of chunks the engine will synthesize for ``text``.

    The response carries no chunk metadata, so the harness computes the
    expected count client-side with the *same* splitter the serving layer uses
    (``harmonyspeech.common.audio_utils.split_text_into_chunks``). This is the
    honest proxy; the engine DEBUG log's per-step composition is the observed
    side.

    ``chunk_size=None`` means the request leaves the budget unset, so the
    serving layer's unified 300-character default applies to every family
    (Phases 12-7/12-8). An explicit ``chunk_size`` always wins.
    """
    from harmonyspeech.common.audio_utils import split_text_into_chunks

    budget = (
        int(chunk_size)
        if chunk_size is not None
        else longform_default_chunk_size(family)
    )
    return len(split_text_into_chunks(text, chunk_size=budget, language=language))


def longform_fixture_metadata(family: str, chunk_size: int | None = None) -> dict:
    """Fixture/budget metadata for ``family`` (shared by both bench modes).

    ``chunk_size=None`` resolves the server's unified 300-character default; the
    returned ``expected_chunks_per_request`` is computed with the serving
    splitter. Both
    ``--scenario longform`` and ``--fidelity`` report this same block, so the
    fidelity rows cannot silently disagree with the throughput matrix.
    """
    spec = FAMILY_SPECS[family]
    text = longform_fixture(family)
    language = spec["payload"].get("language")
    budget = (
        int(chunk_size)
        if chunk_size is not None
        else longform_default_chunk_size(family)
    )
    return {
        "class": LONGFORM_CLASS.get(family, family),
        "fixture": "real_prose",
        "source": LONGFORM_FIXTURE_SOURCE,
        "characters": len(text),
        "words_ref": len(normalize_fidelity_words(text)),
        "sentences": text.count("."),
        "default_chunk_size": budget,
        "expected_chunks_per_request": longform_chunk_count(
            text, chunk_size, language, family=family
        ),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def build_longform_payload(
    family: str,
    reference_audio_b64: str,
    text: str,
    split_text: bool,
    *,
    chunk_size: int | None = None,
    crossfade_ms: int | None = None,
    sentence_pause_ms: int | None = None,
    filters: tuple[bool, bool, bool] = (False, False, False),
) -> dict:
    """One long-form TTS payload; ON/OFF differ only in chunking fields.

    The ``generation_options`` block is always present and always carries the
    three Phase 12-3 filter booleans (off in the core matrix), so the ON and OFF
    payloads are identical outside ``split_text``/``chunk_size``/
    ``crossfade_ms``/``sentence_pause_ms``. The unit self-check asserts that.

    ``chunk_size`` is omitted when ``None`` (its default), so the serving
    layer's **per-model** default applies — 300 characters, 150 for Chatterbox
    Turbo and 75 for Chatterbox Multilingual (Phase 12-5). ``crossfade_ms``/
    ``sentence_pause_ms`` are likewise omitted when ``None``, so a bench run
    measures the serving layer's Phase 12-5 defaults (40 ms / 120 ms) rather
    than pinning 0/0 from the harness. An explicit CLI value is still sent
    verbatim.
    """
    payload = _build_payload(family, reference_audio_b64, text_override=text)
    options: dict = {"split_text": bool(split_text)}
    if split_text:
        # Omitted when None so the serving layer's per-model default applies
        # (300 chars; 150 for Chatterbox Turbo, Phase 12-5). An explicit CLI
        # value is still sent verbatim.
        if chunk_size is not None:
            options["chunk_size"] = int(chunk_size)
        if crossfade_ms is not None:
            options["crossfade_ms"] = int(crossfade_ms)
        if sentence_pause_ms is not None:
            options["sentence_pause_ms"] = int(sentence_pause_ms)
    remove_dc, trim, fix_internal = filters
    options.update(
        remove_dc_offset=bool(remove_dc),
        trim_silence=bool(trim),
        fix_internal_silence=bool(fix_internal),
    )
    payload["generation_options"] = options
    return payload


def longform_payload_pair(
    family: str,
    reference_audio_b64: str,
    text: str,
    *,
    chunk_size: int | None = None,
    crossfade_ms: int | None = None,
    sentence_pause_ms: int | None = None,
    filters: tuple[bool, bool, bool] = (False, False, False),
) -> dict[str, dict]:
    """``{"off": payload, "on": payload}`` for the same text/voice/mode."""
    return {
        split: build_longform_payload(
            family,
            reference_audio_b64,
            text,
            split == "on",
            chunk_size=chunk_size,
            crossfade_ms=crossfade_ms,
            sentence_pause_ms=sentence_pause_ms,
            filters=filters,
        )
        for split in ("off", "on")
    }


def longform_scenario_matrix(
    concurrencies: tuple[int, ...] = LONGFORM_CONCURRENCY,
    families: tuple[str, ...] = LONGFORM_FAMILIES,
) -> list[dict]:
    """The Phase 12-4 benchmark cells (config-side, so the script prints them).

    Every family gets a ``split_text`` ON and OFF cell at each concurrency
    level. Each cell is one bench run; ``--longform-split both`` runs the pair
    back to back on the same boot.
    """
    cells: list[dict] = []
    for family in families:
        for split in ("off", "on"):
            for concurrency in concurrencies:
                cells.append(
                    {
                        "label": f"phase12-{family}-split-{split}-c{concurrency}",
                        "scenario": "longform",
                        "family": family,
                        "split_text": split == "on",
                        "concurrency": concurrency,
                        "config_note": f"{LONGFORM_CLASS.get(family, family)}, split_text={split}",
                    }
                )
    return cells


def longform_families_from_args(args) -> list[str]:
    """Resolve ``--families`` for ``--scenario longform``.

    Leaving ``--families`` at its ordinary default means the long-form set, so
    ``--scenario longform`` works on its own; an explicit list is validated
    against the six measured families.
    """
    families = [name.strip() for name in args.families.split(",") if name.strip()]
    if args.families == DEFAULT_FAMILIES:
        families = list(LONGFORM_FAMILIES)
    unknown = [name for name in families if name not in LONGFORM_FAMILIES]
    if unknown:
        raise SystemExit(f"unknown long-form families: {', '.join(unknown)}")
    return families


def _wav_seconds(audio_b64: str) -> float | None:
    """Duration of a base64 WAV response, or ``None`` if it is not decodable."""
    import base64
    import io

    import soundfile as sf

    try:
        raw = base64.b64decode(audio_b64)
        info = sf.info(io.BytesIO(raw))
        return round(info.frames / float(info.samplerate), 3)
    except Exception:  # noqa: BLE001 - a missing duration is not a request failure
        return None


async def _run_longform_level(
    client,
    base_url: str,
    family: str,
    payload: dict,
    concurrency: int,
    total: int,
    timeout: float,
    log_tail=None,
) -> LevelResult:
    """Run one long-form level, keeping the first successful response duration.

    The HTTP response has no chunk metadata, so the observed side of the
    chunking question is (a) the engine DEBUG log's per-step composition and
    (b) the decoded audio duration of the first success. Expected chunk counts
    are computed client-side by the caller.
    """
    queue: asyncio.Queue = asyncio.Queue()
    for index in range(total):
        queue.put_nowait(index)
    latencies: list[float] = []
    failures = 0
    first_audio: float | None = None
    claim_audio = True
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal failures, first_audio, claim_audio
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            started = time.perf_counter()
            elapsed = 0.0
            ok = False
            audio_seconds = None
            try:
                response = await client.post(
                    base_url + FAMILY_SPECS[family]["path"], json=payload, timeout=timeout
                )
                elapsed = time.perf_counter() - started
                ok = response.status_code == 200
                if ok:
                    body = response.json()
                    ok = bool(body.get("data") or body.get("text"))
                    if ok and body.get("data"):
                        # Decode at most one response per level: the duration is
                        # informational and a per-response decode would add
                        # client CPU time to the measured wall clock.
                        take = False
                        async with lock:
                            if claim_audio and first_audio is None:
                                claim_audio = False
                                take = True
                        if take:
                            audio_seconds = _wav_seconds(body["data"])
            except Exception:  # noqa: BLE001 - a client timeout/failure is data
                elapsed = time.perf_counter() - started
                ok = False
            async with lock:
                if ok:
                    latencies.append(elapsed)
                    if first_audio is None and audio_seconds is not None:
                        first_audio = audio_seconds
                else:
                    failures += 1

    if log_tail is not None:
        log_tail.flush()
    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(max(1, int(concurrency)))))
    wall = time.perf_counter() - started
    steps = parse_step_composition(log_tail.drain()) if log_tail is not None else {}
    return LevelResult(
        concurrency=concurrency,
        requests=total,
        failures=failures,
        wall_s=wall,
        latencies_s=latencies,
        steps=steps,
        audio_seconds=first_audio,
    )


async def run_longform_benchmark(args) -> dict:
    """Run the Phase 12-4 long-form matrix (chunking on/off, flood)."""
    import httpx

    from tests.audio_fixtures import reference_wav_b64

    reference = reference_wav_b64(args.reference)
    families = longform_families_from_args(args)
    levels = list(args.concurrency)
    split_modes = longform_split_modes(args.longform_split)
    filters = (
        bool(args.longform_remove_dc_offset),
        bool(args.longform_trim_silence),
        bool(args.longform_fix_internal_silence),
    )

    max_connections, max_keepalive = http_client_limits(levels, args.http_max_connections)
    log_tail = EngineLogTail(args.engine_log) if args.engine_log else None
    runs: dict[str, list[dict]] = {}
    skipped: dict[str, str] = {}
    fixture_report: dict[str, dict] = {}

    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
    async with httpx.AsyncClient(limits=limits) as client:
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
            text = longform_fixture(family)
            # ``--longform-chunk-size`` defaults to None, so the ON payload
            # leaves the budget unset and the server resolves it per model
            # (300; 150 for turbo; 75 for multilingual). The expected count must
            # mirror that.
            fixture_report[family] = longform_fixture_metadata(
                family, args.longform_chunk_size
            )
            chunks = fixture_report[family]["expected_chunks_per_request"]
            budget = fixture_report[family]["default_chunk_size"]
            print(
                f"[bench] {family}: real-prose fixture, {len(text)} chars, "
                f"{text.count('.')} sentence(s), {chunks} expected chunk(s)/request "
                f"at {budget} chars",
                flush=True,
            )
            for split in split_modes:
                payload = build_longform_payload(
                    family,
                    reference,
                    text,
                    split == "on",
                    chunk_size=args.longform_chunk_size,
                    crossfade_ms=args.longform_crossfade_ms,
                    sentence_pause_ms=args.longform_sentence_pause_ms,
                    filters=filters,
                )
                payload["model"] = model
                for _ in range(args.warmup):
                    await _one_request(client, args.base_url, family, payload, args.timeout)

                results = []
                for level in levels:
                    result = await _run_longform_level(
                        client,
                        args.base_url,
                        family,
                        payload,
                        level,
                        args.longform_requests,
                        args.timeout,
                        log_tail=log_tail,
                    )
                    results.append(result)
                    print(
                        f"[bench] {family} split={split} c={level}: "
                        f"{result.throughput:.2f} req/s "
                        f"p50={result.percentile(0.5) * 1000:.0f}ms "
                        f"p95={result.percentile(0.95) * 1000:.0f}ms "
                        f"audio={result.audio_seconds}s failures={result.failures}",
                        flush=True,
                    )
                    for step, stats in sorted(result.steps.items()):
                        print(
                            f"[bench]   {step}: {stats['steps']} step(s), "
                            f"{stats['sub_batches']} sub-batch(es), "
                            f"avg {stats['avg_sub_batches_per_step']}/step",
                            flush=True,
                        )
                runs[f"{family}-split-{split}"] = [result.as_dict() for result in results]

    return {
        "scenario": "longform",
        "label": args.label,
        "config": args.config_note,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_commit(),
        "env": _gpu_description(),
        "base_url": args.base_url,
        "requests_per_level": args.longform_requests,
        "concurrency_levels": levels,
        "clone_reference": args.clone_reference,
        "longform": {
            "seed": LONGFORM_SEED,
            "fixture": LONGFORM_FIXTURE_SOURCE,
            "chunk_size": args.longform_chunk_size,
            "crossfade_ms": args.longform_crossfade_ms,
            "sentence_pause_ms": args.longform_sentence_pause_ms,
            "filters": {
                "remove_dc_offset": filters[0],
                "trim_silence": filters[1],
                "fix_internal_silence": filters[2],
            },
            "fixtures": fixture_report,
        },
        "runs": runs,
        "skipped": skipped,
    }


def _longform_markdown(report: dict) -> str:
    """Markdown block for one long-form run (rows map 1:1 to the docs matrix)."""
    env = report.get("env") or {}
    longform = report.get("longform") or {}
    filters = longform.get("filters") or {}
    levels = report.get("concurrency_levels") or []
    crossfade = longform.get("crossfade_ms")
    pause = longform.get("sentence_pause_ms")
    chunk_size = longform.get("chunk_size")
    crossfade_cell = f"{crossfade} ms" if crossfade is not None else "server default (40 ms)"
    pause_cell = f"{pause} ms" if pause is not None else "server default (120 ms)"
    chunk_cell = (
        f"{chunk_size} chars"
        if chunk_size is not None
        else "server default (300) chars"
    )
    fixture_note = longform.get("fixture") or "long-form fixture"
    lines = [
        f"### {report['label']} — {report['timestamp_utc']}",
        "",
        f"* commit `{report.get('commit') or 'dirty'}` · torch {env.get('torch')} · "
        f"{env.get('gpu') or 'CPU'} · driver {env.get('driver') or 'n/a'} · "
        f"CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices') or 'all'}",
        f"* config: {report.get('config') or 'n/a'}",
        f"* fixture: {fixture_note} · chunk budget {chunk_cell} · "
        f"crossfade {crossfade_cell} · pause {pause_cell} · "
        f"filters dc={filters.get('remove_dc_offset')} trim={filters.get('trim_silence')} "
        f"internal={filters.get('fix_internal_silence')}",
        f"* {report.get('requests_per_level')} requests per concurrency level, HTTP against "
        f"{report.get('base_url')}",
        "",
    ]

    # Wide matrix: one row per (family, chunking), one pair of columns per level.
    header = ["| Family | Class | Chunking | Chars | Chunks/req |"]
    divider = ["|---|---|---|---|---|"]
    for level in levels:
        header.append(f" c={level} req/s | c={level} p50 (ms) |")
        divider.append("---|---|")
    lines.append("".join(header))
    lines.append("".join(divider))
    fixtures = longform.get("fixtures") or {}
    for run_name, results in (report.get("runs") or {}).items():
        family, _, split = run_name.partition("-split-")
        fixture = fixtures.get(family) or {}
        # With chunking off the serving layer takes the single-shot path, so the
        # observed chunk count is 1; the fixture's expected count is the ON row.
        chunks_cell = str(fixture.get("expected_chunks_per_request", "n/a")) if split == "on" else "1"
        cells = [
            family,
            fixture.get("class", "n/a"),
            split,
            str(fixture.get("characters", "n/a")),
            chunks_cell,
        ]
        by_level = {result["concurrency"]: result for result in results}
        for level in levels:
            result = by_level.get(level)
            if result is None:
                cells.extend(["_tbd_", "_tbd_"])
            else:
                cells.extend([str(result["throughput_rps"]), str(result["p50_ms"])])
        lines.append("| " + " | ".join(cells) + " |")

    composition = [
        (run_name, result["concurrency"], step, stats)
        for run_name, results in (report.get("runs") or {}).items()
        for result in results
        for step, stats in sorted((result.get("steps") or {}).items())
    ]
    if composition:
        lines += [
            "",
            "Per-step sub-batch composition (engine DEBUG log; Phase 12-4):",
            "",
            "| Run | Concurrency | Step | Steps logged | Sub-batches | avg/step | Rows | Requests | Lanes |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for run_name, concurrency, step, stats in composition:
            lines.append(
                f"| {run_name} | {concurrency} | {step} | {stats['steps']} | "
                f"{stats['sub_batches']} | {stats['avg_sub_batches_per_step']} | "
                f"{stats['rows']} | {stats['requests']} | {stats.get('lanes', '—')} |"
            )
    if report.get("skipped"):
        lines.append("")
        for family, reason in report["skipped"].items():
            lines.append(f"* {family}: skipped — {reason}")
    lines.append("")
    return "\n".join(lines)


def longform_scenario_recipe(args) -> str:
    """Shell recipe for the Phase 12-4 long-form matrix + regression gates."""
    port = args.base_url.rsplit(":", 1)[-1]
    engine_log = args.engine_log or "out/bench/engine-phase12-info.log"
    debug_log = "out/bench/engine-phase12-debug.log"
    md_flag = " --md-out docs/benchmarks/throughput.md"
    gate_families = (
        "chatterbox,chatterbox_turbo,chatterbox_multilingual,openvoice_v1,"
        "openvoice_v2,melotts,harmonyspeech,kittentts"
    )
    lines = [
        "# Phase 12-4 long-form matrix. One idle benchmark host (RTX 5060 Ti,",
        "# CUDA_VISIBLE_DEVICES=1). The results are _tbd_ in",
        "# docs/benchmarks/throughput.md §Phase 12 until this session fills them.",
        "#",
        "# Fixture: one fixed real-prose paragraph shared by every family",
        "# (owner-provided extract from The Villainess Malfoy, 2026-11-09):",
        "# 807 characters, 8 sentences; 4 chunks at the unified",
        "# 300-character default for every family.",
        "# ON runs omit chunk_size (unless --longform-chunk-size is set) so the",
        "# serving layer applies that unified default, and omit",
        "# crossfade_ms/sentence_pause_ms unless the flags below set them, so they",
        "# measure the Phase 12-5 defaults (40 ms fade / 120 ms pause).",
        "",
        "# Boot A - headline numbers. INFO logging: DEBUG depresses the clone",
        "# pipeline ~3x (see the Method host note).",
        "CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=INFO \\",
        "    python -m harmonyspeech.endpoints.openai.api_server --config-file-path config.gpu.yml \\",
        f"    --port {port} > {engine_log} 2>&1",
        "",
        "# Wait for health (bounded), then run one family at a time. Each run emits",
        "# both the split_text OFF and ON rows at c=1,4,8,20 (single-request",
        "# latency is the c=1 level).",
    ]
    for family in LONGFORM_FAMILIES:
        lines.append(
            f"python scripts/bench_throughput.py --scenario longform --families {family} \\\n"
            f"    --longform-split both --concurrency 1,4,8,20 --longform-requests "
            f"{LONGFORM_REQUESTS} --label phase12-{family} \\\n"
            f"    --json-out out/bench/throughput-phase12-{family}.json{md_flag}"
        )
    lines += [
        "",
        "# Boot B - composition cells only (DEBUG + --engine-log). Each command is",
        "# the ON row for one family; the markdown block then carries the per-step",
        "# sub-batch table.",
        "CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=DEBUG \\",
        "    python -m harmonyspeech.endpoints.openai.api_server --config-file-path config.gpu.yml \\",
        f"    --port {port} > {debug_log} 2>&1",
        "",
    ]
    for family in LONGFORM_FAMILIES:
        lines.append(
            f"python scripts/bench_throughput.py --scenario longform --families {family} \\\n"
            f"    --longform-split on --concurrency 1,4,8,20 --longform-requests "
            f"{LONGFORM_REQUESTS} --engine-log {debug_log} \\\n"
            f"    --label phase12-{family}-debug "
            f"--json-out out/bench/throughput-phase12-{family}-debug.json{md_flag}"
        )
    lines += [
        "",
        "# Regression gate (a): short-text split_text:true is byte-identical to off",
        "# at family breadth. The in-process 12-2 gate is bit-exact; this cross-",
        "# process re-run disables the watermark (Perth's conv kernels are not",
        "# bit-reproducible across processes) and requires identical length + a",
        "# bit-identical WAV for deterministic families (documented CUDA noise",
        "# floor for stochastic ones).",
        f"python scripts/audio_regression.py --families {gate_families} \\",
        "    --axis longform:off-vs-on --gpu \\",
        "    --json-out out/bench/gate-12-4-identity.json",
        "",
        "# Regression gate (b): seeded chunked determinism. Both sides run the same",
        "# chunked request with seed 1234; same seed must give the same stitched",
        "# bytes (identical WAV).",
        f"python scripts/audio_regression.py --families {gate_families} \\",
        "    --axis determinism:run1-vs-run2 --gpu \\",
        "    --json-out out/bench/gate-12-4-determinism.json",
        "",
        "# H3 crossfade-cost probe: one ON family, c=1, crossfade 0 vs 50 ms.",
        "python scripts/bench_throughput.py --scenario longform --families chatterbox \\",
        "    --longform-split on --concurrency 1 --longform-requests 4 \\",
        "    --longform-crossfade-ms 50 --longform-sentence-pause-ms 50 \\",
        "    --label phase12-crossfade-50 --json-out out/bench/throughput-phase12-crossfade.json",
        "",
        "# H4 filter-cost probe: one ON family with all three Phase 12-3 filters.",
        "python scripts/bench_throughput.py --scenario longform --families melo \\",
        "    --longform-split on --concurrency 1,4 --longform-requests 8 \\",
        "    --longform-remove-dc-offset --longform-trim-silence \\",
        "    --longform-fix-internal-silence --label phase12-filters-on \\",
        "    --json-out out/bench/throughput-phase12-filters.json",
        "",
        "# Print this recipe",
        "python scripts/bench_throughput.py --print-longform-scenarios",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 12-5 follow-up — Whisper text-fidelity gate (owner decision 2026-11-09)
# ---------------------------------------------------------------------------
#
# The gap that let Turbo's word-dropping ship unnoticed: the throughput matrix
# measured durations, never whether the audio said the text. ``--fidelity``
# renders each family's fixture once per variant, transcribes the WAV with
# faster-whisper and scores word edit distance against the fixture. The
# transcription noise floor on this fixture is ~2–6 edit distance
# ("Karina/Carina", "weaves/leaves"); values near that are not model errors.

#: Default directory for fidelity renders (WAVs) and transcripts.
FIDELITY_OUT_DIR = "out/bench/fidelity-12"
#: The transcription model the fidelity gate uses (the same large-v3-turbo the
#: Phase 12-5 investigation used).
FIDELITY_WHISPER_MODEL = "large-v3-turbo"
FIDELITY_LANGUAGE = "en"
FIDELITY_BEAM_SIZE = 5
#: Documented noise floor: edit distances at or below this are transcription
#: noise on the real-prose fixture, not a fidelity failure.
FIDELITY_NOISE_FLOOR = 6

_FIDELITY_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_fidelity_words(text: str) -> list[str]:
    """Case/punctuation-insensitive word tokens.

    Punctuation (including apostrophes) becomes a separator, matching the
    Phase 12-5 orchestrator probes: the real-prose fixture is **138 words**
    under this normalisation, so the sweep table's ``73/138`` / ``139/138``
    figures are directly comparable.
    """
    return _FIDELITY_PUNCT_RE.sub(" ", (text or "").lower()).split()


def word_edit_distance(reference: str, hypothesis: str) -> int:
    """Word edit distance (delete + insert; substitutions cost two).

    This is the LCS-based distance the Phase 12-5 orchestrator probes used:
    ``len(ref) + len(hyp) - 2 * LCS``. A hypothesis that silently skips words
    therefore scores exactly the number of skipped words plus insertion noise,
    which is what makes the budget sweep legible.
    """
    ref = normalize_fidelity_words(reference)
    hyp = normalize_fidelity_words(hypothesis)
    if not ref or not hyp:
        return len(ref) + len(hyp)
    previous = [0] * (len(hyp) + 1)
    for ref_token in ref:
        current = [0]
        for index, hyp_token in enumerate(hyp, start=1):
            if ref_token == hyp_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    longest_common = previous[-1]
    return len(ref) + len(hyp) - 2 * longest_common


def score_fidelity(reference_text: str, transcript: str) -> dict:
    """Score one transcript against the fixture text (the report row shape)."""
    return {
        "edit_distance": word_edit_distance(reference_text, transcript),
        "words_heard": len(normalize_fidelity_words(transcript)),
        "words_ref": len(normalize_fidelity_words(reference_text)),
        "transcript": transcript,
    }


def transcribe_and_score(reference_text: str, wav_path: str, transcriber) -> dict:
    """Transcribe ``wav_path`` with an injected ``transcriber`` and score it.

    ``transcriber`` is ``path -> text``; unit tests pass a stub so no Whisper
    install or model weights are needed.
    """
    return score_fidelity(reference_text, transcriber(wav_path))


def load_fidelity_transcriber(
    gpu: bool = False,
    model_id: str = FIDELITY_WHISPER_MODEL,
    language: str = FIDELITY_LANGUAGE,
    beam_size: int = FIDELITY_BEAM_SIZE,
):
    """Load faster-whisper lazily and return ``path -> transcript``.

    A missing install fails with a clear, actionable message instead of a bare
    traceback, because the flag is optional.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - exercised via monkeypatch in unit tests
        raise SystemExit(
            "--fidelity needs faster-whisper to transcribe the renders. Install it "
            "(pip install faster-whisper) or run the throughput matrix without "
            f"--fidelity. Import error: {error}"
        ) from error
    device = "cuda" if gpu else "cpu"
    compute_type = "float16" if gpu else "int8"
    model = WhisperModel(model_id, device=device, compute_type=compute_type)

    def transcribe(path: str) -> str:
        segments, _info = model.transcribe(
            path,
            language=language,
            beam_size=int(beam_size),
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe


def longform_split_modes(value: str | None, *, fidelity: bool = False) -> list[str]:
    """Resolve ``--longform-split`` for the throughput vs fidelity mode.

    Unset means "both" for the throughput matrix and "on" for the fidelity gate
    (the ON variant at the server's per-model budget is the shipped default;
    pass ``--longform-split both`` to score the OFF baseline too).
    """
    if value is None:
        value = "on" if fidelity else "both"
    return ["off", "on"] if value == "both" else [value]


async def _render_fidelity_wav(client, base_url: str, family: str, payload: dict, timeout: float, out_path: str):
    """Render one request, save its WAV and return ``(path, audio_seconds)``."""
    import base64

    response = await client.post(
        base_url + FAMILY_SPECS[family]["path"], json=payload, timeout=timeout
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"{family}: HTTP {response.status_code}: {(response.text or '')[:300]}"
        )
    body = response.json()
    data = body.get("data")
    if not data:
        raise RuntimeError(f"{family}: response carried no audio")
    raw = base64.b64decode(data)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(raw)
    return out_path, _wav_seconds(data)


async def run_fidelity_benchmark(args, transcriber=None) -> dict:
    """Render each family's fixture once per variant and score its transcript."""
    import httpx

    from tests.audio_fixtures import reference_wav_b64

    gpu = bool(getattr(args, "gpu", False))
    if transcriber is None:
        transcriber = load_fidelity_transcriber(
            gpu=gpu,
            model_id=getattr(args, "fidelity_model", FIDELITY_WHISPER_MODEL),
            language=getattr(args, "fidelity_language", FIDELITY_LANGUAGE),
            beam_size=getattr(args, "fidelity_beam", FIDELITY_BEAM_SIZE),
        )

    reference = reference_wav_b64(args.reference)
    families = longform_families_from_args(args)
    variants = longform_split_modes(args.longform_split, fidelity=True)
    out_dir = getattr(args, "fidelity_out", FIDELITY_OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    filters = (
        bool(args.longform_remove_dc_offset),
        bool(args.longform_trim_silence),
        bool(args.longform_fix_internal_silence),
    )

    families_report: dict[str, dict] = {}
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

            text = longform_fixture(family)
            metadata = longform_fixture_metadata(family, args.longform_chunk_size)
            row = {"model": model, **metadata, "variants": {}}
            for split in variants:
                payload = build_longform_payload(
                    family,
                    reference,
                    text,
                    split == "on",
                    chunk_size=args.longform_chunk_size,
                    crossfade_ms=args.longform_crossfade_ms,
                    sentence_pause_ms=args.longform_sentence_pause_ms,
                    filters=filters,
                )
                payload["model"] = model
                wav_path = os.path.join(out_dir, f"{family}-split-{split}.wav")
                try:
                    path, audio_seconds = await _render_fidelity_wav(
                        client, args.base_url, family, payload, args.timeout, wav_path
                    )
                    scored = transcribe_and_score(text, path, transcriber)
                except Exception as error:  # noqa: BLE001 - a failed render is report data
                    row["variants"][split] = {"error": str(error), "wav": wav_path}
                    print(f"[bench] {family} split={split}: ERROR {error}", flush=True)
                    continue
                row["variants"][split] = {
                    "wav": path,
                    "audio_seconds": audio_seconds,
                    "chunks": metadata["expected_chunks_per_request"] if split == "on" else 1,
                    **scored,
                }
                print(
                    f"[bench] {family} split={split}: edit distance "
                    f"{scored['edit_distance']} · words {scored['words_heard']}/"
                    f"{scored['words_ref']} · audio {audio_seconds}s",
                    flush=True,
                )
            families_report[family] = row

    return {
        "scenario": "longform-fidelity",
        "label": args.label,
        "config": args.config_note,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_commit(),
        "env": _gpu_description(),
        "base_url": args.base_url,
        "out_dir": out_dir,
        "clone_reference": args.clone_reference,
        "method": {
            "transcriber": getattr(args, "fidelity_model", FIDELITY_WHISPER_MODEL),
            "device": "cuda" if gpu else "cpu",
            "compute_type": "float16" if gpu else "int8",
            "language": getattr(args, "fidelity_language", FIDELITY_LANGUAGE),
            "beam_size": getattr(args, "fidelity_beam", FIDELITY_BEAM_SIZE),
            "normalization": (
                "lowercase, punctuation stripped (apostrophes split); word edit "
                "distance = len(ref) + len(hyp) - 2 * LCS (delete + insert)"
            ),
            "noise_floor_edit_distance": FIDELITY_NOISE_FLOOR,
            "noise_floor_examples": "Karina/Carina, weaves/leaves",
            "variants": variants,
            "chunk_budget": args.longform_chunk_size,
        },
        "families": families_report,
        "skipped": skipped,
    }


def _fidelity_markdown(report: dict) -> str:
    """Markdown block for one fidelity run (rows map 1:1 to the docs table)."""
    env = report.get("env") or {}
    method = report.get("method") or {}
    lines = [
        f"### {report['label']} — {report['timestamp_utc']}",
        "",
        f"* commit `{report.get('commit') or 'dirty'}` · torch {env.get('torch')} · "
        f"{env.get('gpu') or 'CPU'} · driver {env.get('driver') or 'n/a'} · "
        f"CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices') or 'all'}",
        f"* config: {report.get('config') or 'n/a'}",
        f"* method: Whisper `{method.get('transcriber')}` "
        f"({method.get('device')}/{method.get('compute_type')}), language "
        f"`{method.get('language')}`, beam {method.get('beam_size')}; "
        f"{method.get('normalization')}. Transcription noise floor ≈ "
        f"{method.get('noise_floor_edit_distance')} edit distance "
        f"({method.get('noise_floor_examples')}).",
        f"* renders: `{report.get('out_dir')}` · variants: "
        f"{', '.join(method.get('variants') or [])} · chunk budget: "
        f"{method.get('chunk_budget') if method.get('chunk_budget') is not None else 'server per-model default'}",
        "",
        "| family | variant | edit distance | words heard / ref |",
        "|---|---|---|---|",
    ]
    for family, row in (report.get("families") or {}).items():
        for split, variant in (row.get("variants") or {}).items():
            if variant.get("error"):
                lines.append(f"| {family} | {split} | error | — |")
            else:
                lines.append(
                    f"| {family} | {split} | {variant['edit_distance']} | "
                    f"{variant['words_heard']} / {variant['words_ref']} |"
                )
    if report.get("skipped"):
        lines.append("")
        for family, reason in report["skipped"].items():
            lines.append(f"* {family}: skipped — {reason}")
    lines.append("")
    return "\n".join(lines)


async def run_chatterbox_exhaustion(args) -> dict:
    """Run the Phase 9-6 tight-pool exhaustion scenario."""
    import httpx

    from tests.audio_fixtures import reference_wav_b64

    model = args.model.get("chatterbox") or FAMILY_SPECS["chatterbox"]["model"]
    specs = chatterbox_flood_scenario(args.chatterbox_flood_requests)
    reference = reference_wav_b64(args.reference)
    references = (
        distinct_reference_variants(reference, len(specs))
        if args.clone_reference == "distinct"
        else [reference] * len(specs)
    )
    payloads = [
        build_chatterbox_payload(spec, model, references[index])
        for index, spec in enumerate(specs)
    ]

    concurrency = args.chatterbox_flood_concurrency or len(specs)
    max_connections, max_keepalive = http_client_limits([concurrency], args.http_max_connections)
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
    async with httpx.AsyncClient(limits=limits) as client:
        print(
            f"[bench] chatterbox-exhaustion: {len(specs)} distinct-voice request(s), "
            f"concurrency {concurrency}, pool slots ~{args.chatterbox_pool_slots}",
            flush=True,
        )
        sampler = ARMetricsSampler(client, args.base_url, args.ar_sample_interval)
        records, wall, samples = await _run_ar_scenario(
            client, args.base_url, specs, payloads, concurrency, args.timeout, sampler
        )

    summary = summarize_ar_scenario(specs, records, wall, samples)
    print(
        f"[bench] chatterbox-exhaustion: {summary['completed']}/{summary['requests']} completed, "
        f"{summary['failures']} failed ({summary['timeouts']} timeout(s)), "
        f"loss={summary['request_loss']}, refusals={summary['refusals_delta']}, "
        f"max slots active={summary['max_slots_active']}",
        flush=True,
    )
    return {
        "scenario": "chatterbox-exhaustion",
        "label": args.label,
        "config": args.config_note,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_commit(),
        "env": _gpu_description(),
        "base_url": args.base_url,
        "model": model,
        "clone_reference": args.clone_reference,
        "caps": {
            "pool_slots": args.chatterbox_pool_slots,
            "concurrency": concurrency,
            "requests": len(specs),
        },
        "summary": {
            key: value for key, value in summary.items() if key not in ("ar_trace", "records_detail")
        },
        "ar_trace": summary["ar_trace"],
        "records": summary["records_detail"],
    }


def _ar_exhaustion_markdown(report: dict) -> str:
    """Markdown block for one AR exhaustion run."""
    summary = report["summary"]
    caps = report.get("caps") or {}
    env = report.get("env") or {}
    lines = [
        f"### {report['label']} — {report['timestamp_utc']}",
        "",
        f"* commit `{report.get('commit') or 'dirty'}` · torch {env.get('torch')} · "
        f"{env.get('gpu') or 'CPU'} · driver {env.get('driver') or 'n/a'}",
        f"* config: {report.get('config') or 'n/a'}",
        f"* pool slots ~{caps.get('pool_slots')} · client concurrency {caps.get('concurrency')} · "
        f"{caps.get('requests')} requests · reference mode {report.get('clone_reference')}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Requests submitted | {summary['requests']} |",
        f"| Requests answered (loss) | {summary['records']} ({summary['request_loss']}) |",
        f"| Completed / failed | {summary['completed']} / {summary['failures']} |",
        f"| Timeouts | {summary['timeouts']} |",
        f"| Wall / throughput | {summary['wall_s']} s / {summary['throughput_rps']} req/s |",
        f"| p50 / p95 | {summary['p50_ms']} ms / {summary['p95_ms']} ms |",
        f"| Max slots active | {summary['max_slots_active']} |",
        f"| Slot grants / refusals (delta) | {summary['grants_delta']} / {summary['refusals_delta']} |",
        f"| Max tokens/step | {summary['max_tokens_per_step']} |",
        "",
    ]
    return "\n".join(lines)


def ar_scenario_recipe(args) -> str:
    """Shell recipe for the Phase 9-6 three-way comparison + exhaustion."""
    port = args.base_url.rsplit(":", 1)[-1]
    engine_log = args.engine_log or "out/bench/engine-ar.log"
    log_flag = f" --engine-log {engine_log}"
    md_flag = " --md-out docs/benchmarks/throughput.md"
    lines = [
        "# Phase 9-6 Chatterbox AR scenarios. One idle benchmark host (RTX 5060 Ti,",
        "# CUDA_VISIBLE_DEVICES=1). Every cell needs a matching engine boot; the",
        "# results are _tbd_ in docs/benchmarks/throughput.md §Phase 9.",
        "",
        "# Boot A - shipped AR runner (max_batch_size 4, ar_inference 1 GiB / 768)",
        "CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=DEBUG \\",
        "    python -m harmonyspeech.endpoints.openai.api_server --config-file-path config.gpu.yml \\",
        f"    --port {port} > {engine_log} 2>&1",
        "",
        "# Three-way: full AR runner at c=4,8,20 for both families",
        "python scripts/bench_throughput.py --scenario chatterbox-ar --families chatterbox \\",
        f"    --concurrency 4,8,20 --requests 20 --label phase9-chatterbox-ar-runner{log_flag}{md_flag}",
        "python scripts/bench_throughput.py --scenario chatterbox-ar --families chatterbox_turbo \\",
        f"    --concurrency 4,8,20 --requests 20 --label phase9-chatterbox-turbo-ar-runner{log_flag}{md_flag}",
        "",
        "# Boot B - sequential loop (batched_inference: false, max_batch_size 1)",
        "CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=DEBUG \\",
        "    python -m harmonyspeech.endpoints.openai.api_server --config-file-path out/bench/config-phase9-loop.yml \\",
        f"    --port {port} > {engine_log} 2>&1",
        "python scripts/bench_throughput.py --scenario chatterbox-ar --families chatterbox \\",
        f"    --concurrency 4,8,20 --requests 20 --label phase9-chatterbox-sequential{log_flag}{md_flag}",
        "python scripts/bench_throughput.py --scenario chatterbox-ar --families chatterbox_turbo \\",
        f"    --concurrency 4,8,20 --requests 20 --label phase9-chatterbox-turbo-sequential{log_flag}{md_flag}",
        "",
        "# AR-0: the same boots on the parent commit (Phase 9-1's cleaned loop)",
        "git worktree add ../hse-ar0-before 4081d97",
        "python scripts/bench_throughput.py --scenario chatterbox-ar --families chatterbox \\",
        f"    --concurrency 4,8,20 --requests 20 --label phase9-chatterbox-ar0{log_flag}{md_flag}",
        "",
        "# Cache composition: same vs distinct voice x cache on/off (two boots)",
        "python scripts/bench_throughput.py --scenario chatterbox-ar --families chatterbox \\",
        f"    --concurrency 8 --requests 20 --clone-reference same --embedding-cache on \\",
        f"    --label phase9-cache-same-on{log_flag}{md_flag}",
        "python scripts/bench_throughput.py --scenario chatterbox-ar --families chatterbox \\",
        f"    --concurrency 8 --requests 20 --clone-reference distinct --embedding-cache on \\",
        f"    --label phase9-cache-distinct-on{log_flag}{md_flag}",
        "# restart with --no-embedding-cache, then the two off rows",
        "python scripts/bench_throughput.py --scenario chatterbox-ar --families chatterbox \\",
        f"    --concurrency 8 --requests 20 --clone-reference same --embedding-cache off \\",
        f"    --label phase9-cache-same-off{log_flag}{md_flag}",
        "",
        "# Mixed-length rows: the flood rotates CHATTERBOX_TEXTS, so short and long",
        "# rows leave the batch at different decode steps. Watch hse_ar_decode_* in",
        "# the metrics trace (tokens/step = active batch width).",
        "",
        "# Exhaustion: 100 distinct-voice requests against a pool sized for ~8 slots.",
        "# Boot with ar_inference.kv_cache_budget_bytes = 8 x slab (the startup arena",
        "# table prints the slab size). Record loss (must be 0) and refusal delta.",
        "python scripts/bench_throughput.py --scenario chatterbox-exhaustion \\",
        "    --clone-reference distinct --chatterbox-flood-requests 100 \\",
        "    --chatterbox-flood-concurrency 100 --chatterbox-pool-slots 8 \\",
        f"    --label phase9-exhaustion{log_flag}{md_flag}",
        "",
        "# Print this recipe",
        "python scripts/bench_throughput.py --print-ar-scenarios",
        "",
    ]
    return "\n".join(lines)


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
    client,
    base_url,
    family,
    payloads,
    concurrency: int,
    total: int,
    timeout: float,
    log_tail=None,
    ar_sampler=None,
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
    if ar_sampler is not None:
        await ar_sampler.start()
    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - started
    ar_trace = await ar_sampler.stop() if ar_sampler is not None else []
    steps = parse_step_composition(log_tail.drain()) if log_tail is not None else {}
    return LevelResult(
        concurrency=concurrency,
        requests=total,
        failures=failures,
        wall_s=wall,
        latencies_s=latencies,
        steps=steps,
        ar_trace=ar_trace,
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
    ar_sampler_enabled = getattr(args, "scenario", "throughput") == "chatterbox-ar"
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
            if getattr(args, "mixed_lengths", False) and family in ("chatterbox", "chatterbox_turbo"):
                # Mixed-length rows: short and long texts leave the AR batch at
                # different decode steps (the owner's sub-batch sanity check).
                for index, payload in enumerate(payloads):
                    payload["input"] = CHATTERBOX_TEXTS[index % len(CHATTERBOX_TEXTS)]
            for payload in payloads:
                payload["model"] = model

            for _ in range(args.warmup):
                await _one_request(client, args.base_url, family, payloads[0], args.timeout)

            results = []
            for level in levels:
                level_sampler = (
                    ARMetricsSampler(client, args.base_url, args.ar_sample_interval)
                    if ar_sampler_enabled
                    else None
                )
                result = await _run_level(
                    client,
                    args.base_url,
                    family,
                    payloads,
                    level,
                    args.requests,
                    args.timeout,
                    log_tail=log_tail,
                    ar_sampler=level_sampler,
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


async def _discover_whisper_budget(client, base_url: str) -> dict | None:
    """Read the configured Whisper budget from ``/health`` (best effort)."""
    try:
        response = await client.get(base_url + "/health", timeout=30)
        if response.status_code != 200:
            return None
        body = response.json() or {}
        whisper = body.get("whisper_vram") or {}
        vram = body.get("vram") or {}
        budget = whisper.get("budget_bytes", vram.get("whisper_budget_bytes"))
        return {"budget_bytes": budget, "status": whisper.get("status")}
    except Exception:  # pragma: no cover - verification is best effort
        return None


async def run_whisper_exhaustion(args) -> dict:
    """Run the Phase 10-5 exhaustion / fairness / budget-trip scenario."""
    import httpx

    from tests.audio_fixtures import reference_wav_b64

    scenario = args.scenario
    reference_audio = reference_wav_b64(args.reference)
    model = args.model.get("whisper") or FAMILY_SPECS["whisper"]["model"]

    if scenario == "whisper-fairness":
        specs = whisper_fairness_scenario(args.whisper_fairness_long_seconds)
        concurrency = len(specs)
    else:
        specs = whisper_flood_scenario(args.whisper_flood_requests, args.whisper_max_seconds)
        concurrency = args.whisper_flood_concurrency

    audio_cache: dict = {}
    payloads = [
        build_whisper_payload(
            spec, duration_audio_b64(reference_audio, spec["duration_s"], audio_cache), model
        )
        for spec in specs
    ]

    max_connections, max_keepalive = http_client_limits([concurrency], args.http_max_connections)
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
    async with httpx.AsyncClient(limits=limits) as client:
        budget_report = None
        if scenario == "whisper-budget-trip":
            if args.whisper_vram_budget_bytes is None:
                raise SystemExit(
                    "--scenario whisper-budget-trip requires --whisper-vram-budget-bytes "
                    "(set the same value on the engine config)"
                )
            budget_report = await _discover_whisper_budget(client, args.base_url)
            observed = (budget_report or {}).get("budget_bytes")
            if observed is not None and int(observed) != int(args.whisper_vram_budget_bytes):
                raise SystemExit(
                    f"--whisper-vram-budget-bytes {args.whisper_vram_budget_bytes} does not match the "
                    f"running engine ({observed}). Boot the engine with that budget and rerun."
                )
            if observed is None:
                print(
                    "[bench] the engine did not report a whisper budget on /health; recording the "
                    "requested value without verification",
                    flush=True,
                )

        print(
            f"[bench] {scenario}: {len(specs)} request(s), concurrency {concurrency}, "
            f"declared caps M={args.whisper_workers} window_batch={args.whisper_window_batch} "
            f"queue={args.whisper_queue}",
            flush=True,
        )
        sampler = MetricsSampler(client, args.base_url, args.whisper_sample_interval)
        records, wall, samples = await _run_whisper_scenario(
            client,
            args.base_url,
            specs,
            payloads,
            concurrency,
            args.timeout,
            sampler,
        )

    summary = summarize_whisper_scenario(specs, records, wall, samples)
    budget_bytes = args.whisper_vram_budget_bytes
    within_budget = None
    if budget_bytes is not None and summary["max_device_used_bytes"] is not None:
        within_budget = summary["max_device_used_bytes"] <= int(budget_bytes)

    print(
        f"[bench] {scenario}: {summary['completed']}/{summary['requests']} completed, "
        f"{summary['failures']} failed ({summary['timeouts']} timeout(s)), "
        f"loss={summary['request_loss']}, p50={summary['p50_ms']:.0f}ms "
        f"p95={summary['p95_ms']:.0f}ms",
        flush=True,
    )
    if within_budget is not None:
        print(
            f"[bench] budget hold: max device used {summary['max_device_used_bytes']} B vs "
            f"budget {budget_bytes} B -> {within_budget}",
            flush=True,
        )

    return {
        "scenario": scenario,
        "label": args.label,
        "config": args.config_note,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_commit(),
        "env": _gpu_description(),
        "base_url": args.base_url,
        "model": model,
        "caps": {
            "workers": args.whisper_workers,
            "window_batch": args.whisper_window_batch,
            "queue_depth": args.whisper_queue,
            "concurrency": concurrency,
        },
        "budget": {
            "requested_bytes": budget_bytes,
            "observed": budget_report,
            "device_used_within_budget": within_budget,
        },
        "summary": {
            key: value
            for key, value in summary.items()
            if key not in ("lane_trace", "records_detail")
        },
        "lane_trace": summary["lane_trace"],
        "records": summary["records_detail"],
    }


def _whisper_markdown(report: dict) -> str:
    """Markdown block for one exhaustion/fairness/budget-trip run."""
    summary = report["summary"]
    caps = report.get("caps") or {}
    budget = report.get("budget") or {}
    env = report.get("env") or {}
    lines = [
        f"### {report['label']} — {report['timestamp_utc']}",
        "",
        f"* commit `{report.get('commit') or 'dirty'}` · torch {env.get('torch')} · "
        f"{env.get('gpu') or 'CPU'} · driver {env.get('driver') or 'n/a'} · "
        f"CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices') or 'all'}",
        f"* config: {report.get('config') or 'n/a'}",
        f"* declared caps: M={caps.get('workers')} · window batch={caps.get('window_batch')} · "
        f"queue depth={caps.get('queue_depth')} · client concurrency={caps.get('concurrency')}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Requests submitted | {summary['requests']} |",
        f"| Requests answered (loss) | {summary['records']} ({summary['request_loss']}) |",
        f"| Completed / failed | {summary['completed']} / {summary['failures']} |",
        f"| Timeouts | {summary['timeouts']} |",
        f"| Format/pairing errors | {summary['format_errors']} |",
        f"| Wall | {summary['wall_s']} s |",
        f"| Throughput | {summary['throughput_rps']} req/s |",
        f"| p50 / p95 | {summary['p50_ms']} ms / {summary['p95_ms']} ms |",
    ]
    if summary.get("short_p50_ms") is not None:
        lines.append(
            f"| Short-clip p50 / p95 | {summary['short_p50_ms']} ms / {summary['short_p95_ms']} ms |"
        )
    lines.append(
        f"| Queue wait (histogram sum/count) | {summary['queue_wait_sum_s']} s / {summary['queue_wait_count']} |"
    )
    if summary.get("max_device_used_bytes") is not None:
        lines.append(f"| Max device used | {summary['max_device_used_bytes']} B |")
    if budget.get("requested_bytes") is not None:
        lines.append(
            f"| Budget / device-used within budget | {budget['requested_bytes']} B / "
            f"{budget.get('device_used_within_budget')} |"
        )

    trace = report.get("lane_trace") or []
    if trace:
        # Keep the table readable: up to 20 evenly spaced samples.
        step = max(1, len(trace) // 20)
        rows = trace[::step]
        lines += [
            "",
            "Lane depths over time (sampled from `/metrics`):",
            "",
            "| t (s) | Model | In flight | Short lane | Long lane | CT2 queued | CT2 active |",
            "|---|---|---|---|---|---|---|",
        ]
        for sample in rows:
            for model_name, detail in (sample.get("models") or {}).items():
                lines.append(
                    f"| {sample.get('t_s')} | {model_name} | {detail.get('in_flight', 0)} | "
                    f"{detail.get('lane_short_depth', 0)} | {detail.get('lane_long_depth', 0)} | "
                    f"{detail.get('ct2_queued', 0)} | {detail.get('ct2_active', 0)} |"
                )
    lines.append("")
    return "\n".join(lines)


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
    ar_rows = [
        (family, level["concurrency"], level["ar_trace"])
        for family, levels in report["families"].items()
        for level in levels
        if level.get("ar_trace")
    ]
    if ar_rows:
        lines += [
            "",
            "AR pool utilization (sampled from `/metrics`; Phase 9-6):",
            "",
            "| Family | Concurrency | Max slots active | Grants Δ | Refusals Δ | Max tokens/step |",
            "|---|---|---|---|---|---|",
        ]
        for family, concurrency, trace in ar_rows:
            first = next((sample["cache"] for sample in trace if sample.get("cache")), {})
            last = next((sample["cache"] for sample in reversed(trace) if sample.get("cache")), {})
            grants = max(0, int(last.get("grants_total", 0) or 0) - int(first.get("grants_total", 0) or 0))
            refusals = max(
                0, int(last.get("refusals_total", 0) or 0) - int(first.get("refusals_total", 0) or 0)
            )
            max_active = max(
                (int(sample.get("cache", {}).get("slots_active", 0) or 0) for sample in trace), default=0
            )
            max_tps = max(
                (
                    float(detail.get("tokens_per_step", 0.0) or 0.0)
                    for sample in trace
                    for detail in (sample.get("models") or {}).values()
                ),
                default=0.0,
            )
            lines.append(
                f"| {family} | {concurrency} | {max_active} | {grants} | {refusals} | {round(max_tps, 3)} |"
            )
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


def whisper_scenario_recipe(args) -> str:
    """Shell recipe for the Phase 10-5 Whisper matrix (config-side cells)."""
    port = args.base_url.rsplit(":", 1)[-1]
    engine_log = args.engine_log or "out/bench/engine.log"
    log_flag = f" --engine-log {engine_log}"
    md_flag = " --md-out docs/benchmarks/throughput.md"
    lines = [
        "# Phase 10-5 Whisper scenarios. Every cell needs an idle benchmark host",
        "# (RTX 5060 Ti, CUDA_VISIBLE_DEVICES=1) and a server boot with that cell's",
        "# caps; the results are _tbd_ in docs/benchmarks/throughput.md §Phase 10.",
        "",
        "# Boot A - lanes only (batched_inference: false / unset), turbo num_workers=2",
        "CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=DEBUG \\",
        "    python -m harmonyspeech.endpoints.openai.api_server --config-file-path out/bench/config-phase10-lanes.yml \\",
        f"    --port {port} > {engine_log} 2>&1",
        "",
        "python scripts/bench_throughput.py --families whisper --concurrency 4,8,20 \\",
        f"    --requests 20 --label phase10-lanes-c4-8-20{log_flag}{md_flag}",
        "",
        "# Boot B - lanes + windows (batched_inference: true, window cap 8), same M",
        "CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=DEBUG \\",
        "    python -m harmonyspeech.endpoints.openai.api_server --config-file-path out/bench/config-phase10-windows.yml \\",
        f"    --port {port} > {engine_log} 2>&1",
        "",
        "python scripts/bench_throughput.py --families whisper --concurrency 4,8,20 \\",
        f"    --requests 20 --label phase10-lanes-windows-c4-8-20{log_flag}{md_flag}",
        "",
        "# 200-connection observation (client pool sized from the level)",
        "python scripts/bench_throughput.py --families whisper --concurrency 200 \\",
        f"    --requests 200 --label phase10-c200{log_flag}{md_flag}",
        "",
        "# Fairness proof: one >=10-min file first, then 20 short clips",
        "python scripts/bench_throughput.py --scenario whisper-fairness --label phase10-fairness \\",
        f"    --whisper-fairness-long-seconds 600{log_flag}{md_flag}",
        "",
        "# M-sweep: boot turbo with num_workers 1 / 2 / 4 and tiny with 1 / 4 / 6",
        "# (lanes only), then run each boot at c=8:",
        "python scripts/bench_throughput.py --families whisper --concurrency 8 --requests 20 \\",
        f"    --label phase10-turbo-m1{log_flag}{md_flag}",
        "",
        "# Window sweep: boot batched_inference true with the cap at 4 and 8, then:",
        "python scripts/bench_throughput.py --families whisper --concurrency 8 --requests 20 \\",
        f"    --label phase10-window-b4{log_flag}{md_flag}",
        "",
        "# Exhaustion: tight caps (M=2, window batch=4, queue 16), 100 heterogeneous",
        "# requests (2 s-8 min, mixed languages/tasks/formats). Set",
        "# whisper_max_audio_seconds: 480 on this boot so the probe validates the",
        "# 8-minute shape (the key is the probe's longest shape, not a rejection).",
        "python scripts/bench_throughput.py --scenario whisper-flood \\",
        "    --whisper-workers 2 --whisper-window-batch 4 --whisper-queue 16 \\",
        "    --whisper-flood-requests 100 --whisper-flood-concurrency 100 \\",
        f"    --label phase10-exhaustion{log_flag}{md_flag}",
        "",
        "# Memory-budget trip: boot with whisper_vram_budget_bytes below demand",
        "# (e.g. 6 GiB on a 16 GiB card), then run the same flood and record that",
        "# admission scales, latency degrades and no request is lost.",
        "python scripts/bench_throughput.py --scenario whisper-budget-trip \\",
        "    --whisper-vram-budget-bytes 6442450944 \\",
        "    --whisper-flood-requests 100 --whisper-flood-concurrency 100 \\",
        f"    --label phase10-budget-trip{log_flag}{md_flag}",
        "",
        "# Real-model spot-check (10-4): translation + srt/vtt through the wire",
        "python scripts/bench_throughput.py --scenario whisper-flood --whisper-flood-requests 8 \\",
        f"    --whisper-flood-concurrency 8 --label phase10-stt-formats{md_flag}",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Throughput benchmark against a running engine (Phase 5-3 / 7-3 / 8-6)"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:2242")
    parser.add_argument(
        "--families",
        default=DEFAULT_FAMILIES,
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
        "--scenario",
        choices=(
            "throughput",
            "whisper-flood",
            "whisper-fairness",
            "whisper-budget-trip",
            "chatterbox-ar",
            "chatterbox-exhaustion",
            "longform",
        ),
        default="throughput",
        help=(
            "throughput (default) runs the family matrix; whisper-flood / whisper-fairness / "
            "whisper-budget-trip run the Phase 10-5 exhaustion scenarios; chatterbox-ar runs "
            "the Phase 9-6 AR three-way comparison with pool-metrics sampling; "
            "chatterbox-exhaustion runs the tight-pool 100-request trace; longform runs "
            "the Phase 12-4 chunking on/off matrix"
        ),
    )
    parser.add_argument(
        "--whisper-flood-requests",
        type=int,
        default=WHISPER_FLOOD_REQUESTS,
        help="requests in the heterogeneous exhaustion flood (default 100)",
    )
    parser.add_argument(
        "--whisper-flood-concurrency",
        type=int,
        default=WHISPER_FLOOD_REQUESTS,
        help="client concurrency for the exhaustion flood (default: all requests at once)",
    )
    parser.add_argument(
        "--whisper-max-seconds",
        type=float,
        default=WHISPER_FLOOD_MAX_SECONDS,
        help="longest flood request in seconds (default 480 = 8 min)",
    )
    parser.add_argument(
        "--whisper-workers",
        type=int,
        default=WHISPER_FLOOD_WORKERS,
        help="declared M (num_workers) of the boot under test; recorded in the block",
    )
    parser.add_argument(
        "--whisper-window-batch",
        type=int,
        default=WHISPER_FLOOD_WINDOW_BATCH,
        help="declared window batch cap of the boot under test; recorded in the block",
    )
    parser.add_argument(
        "--whisper-queue",
        type=int,
        default=WHISPER_FLOOD_QUEUE,
        help="declared admission/queue depth (max_batch_size) of the boot under test",
    )
    parser.add_argument(
        "--whisper-vram-budget-bytes",
        type=int,
        default=None,
        help="whisper_vram_budget_bytes configured on the boot under test (required for the budget-trip run)",
    )
    parser.add_argument(
        "--whisper-fairness-long-seconds",
        type=float,
        default=WHISPER_FAIRNESS_LONG_SECONDS,
        help="duration of the pinned long-first request in the fairness proof (default 600)",
    )
    parser.add_argument(
        "--whisper-sample-interval",
        type=float,
        default=0.5,
        help="seconds between /metrics samples during a Whisper scenario (lane-depth trace)",
    )
    parser.add_argument(
        "--print-whisper-scenarios",
        action="store_true",
        help="print the Phase 10-5 Whisper recipe (boots, sweeps, exhaustion, budget trip) and exit",
    )
    parser.add_argument(
        "--print-ar-scenarios",
        action="store_true",
        help="print the Phase 9-6 Chatterbox AR recipe (three-way, cache matrix, exhaustion) and exit",
    )
    parser.add_argument(
        "--chatterbox-flood-requests",
        type=int,
        default=CHATTERBOX_FLOOD_REQUESTS,
        help="requests in the AR tight-pool exhaustion trace (default 100)",
    )
    parser.add_argument(
        "--chatterbox-flood-concurrency",
        type=int,
        default=CHATTERBOX_FLOOD_REQUESTS,
        help="client concurrency for the AR exhaustion trace (default: all at once)",
    )
    parser.add_argument(
        "--chatterbox-pool-slots",
        type=int,
        default=CHATTERBOX_FLOOD_POOL_SLOTS,
        help="declared slot count of the tight AR pool (recorded in the block)",
    )
    parser.add_argument(
        "--ar-sample-interval",
        type=float,
        default=0.5,
        help="seconds between /metrics AR-pool samples during a chatterbox-ar run",
    )
    parser.add_argument(
        "--mixed-lengths",
        action="store_true",
        help="rotate CHATTERBOX_TEXTS across AR payloads so rows leave the batch at different steps",
    )
    # Phase 12-4 — long-form scenarios.
    parser.add_argument(
        "--longform-split",
        choices=("off", "on", "both"),
        default=None,
        help=(
            "chunking variant(s) for --scenario longform: 'off' (single-shot), 'on' "
            "(split_text: true) or 'both' (the ON/OFF pair on the same boot). Default: "
            "'both' for the throughput matrix, 'on' for --fidelity"
        ),
    )
    parser.add_argument(
        "--longform-requests",
        type=int,
        default=LONGFORM_REQUESTS,
        help=f"requests per concurrency level in the long-form matrix (default {LONGFORM_REQUESTS})",
    )
    parser.add_argument(
        "--longform-chunk-size",
        type=int,
        default=None,
        help=(
            "explicit character budget sent in the ON payload and used for the expected "
            "chunks/request count. Default: omitted, so the serving layer resolves its "
            f"unified default ({LONGFORM_DEFAULT_CHUNK_SIZE} characters for every family)"
        ),
    )
    parser.add_argument(
        "--longform-crossfade-ms",
        type=int,
        default=None,
        help=(
            "crossfade_ms in the ON payload (default: omitted, so the serving "
            "layer's Phase 12-5 default of 40 ms applies; the H3 probe uses 50)"
        ),
    )
    parser.add_argument(
        "--longform-sentence-pause-ms",
        type=int,
        default=None,
        help=(
            "sentence_pause_ms in the ON payload (default: omitted, so the serving "
            "layer's Phase 12-5 default of 120 ms applies; the H3 probe uses 50)"
        ),
    )
    parser.add_argument(
        "--longform-remove-dc-offset",
        action="store_true",
        help="set remove_dc_offset on the long-form payload (H4 filter probe)",
    )
    parser.add_argument(
        "--longform-trim-silence",
        action="store_true",
        help="set trim_silence on the long-form payload (H4 filter probe)",
    )
    parser.add_argument(
        "--longform-fix-internal-silence",
        action="store_true",
        help="set fix_internal_silence on the long-form payload (H4 filter probe)",
    )
    parser.add_argument(
        "--print-longform-scenarios",
        action="store_true",
        help="print the Phase 12-4 long-form recipe (matrix, composition boot, gates) and exit",
    )
    # Phase 12-5 follow-up — Whisper text-fidelity gate.
    parser.add_argument(
        "--fidelity",
        action="store_true",
        help=(
            "with --scenario longform: render each family's fixture once per variant, "
            "transcribe the WAVs with faster-whisper and score word edit distance "
            "against the fixture (writes out/bench/fidelity-12/ by default)"
        ),
    )
    parser.add_argument(
        "--fidelity-out",
        default=FIDELITY_OUT_DIR,
        help=f"directory for fidelity renders/transcripts (default {FIDELITY_OUT_DIR})",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="run the fidelity transcriber on CUDA (default CPU int8)",
    )
    parser.add_argument(
        "--fidelity-model",
        default=FIDELITY_WHISPER_MODEL,
        help=f"faster-whisper model for --fidelity (default {FIDELITY_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--fidelity-language",
        default=FIDELITY_LANGUAGE,
        help=f"transcription language for --fidelity (default {FIDELITY_LANGUAGE})",
    )
    parser.add_argument(
        "--fidelity-beam",
        type=int,
        default=FIDELITY_BEAM_SIZE,
        help=f"transcription beam size for --fidelity (default {FIDELITY_BEAM_SIZE})",
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
    if args.print_whisper_scenarios:
        print(whisper_scenario_recipe(args))
        return 0
    if args.print_ar_scenarios:
        print(ar_scenario_recipe(args))
        return 0
    if args.print_longform_scenarios:
        print(longform_scenario_recipe(args))
        return 0

    if args.fidelity:
        if args.scenario != "longform":
            parser.error("--fidelity requires --scenario longform")
        report = asyncio.run(run_fidelity_benchmark(args))
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        table = _fidelity_markdown(report)
        if args.md_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
            with open(args.md_out, "a", encoding="utf-8") as handle:
                handle.write(table)
        print(table)
        failures = sum(
            1
            for row in report["families"].values()
            for variant in row["variants"].values()
            if variant.get("error")
        )
        return 0 if failures == 0 else 1

    if args.scenario == "longform":
        report = asyncio.run(run_longform_benchmark(args))
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        table = _longform_markdown(report)
        if args.md_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
            with open(args.md_out, "a", encoding="utf-8") as handle:
                handle.write(table)
        print(table)
        failures = sum(
            level["failures"] for levels in report["runs"].values() for level in levels
        )
        return 0 if failures == 0 else 1

    if args.scenario == "chatterbox-exhaustion":
        report = asyncio.run(run_chatterbox_exhaustion(args))
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        table = _ar_exhaustion_markdown(report)
        if args.md_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
            with open(args.md_out, "a", encoding="utf-8") as handle:
                handle.write(table)
        print(table)
        summary = report["summary"]
        return 0 if summary["failures"] == 0 and summary["request_loss"] == 0 else 1

    whisper_scenarios = {"whisper-flood", "whisper-fairness", "whisper-budget-trip"}
    if args.scenario in whisper_scenarios:
        report = asyncio.run(run_whisper_exhaustion(args))
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        table = _whisper_markdown(report)
        if args.md_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
            with open(args.md_out, "a", encoding="utf-8") as handle:
                handle.write(table)
        print(table)
        summary = report["summary"]
        return 0 if summary["failures"] == 0 and summary["request_loss"] == 0 else 1

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
