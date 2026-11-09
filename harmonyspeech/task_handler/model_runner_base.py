"""Model runners that execute prepared inputs against a loaded model.

Phase 0 invariants (engine modernization plan)
----------------------------------------------
* **No pickle execution for client payloads.** Every ``torch.load`` of request
  data uses ``weights_only=True`` (see :func:`_load_embedding_tensor`) and
  raises a typed ``ValueError`` on malformed/hostile payloads. The Chatterbox
  conditionals payload is a plain dict of tensors/primitives
  (:func:`_conditionals_to_payload`) and is read back through
  ``Conditionals.load`` (which is ``weights_only=True`` as well).
* **Per-request state isolation.** Chatterbox conditionals are still assigned
  to the shared model instance (upstream reads them from ``self.model.conds`` /
  ``self.model.ref_dict``), but every handler restores the baseline snapshot in
  a ``finally`` block, so one request can never inherit another's voice.
* **Batch pairing.** Every worker helper receives its ``EngineRequest`` as an
  explicit argument (no closure over the loop variable) and every handler
  asserts that inputs and requests are aligned.

Phase 2 invariants (artifact bus)
---------------------------------
* Handlers consume and produce typed artifacts
  (:mod:`harmonyspeech.common.artifacts`) — no text/JSON-float intermediates
  between pipeline steps. Output objects carry optional artifact slots that the
  serving layer serializes; ``output`` stays empty for artifact-carrying results.
* Every cross-device move is explicit (``artifact.to(self.device)``) and the
  source device is asserted first (:func:`_bind_to_device`), so a missing device
  tag fails loudly instead of silently copying in the wrong place.
* Chatterbox conditionals travel as object references
  (``SpeakerArtifact.conditionals``); the consuming handler moves them onto its
  own device. No pickle round trip on internal hops.

Phase 5 invariants (honest batching)
------------------------------------
* **Type-homogeneous batches.** ``execute_model`` partitions by request-input
  class *before* ``prepare_inputs``, so one prepare + one handler call happen per
  ``(model, request type)`` partition and every handler sees aligned lists.
* **Alignment is a typed error.** ``assert_batch_aligned`` raises
  ``EngineBatchError``; the runner converts it into per-request error results.
* **One result per request per step.** ``_build_result`` refuses to build a
  second result for the same ``request_id`` inside one step; a pairing
  regression surfaces immediately instead of silently duplicating an output.
* **Real padded batches.** The one-shot synthesis families (OpenVoice V1,
  MeloTTS) forward a padded ``[B, T_max]`` batch per step when
  ``batched_inference`` is enabled (default), with length-similar sub-batching
  and a per-model token budget; the per-item loop remains the ``len == 1`` fast
  path and the fallback. Phase 8-2 extends the same machinery to the speaker
  embedders: the OpenVoice tone-converter encoder forwards its
  ``(request, VAD segment)`` rows as one padded mel batch per sub-batch (rows
  masked per layer, GRU packed over each row's valid frames), and the
  HarmonySpeech encoder forwards its fixed-length ``(request, partial)`` rows in
  one LSTM batch. Embedding rows are always reassembled per request in input
  order, and cache hits never enter a batch (Phase 7-2). Phase 8-3 adds the
  tone converters: one ``(request)`` row of per-clip mel frames per padded
  sub-batch, unpadded in the sample domain with the model's own frame counts,
  with per-sub-batch loop fallback.
* **Parallel lanes for non-batchable steps (Phase 8-5).** KittenTTS has a static
  batch dimension of 1 and Silero VAD is a single-stream API, so those steps run
  one request per lane instead (``max_batch_size`` lanes over one shared
  session/serialised phonemizer, per-request error isolation, results in input
  order). Silero lanes measured below break-even and stay sequential; the numbers
  and the mechanism are documented in ``docs/batching.md``.

Phase 10 invariants (Whisper serving)
-------------------------------------
* **M-worker dispatch, not an unbounded thread pool.** A Whisper step runs at
  most ``num_workers`` concurrent ``transcribe()`` calls through the runner's
  own :class:`WhisperLanePool`; a semaphore makes the bound real even if two
  callers race. ``num_workers`` is CT2 ``inter_threads`` (shared weights).
* **Generators are consumed inside the worker.** The segment generator is
  iterated in the worker thread, so compute time and exceptions belong to that
  request; nothing is handed back to the engine half-consumed.
* **Cancellation is cooperative at segment granularity.** faster-whisper has no
  mid-``generate`` abort, so the worker checks the request state between
  segments. An aborted request errors alone; its neighbours complete.
* **Cross-request window batches (Phase 10-2).** A step with more than one
  request folds every admitted request's ≤30 s windows into one CT2
  ``encode`` + one ``generate`` per formed batch (≤8 windows), with per-request
  prompts and segment stitching. ``len == 1``, ``batched_inference: false`` and
  any batch-level batcher failure keep the per-request pipeline path; a single
  poisoned request/window errors alone.
"""

import contextlib
import copy
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import torch
from faster_whisper import BatchedInferencePipeline
from loguru import logger
from silero_vad import get_speech_timestamps

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.device import resolve_device
from harmonyspeech.common.errors import EngineBatchError
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput
from harmonyspeech.common.outputs import (
    AudioConversionRequestOutput,
    DetectVoiceActivityRequestOutput,
    RequestOutput,
    SpeechEmbeddingRequestOutput,
    SpeechSynthesisRequestOutput,
    SpeechTranscriptionRequestOutput,
    TextToSpeechRequestOutput,
    UnfinishedRequestOutput,
    VocodeRequestOutput,
    VoiceConversionRequestOutput,
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult

# Re-exported for tests and callers that used the Phase 0 helper names; the
# implementations live in ``common.serialization`` (weights_only=True loaders).
from harmonyspeech.common.serialization import conditionals_to_payload as _conditionals_to_payload  # noqa: F401
from harmonyspeech.common.serialization import load_embedding_tensor as _load_embedding_tensor  # noqa: F401
from harmonyspeech.engine.whisper_lane_metrics import WHISPER_QUEUE_WAIT_BUCKETS
from harmonyspeech.modeling.loader import get_model, get_model_config, get_model_flavour
from harmonyspeech.modeling.models.harmonyspeech.common import preprocess_wav
from harmonyspeech.modeling.models.harmonyspeech.encoder.inputs import get_input_frames
from harmonyspeech.modeling.models.openvoice.mel_processing import spectrogram_torch
from harmonyspeech.task_handler.batching import (
    DEFAULT_MEL_FRAME_BATCH_BUDGET,
    DEFAULT_PHONE_BATCH_BUDGET,
    ar_batching_enabled,
    batched_inference_enabled,
    collapse_per_sample,
    default_token_budget,
    lane_count,
    pad_feature_batch,
    pad_phone_batch,
    plan_batches,
    run_in_lanes,
    trim_decoder_output,
    trim_feature_output,
)
from harmonyspeech.task_handler.inputs import (
    as_request_input,
    get_audio_artifact,
    get_vad_artifact,
    prepare_inputs,
)
from harmonyspeech.task_handler.whisper_batching import (
    WHISPER_WINDOW_BATCH_SIZE,
    WhisperRequestAbortedError,
    WhisperWindowBatcher,
    whisper_batched_inference_enabled,
    whisper_request_options,
)


def assert_batch_aligned(handler_name: str, inputs: list, requests: list) -> None:
    """Assert that prepared inputs and requests are positionally aligned.

    Every handler calls this before touching position ``i``. A violation is an
    :class:`~harmonyspeech.common.errors.EngineBatchError`: the model runner
    converts it into typed per-request error results (Phase 5-1), so a pairing
    regression becomes an HTTP error for the affected requests instead of an
    unhandled crash in the engine's background loop.
    """
    if len(inputs) != len(requests):
        raise EngineBatchError(f"{handler_name}: {len(inputs)} inputs vs {len(requests)} requests")


def _require_aligned(inputs, requests_to_batch, handler_name: str) -> None:
    """Backwards-compatible alias for :func:`assert_batch_aligned`."""
    assert_batch_aligned(handler_name, inputs, requests_to_batch)


def _bind_to_device(artifact, device):
    """Assert an artifact carries its source device, then move it explicitly.

    This is the single place handlers cross the device boundary, which makes a
    second (accidental) transfer or an untagged artifact fail loudly instead of
    silently degrading throughput.
    """
    if artifact is None:
        return None
    current = getattr(artifact, "device", None)
    if current is None:
        raise AssertionError(f"{type(artifact).__name__} is missing its device tag")
    return artifact.to(device)


def _wav_buffer(waveform_artifact) -> io.BytesIO:
    """Re-wrap a decoded waveform as in-memory WAV bytes for file-like APIs.

    Chatterbox's upstream ``prepare_conditionals`` / ``set_target_voice`` take a
    file-like object. The artifact is already decoded, so this is an encode-only
    step (no wire encoding, no client-container decode, no tempfile) — it
    replaces the Phase 1 ``ensure_librosa_readable_bytes`` path. The move to CPU
    is explicit (WAV encoding needs host samples).
    """
    return io.BytesIO(_bind_to_device(waveform_artifact, "cpu").to_wav_bytes())


def _chatterbox_conditionals(speaker: SpeakerArtifact | None, device) -> object | None:
    """Return device-local Chatterbox conditionals, moving them explicitly."""
    if speaker is None:
        return None
    conditionals = getattr(speaker, "conditionals", None)
    if conditionals is None:
        return None
    move = getattr(conditionals, "to", None)
    if callable(move):
        return move(device)
    return conditionals


def _audio_artifact_from_tensor(wav: torch.Tensor, sample_rate: int) -> AudioArtifact:
    """Wrap a model-generated waveform tensor as an :class:`AudioArtifact`."""
    tensor = wav.detach().to("cpu", dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return AudioArtifact(waveform=tensor, sample_rate=sample_rate)


class _NullWatermarker:
    """Identity watermarker used while a long-form chunk is synthesized.

    The serving layer (Phase 12-2) watermarks the *stitched* buffer exactly
    once; watermarking each chunk and then crossfading would produce a muddled
    mark. Only the per-request window of the sequential synthesis call is
    covered, so a concurrent request in another runner is unaffected.
    """

    def apply_watermark(self, wav, sample_rate=None):  # noqa: ARG002 - upstream signature
        return wav


class _FilteredS3Gen:
    """Per-wrapper proxy that drops OOV speech tokens before an S3Gen vocode.

    Phase 12-7: upstream ``ChatterboxMultilingualTTS.generate``
    (``chatterbox/mtl_tts.py:290-294``) omits the
    ``speech_tokens = speech_tokens[speech_tokens < 6561]`` filter that English
    ``generate`` applies, so an OOV token can crash the flow embedding. This
    proxy is installed on the *multilingual wrapper's* ``s3gen`` attribute only
    for the sequential call (the shared module is never mutated), and delegates
    every other attribute to it.
    """

    def __init__(self, s3gen, max_token: int = 6561):
        object.__setattr__(self, "_s3gen", s3gen)
        object.__setattr__(self, "_max_token", int(max_token))

    def __getattr__(self, name):
        return getattr(self._s3gen, name)

    def inference(self, speech_tokens=None, *args, **kwargs):
        if speech_tokens is not None:
            speech_tokens = speech_tokens[speech_tokens < self._max_token]
        return self._s3gen.inference(*args, speech_tokens=speech_tokens, **kwargs)


def _watermark_suppressed(request) -> bool:
    """True when this request's runner must not watermark its own slice."""
    request_data = getattr(request, "request_data", request)
    return bool(getattr(request_data, "suppress_watermark", False))


def _pad_mel_batch(mels, device):
    """Pad ``[1, C, T, F]`` mel maps on the time axis; returns ``(padded, lengths)``.

    The frame-axis sibling of :func:`~harmonyspeech.task_handler.batching.pad_feature_batch`
    for the 4-D maps the VoiceFixer restorer exchanges (batch, channel, frames,
    bins). Padding is digital silence in the *mel* domain (zeros), which is what
    the vendored UNet's own ``F.pad`` uses past a clip's last frame.
    """
    if not mels:
        raise ValueError("_pad_mel_batch: empty mel list")
    channels = int(mels[0].shape[1])
    bins = int(mels[0].shape[3])
    lengths = [int(mel.shape[2]) for mel in mels]
    padded = torch.zeros(
        (len(mels), channels, max(lengths), bins), dtype=mels[0].dtype, device=device
    )
    for row, mel in enumerate(mels):
        if int(mel.shape[1]) != channels or int(mel.shape[3]) != bins:
            raise ValueError(
                f"_pad_mel_batch: inconsistent mel shape {tuple(mel.shape)} vs "
                f"{(channels, bins)} channels/bins"
            )
        padded[row, :, : lengths[row]] = mel[0].to(device)
    return padded, torch.tensor(lengths, dtype=torch.long, device=device)


def _clone_tensor(value):
    """Clone a tensor as a detached leaf, or pass a non-tensor through."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tensor(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensor(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor(item) for item in value)
    return value


def _clone_conditionals(conds):
    """Clone Chatterbox ``Conditionals`` (dataclass of ``T3Cond`` + gen dict).

    Deep-copying the upstream dataclass fails when a checkpoint tensor carried
    ``requires_grad`` (``.to(device)`` then yields a non-leaf tensor). Field-wise
    detach+clone keeps the baseline snapshot safe and cheap. Plain dict/list
    stand-ins (used by tests and future wrappers) are cloned recursively so the
    baseline never aliases live request state.
    """
    if conds is None:
        return None
    if isinstance(conds, (dict, list, tuple)):
        return _clone_tensor(conds)
    t3 = getattr(conds, "t3", None)
    gen = getattr(conds, "gen", None)
    if t3 is None and gen is None and not hasattr(conds, "__dict__"):
        return _clone_tensor(conds)
    t3_clone = None
    if t3 is not None:
        try:
            t3_clone = type(t3)(**{key: _clone_tensor(value) for key, value in vars(t3).items()})
        except TypeError:  # pragma: no cover - future upstream field changes
            t3_clone = t3
    gen_clone = {key: _clone_tensor(value) for key, value in (gen or {}).items()}
    try:
        return type(conds)(t3_clone, gen_clone)
    except TypeError:  # pragma: no cover
        return _clone_tensor(conds)


def _finalize_encoder_embedding(partial_embeds) -> SpeakerArtifact:
    """Per-request HarmonySpeech embedding: mean over partials, then L2-norm.

    Shared by the per-item loop and the padded-batch path (Phase 8-2) so both
    compute the historical ``np.mean`` → normalise arithmetic on a
    ``[partials, embedding]`` array, in partial order.
    """
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    return SpeakerArtifact(
        embedding=torch.from_numpy(np.ascontiguousarray(embed, dtype=np.float32)),
        backend="harmonyspeech",
    )


#: faster-whisper's intra-audio clip batch: one request's audio is cut into
#: <=30 s clips and forwarded in groups of this size. Unrelated to
#: ``max_batch_size`` (the scheduler's admission width) and to ``num_workers``
#: (how many requests transcribe at once).
WHISPER_CLIP_BATCH_SIZE = 16


# ``WhisperRequestAbortedError`` is defined in ``whisper_batching`` (Phase 10-2)
# and re-exported here for the Phase 10-1 callers/tests that import it from this
# module. Its contract is unchanged: cancellation is cooperative (segment
# granularity in the lane pool, batch granularity in the window batcher).


def _request_finished(request) -> bool:
    """True when the engine already aborted/finished ``request``."""
    is_finished = getattr(request, "is_finished", None)
    if not callable(is_finished):
        return False
    try:
        return bool(is_finished())
    except Exception:  # pragma: no cover - defensive on odd request stand-ins
        return False


def _ct2_queue_state(model) -> dict:
    """Read CTranslate2's native batch counters when the binding exposes them.

    CT2 4.4.0's ``Whisper`` exposes ``num_queued_batches`` /
    ``num_active_batches`` (verified 2026-11-07); older/newer bindings may not,
    so an absent or unreadable counter reports ``available: False`` with zeros
    instead of raising inside a metrics scrape.
    """
    native = getattr(model, "model", None)
    queued_attr = getattr(native, "num_queued_batches", None)
    active_attr = getattr(native, "num_active_batches", None)
    available = queued_attr is not None or active_attr is not None

    def _int(value) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):  # pragma: no cover - odd binding state
            return 0

    return {"available": available, "queued": _int(queued_attr), "active": _int(active_attr)}


class WhisperLanePool:
    """M-worker dispatch for one Whisper model (Phase 10-1).

    Owns a **persistent** ``ThreadPoolExecutor(max_workers=M)`` and a
    ``threading.BoundedSemaphore(M)``. M is the model's ``num_workers`` (the
    value faster-whisper forwards to CTranslate2 ``inter_threads``), never an
    executor/anyio default: the point is to run exactly as many transcribes as
    the card was sized for.

    Each pool thread keeps its own ``BatchedInferencePipeline`` — a thin view
    over the shared model that carries a word-timestamp cursor — so the
    per-request cursor reset stays race-free while the CT2 weights stay shared.

    Cancellation is cooperative at segment granularity: the worker checks the
    request's engine state between segments and raises
    :class:`WhisperRequestAbortedError`, which the handler turns into an error result
    for that request only.

    The pool is created lazily so ``ModelRunnerBase.__new__``-style test runners
    (which bypass ``__init__``) keep working.
    """

    def __init__(self, model, num_workers: int = 1, clip_batch_size: int = WHISPER_CLIP_BATCH_SIZE) -> None:
        self.model = model
        self.num_workers = max(1, int(num_workers or 1))
        self.clip_batch_size = max(1, int(clip_batch_size or WHISPER_CLIP_BATCH_SIZE))
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(self.num_workers)
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._queue_wait_sum = 0.0
        self._queue_wait_count = 0
        self._queue_wait_max = 0.0
        self._queue_wait_buckets = [0] * (len(WHISPER_QUEUE_WAIT_BUCKETS) + 1)

    # -- pool lifecycle --------------------------------------------------
    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is not None:
            return self._pool
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.num_workers,
                    thread_name_prefix="whisper-worker",
                )
            return self._pool

    def close(self) -> None:
        """Drop the worker pool (model reload / runner teardown)."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False)

    def _pipeline(self):
        """Return this thread's pipeline, creating it on first use."""
        pipeline = getattr(self._local, "pipeline", None)
        if pipeline is None:
            pipeline = BatchedInferencePipeline(model=self.model)
            self._local.pipeline = pipeline
        return pipeline

    # -- accounting ------------------------------------------------------
    def _enter(self) -> None:
        with self._in_flight_lock:
            self._in_flight += 1

    def _leave(self) -> None:
        with self._in_flight_lock:
            self._in_flight -= 1

    def record_queue_wait(self, seconds) -> None:
        """Record one request's admission-to-scheduling wait (histogram)."""
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return
        if value < 0:
            value = 0.0
        with self._stats_lock:
            self._queue_wait_sum += value
            self._queue_wait_count += 1
            self._queue_wait_max = max(self._queue_wait_max, value)
            for index, edge in enumerate(WHISPER_QUEUE_WAIT_BUCKETS):
                if value <= edge:
                    self._queue_wait_buckets[index] += 1
                    break
            else:
                self._queue_wait_buckets[-1] += 1

    def stats(self) -> dict:
        with self._in_flight_lock:
            in_flight = self._in_flight
        with self._stats_lock:
            queue_wait = {
                "sum": self._queue_wait_sum,
                "count": self._queue_wait_count,
                "max": self._queue_wait_max,
                "buckets": list(self._queue_wait_buckets),
            }
        return {
            "workers": self.num_workers,
            "in_flight": in_flight,
            "queue_wait": queue_wait,
            "ct2": _ct2_queue_state(self.model),
        }

    # -- execution -------------------------------------------------------
    def transcribe(self, audio_artifact, initial_request):
        """Transcribe one request, consuming the generator fully in this thread.

        Returns ``(json_payload, vad_artifact)``. The timestamp cursor reset is
        preserved per invocation (Phase 0-2/8-5): the pipeline instance is
        per-thread, and the cursor is zeroed before every call so one request's
        segments can never shift the next request's timestamps.
        """
        if _request_finished(initial_request):
            raise WhisperRequestAbortedError(
                f"request {getattr(initial_request, 'request_id', '?')} was aborted before transcription"
            )
        pipeline = self._pipeline()
        # faster-whisper takes a mono numpy array; the artifact already holds the
        # boundary-decoded 16 kHz waveform.
        audio_ref = _bind_to_device(audio_artifact, "cpu").numpy
        # Task / source-language hint / initial prompt travel on the request
        # input (Phase 10-4). `task` selects the `<|transcribe|>` or
        # `<|translate|>` prompt token inside faster-whisper's tokenizer, exactly
        # as the window batcher does.
        task, language, prompt = whisper_request_options(initial_request)
        pipeline.last_speech_timestamp = 0.0
        segments, info = pipeline.transcribe(
            audio_ref,
            batch_size=self.clip_batch_size,
            task=task,
            language=language,
            initial_prompt=prompt,
        )

        text = ""
        segment_data = []
        # Consume the generator *here*, inside the worker, so the GPU time lands
        # in this worker's accounting (and so an exception surfaces in this
        # request's result, not in a shared step).
        for segment in segments:
            if _request_finished(initial_request):
                raise WhisperRequestAbortedError(
                    f"request {getattr(initial_request, 'request_id', '?')} was aborted during transcription"
                )
            text += segment.text
            segment_data.append(asdict(segment))

        response = {"text": text, "segments": segment_data, "info": asdict(info)}
        return json.dumps(response), VadArtifact.from_payload(response)

    def _run_task(self, function, audio_artifact, initial_request):
        """Run one transcribe under the M-bound semaphore, isolating failures."""
        with self._semaphore:
            self._enter()
            try:
                metrics = getattr(initial_request, "metrics", None)
                wait = getattr(metrics, "time_in_queue", None)
                if wait is not None:
                    self.record_queue_wait(wait)
                # `execute_model` runs under torch.inference_mode(), but that is
                # thread-local; re-enter it in the worker thread.
                with torch.inference_mode():
                    return function(audio_artifact, initial_request), None
            except Exception as error:  # noqa: BLE001 - becomes this request's error result
                return None, error
            finally:
                self._leave()

    def run(self, inputs, requests_to_batch, function) -> list[tuple]:
        """Run ``function(input, request)`` per request, at most M at a time.

        Returns one ``(payload, error)`` pair per request **in input order**.
        ``len(inputs) == 1`` is a direct call on the calling thread (the fast
        path, and the fallback); larger steps go through the owned pool.
        """
        if len(inputs) == 1:
            return [self._run_task(function, inputs[0], requests_to_batch[0])]
        pool = self._ensure_pool()
        futures = [
            pool.submit(self._run_task, function, audio_artifact, request)
            for audio_artifact, request in zip(inputs, requests_to_batch, strict=True)
        ]
        results: list[tuple] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - isolate an unexpected worker failure
                results.append((None, error))
        return results


class ModelRunnerBase:
    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        is_driver_worker: bool = False,
        *args,
        weight_registry=None,
        embedding_cache=None,
        ar_cache_pool=None,
        **kwargs,
    ):
        self.model_config = model_config
        self.is_driver_worker = is_driver_worker
        # Engine-owned shared-weight registry (Phase 4-3); optional so tests and
        # standalone runners keep working without one.
        self.weight_registry = weight_registry
        # Engine-owned cross-request embedding cache (Phase 7-1); None means
        # disabled — every handler guards once at entry and then behaves
        # exactly like pre-Phase-7. Also optional for tests/standalone runners.
        self.embedding_cache = embedding_cache
        # Engine-owned shared AR KV-cache pool (Phase 9-2); None means disabled.
        # 9-3's AR runner registers its descriptor on first load and acquires one
        # slot per in-flight request; a refused grant falls back to the
        # sequential loop. Optional for tests/standalone runners.
        self.ar_cache_pool = ar_cache_pool
        # Cache identity (name/type/dtype/repo + load_version); built on every
        # model load by `_refresh_model_identity`. None until the first load.
        self.model_identity: str | None = None

        self.device_config = device_config if device_config is not None else DeviceConfig()
        # Canonical indexed device (Phase 11-0): ``cuda`` -> ``cuda:<active>``,
        # explicit ``cuda:1`` preserved. Every handler's ``.to(self.device)``
        # then lands on the model's real device instead of whichever device the
        # process happens to have active.
        self.device = resolve_device(self.device_config.device)

        self.model = None
        self.model_memory_usage = 0
        # Startup max-shape probe record for a CTranslate2 Whisper config
        # (Phase 10-3). `None` means "not a whisper runner or not probed";
        # `get_vram_stats` renders it as an explicit "not probed" annotation.
        self.whisper_probe: dict | None = None
        # M-worker dispatch pool for a Whisper config (Phase 10-1), created on
        # first use; None until then (and for every non-Whisper runner).
        self._whisper_pool: WhisperLanePool | None = None
        # Cross-request window batcher (Phase 10-2), created on first use, and
        # the count of batcher failures that fell back to the sequential path.
        self._whisper_batcher_obj: WhisperWindowBatcher | None = None
        self._whisper_batch_fallbacks = 0

        # Baseline Chatterbox conditioning state, captured right after the
        # model is loaded (see `_capture_baseline_state`). None means "not
        # captured yet"; `_restore_chatterbox_state` then treats the baseline
        # as "no conditionals".
        self._baseline_state: dict | None = None
        # Phase 9-3 token-level AR decode runner for the Chatterbox English/Turbo
        # roles; created at model load (`_init_ar_runner`) when the pool and the
        # family flag allow it. None keeps the sequential loop.
        self._ar_runner = None

    def _load_model(self):
        # Every load (first boot or reload) starts a fresh cache generation:
        # the per-name load counter bumps, keys derived under the previous
        # weight load can no longer be produced, and the old identity's entries
        # are dropped explicitly (Phase 7-1).
        self._refresh_model_identity()
        return get_model(self.model_config, self.device_config, weight_registry=self.weight_registry)

    # ------------------------------------------------------------------
    # Cross-request embedding cache (Phase 7-2)
    # ------------------------------------------------------------------

    def _refresh_model_identity(self) -> None:
        """(Re)build the embedding-cache identity, bumping the load version."""
        cache = getattr(self, "embedding_cache", None)
        previous = getattr(self, "model_identity", None)
        if cache is None:
            self.model_identity = None
            return
        if previous is not None:
            cache.invalidate_model(previous)
        config = self.model_config
        version = cache.bump_load_version(config.name)
        self.model_identity = "|".join(
            (
                config.name,
                str(config.model_type),
                str(config.dtype),
                f"{config.model}@{config.revision or 'main'}",
                f"v{version}",
            )
        )

    def _embedding_cache_lookup(self, audio_artifact, params: dict) -> tuple:
        """Resolve one cache slot for a request: ``(key, cached)``.

        ``(None, None)`` means "no cache / not usable" — the caller computes
        and (when a key exists) stores. A cached serve is a per-request deep
        copy (Phase 0-2 isolation).
        """
        cache = getattr(self, "embedding_cache", None)
        if cache is None or audio_artifact is None or self.model_identity is None:
            return None, None
        key = cache.make_key(
            model_identity=self.model_identity,
            audio_hash=audio_artifact.content_hash,
            params=params,
        )
        return key, cache.get(key)

    @staticmethod
    def _chatterbox_embedding_params(initial_request) -> dict:
        """Chatterbox cache params: ``exaggeration`` bakes into conditionals.

        Read from the request's generation options (cloning TTS requests carry
        them; standalone embed requests default to the model's 0.5).
        """
        options = getattr(getattr(initial_request, "request_data", None), "generation_options", None)
        exaggeration = getattr(options, "exaggeration", None)
        return {"exaggeration": float(exaggeration) if exaggeration is not None else 0.5}

    @staticmethod
    def _openvoice_encoder_params(initial_request) -> dict:
        """OpenVoice TCE cache params: the VAD source decides the segment set.

        Whisper and Silero produce different segments for the same audio
        (Phase 3-4), so the key must never conflate them.
        """
        vad = get_vad_artifact(as_request_input(initial_request), initial_request, "input_vad_data")
        return {"vad_source": vad.source or "unknown"}

    def _store_embedding_cache_entry(self, key, artifact) -> None:
        """Put a computed artifact, tagged with this runner's identity.

        Storing is best effort: the computation already succeeded, so a cache
        failure (e.g. an unclonable artifact) is logged and dropped instead of
        failing the request.
        """
        cache = getattr(self, "embedding_cache", None)
        if cache is None or key is None:
            return
        try:
            cache.put(key, artifact, model_identity=self.model_identity)
        except Exception as e:  # noqa: BLE001 - the cache must never fail serving
            logger.debug(f"Embedding cache store failed for {self.model_identity}: {e}")

    @staticmethod
    def _resolve_cache_slots(handler_cache, requests_to_batch, resolver) -> list:
        """Per-request cache resolution for one batch (hits before any compute).

        ``resolver(i)`` returns ``(key, cached)`` for request ``i``. A
        resolution failure can never fail the request — the cache is an
        optimization, so the slot degrades to "miss, do not store".
        """
        slots: list = [(None, None)] * len(requests_to_batch)
        if handler_cache is None:
            return slots
        for index in range(len(requests_to_batch)):
            try:
                slots[index] = resolver(index)
            except Exception:  # noqa: BLE001 - cache resolution must not fail requests
                slots[index] = (None, None)
        return slots

    def _snapshot_chatterbox_state(self) -> dict:
        """Snapshot the model's conditionals so they can be restored later.

        Only Chatterbox-family models expose ``conds`` / ``ref_dict``; other
        models yield an empty snapshot and are never touched again.

        Tensors are detached and cloned field-by-field instead of
        ``copy.deepcopy`` on the whole object: a checkpoint tensor that carries
        ``requires_grad`` becomes a non-leaf when moved to CUDA, and non-leaf
        tensors refuse the deepcopy protocol.
        """
        state: dict = {}
        model = self.model
        if model is None:
            return state
        if hasattr(model, "conds"):
            state["conds"] = _clone_conditionals(model.conds)
        if hasattr(model, "ref_dict"):
            ref_dict = model.ref_dict
            state["ref_dict"] = (
                {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in ref_dict.items()
                }
                if isinstance(ref_dict, dict)
                else ref_dict
            )
        return state

    def _capture_baseline_state(self) -> None:
        """Capture the post-load conditioning baseline (call after loading)."""
        self._baseline_state = self._snapshot_chatterbox_state()

    def _restore_chatterbox_state(self) -> None:
        """Restore the conditioning baseline captured at model load.

        This runs in a ``finally`` block of every Chatterbox handler, so a
        request can never inherit the previous request's voice (Phase 0-2).
        """
        state = getattr(self, "_baseline_state", None) or {}
        model = self.model
        if model is None:
            return
        if hasattr(model, "conds"):
            model.conds = state.get("conds")
        if hasattr(model, "ref_dict"):
            model.ref_dict = state.get("ref_dict")

    def _align_chatterbox_conditionals(self, exaggeration: float | None = None) -> None:
        """Cast per-request conditionals to the T3 weight dtype (Phase 4-1).

        Upstream ``Conditionals.to(device)`` only moves tensors; when the T3
        backbone runs in float16 the per-request ``speaker_emb`` / emotion
        tensors must match or the first conditional projection raises a dtype
        mismatch. This lives here (not on the shared backbone) because
        conditionals are per-executor state (Phase 0-2).

        Setting ``emotion_adv`` to the requested exaggeration up front also
        prevents upstream ``generate`` from rebuilding an fp32 ``T3Cond`` for
        the "exaggeration changed" branch.
        """
        model = self.model
        if model is None:
            return
        conds = getattr(model, "conds", None)
        t3 = getattr(model, "t3", None)
        t3_cond = getattr(conds, "t3", None)
        if t3_cond is None or t3 is None:
            return
        dtype = None
        for param in t3.parameters():
            dtype = param.dtype
            break
        if dtype is None:
            return
        device = getattr(model, "device", None) or str(self.device)
        conds.t3 = t3_cond.to(device=device, dtype=dtype)
        if exaggeration is not None and getattr(conds.t3, "emotion_adv", None) is not None:
            conds.t3.emotion_adv = exaggeration * torch.ones(1, 1, 1, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Token-level AR decode runner (Phase 9-3)
    # ------------------------------------------------------------------
    def _init_ar_runner(self):
        """Create + register the AR runner for a Chatterbox English/Turbo model.

        Registration happens here (right after model load) so the engine's
        startup arena table lists the arena, and the arena is keyed by the
        scheduler's model name (9-4's capacity-admission convention). Returns
        ``None`` for every other model, when the family flag is off, or when the
        pool refuses the descriptor — the sequential loop then remains the path.
        """
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "t3"):
            self._ar_runner = None
            return None
        if not ar_batching_enabled(self.model_config):
            self._ar_runner = None
            return None
        from harmonyspeech.task_handler.ar_runner import ARBatchRunner

        runner = ARBatchRunner(
            t3=model.t3,
            pool=getattr(self, "ar_cache_pool", None),
            model_key=self.model_config.name,
            model_config=self.model_config,
        )
        runner.register()
        self._ar_runner = runner
        return runner

    def _get_ar_runner(self):
        """Return the AR runner, creating it lazily for shell/test runners."""
        runner = getattr(self, "_ar_runner", None)
        if runner is None:
            runner = self._init_ar_runner()
        return runner

    def release_request(self, request_id: str) -> None:
        """Engine hook (Phase 9-4): drop an AR row that will never run again.

        Called on abort/disconnect and deadline expiry mid-decode. A runner
        without an active row for ``request_id`` is a no-op.
        """
        runner = getattr(self, "_ar_runner", None)
        if runner is not None:
            runner.release_request(request_id)

    def _ar_conditionals(self, params, initial_request, *, turbo: bool, multilingual: bool = False):
        """Resolve per-request conditionals without leaving them on the model.

        Pre-computed conditionals (the embedding-step hop) are used by
        reference. A request that clones from raw audio goes through upstream's
        ``prepare_conditionals`` and the baseline object is restored immediately:
        the AR decode never reads or writes ``model.conds`` (the conditionals
        live on the runner's row), so the object identity the Phase 0-2 restore
        protects is unchanged for the whole decode. A request with no
        conditioning uses a copy of the model's bundled default voice.
        """
        from harmonyspeech.task_handler.ar_runner import cast_conditionals

        if turbo:
            _text, speaker, _temperature, _repetition, _top_p, _top_k, norm_loudness = params
            exaggeration = None
        elif multilingual:
            (
                _text,
                _language_id,
                speaker,
                exaggeration,
                _cfg,
                _temperature,
                _repetition,
                _top_p,
                _min_p,
            ) = params
            norm_loudness = None
        else:
            _text, speaker, exaggeration, _cfg, _temperature, _repetition, _top_p, _min_p = params
            norm_loudness = None

        conditionals = _chatterbox_conditionals(speaker, self.device)
        if conditionals is None and speaker is not None and speaker.reference_audio is not None:
            previous = getattr(self.model, "conds", None)
            try:
                if turbo:
                    self.model.prepare_conditionals(
                        _wav_buffer(speaker.reference_audio), norm_loudness=norm_loudness
                    )
                else:
                    self.model.prepare_conditionals(
                        _wav_buffer(speaker.reference_audio), exaggeration=exaggeration
                    )
                conditionals = self.model.conds
            finally:
                if hasattr(self.model, "conds"):
                    self.model.conds = previous
        if conditionals is None:
            baseline = (getattr(self, "_baseline_state", None) or {}).get("conds")
            if baseline is None and getattr(self.model, "conds", None) is None:
                raise ValueError("voice cloning requires input_audio or a precomputed embedding/conditionals")
            source = baseline if baseline is not None else getattr(self.model, "conds", None)
            conditionals = _clone_conditionals(source)
        # Phase 12-8: clone the T3 half before the dtype move and apply
        # `exaggeration`, so a cached/shared SpeakerArtifact is never mutated.
        return cast_conditionals(conditionals, self.model.t3, exaggeration=exaggeration)

    def _execute_chatterbox_tts_ar(self, runner, inputs, requests_to_batch, *, turbo: bool, multilingual: bool = False):
        """One batched AR decode step for the requests in this partition.

        New rows are prefilled and then sample exactly one token, so a freshly
        admitted row returns immediately. A continuation step (no newly admitted
        rows) decodes up to ``runner.step_tokens`` tokens in this one engine
        invocation, stopping immediately when any row finishes or errors
        (Phase 12-8). A row that
        hits EOS/the token cap releases its KV slots immediately but
        its S3Gen vocode is **held** until every active row in the batch finishes
        T3 decoding (Phase 12-8), so staggered-EOS chunks vocode together in one
        padded ``inference_batched`` call instead of stalling active rows.
        ``len == 1`` for a *new* request and a refused slot fall back to the
        sequential loop for that request only.
        """
        from harmonyspeech.task_handler.ar_runner import (
            english_text_tokens,
            multilingual_text_tokens,
            turbo_text_tokens,
        )

        if turbo:
            sequential = self._execute_chatterbox_turbo_tts_sequential
        elif multilingual:
            sequential = self._execute_chatterbox_multilingual_tts_sequential
        else:
            sequential = self._execute_chatterbox_tts_sequential

        def _text_tokens_for(model, params):
            if multilingual:
                return multilingual_text_tokens(model, params[0], params[1])
            if turbo:
                return turbo_text_tokens(model, params[0])
            return english_text_tokens(model, params[0])

        new_indices = [
            index
            for index, request in enumerate(requests_to_batch)
            if not runner.has_row(request.request_id)
            and not runner.has_pending_vocode(request.request_id)
        ]
        # The single-request fast path stays 9-1's cleaned sequential loop.
        if len(inputs) == 1 and new_indices:
            return sequential(inputs, requests_to_batch)

        results: list = [None] * len(inputs)
        for index in new_indices:
            params = inputs[index]
            request = requests_to_batch[index]
            if _request_finished(request):
                # Aborted between scheduling and execution: never start a row
                # (9-4 already released anything that existed).
                continue
            try:
                conditionals = self._ar_conditionals(
                    params, request, turbo=turbo, multilingual=multilingual
                )
                text_tokens = _text_tokens_for(self.model, params)
                options = getattr(request.request_data, "generation_options", None)
                seed = getattr(options, "seed", None)
                if turbo:
                    _text, _speaker, temperature, repetition_penalty, top_p, top_k, _norm = params
                    row = runner.begin(
                        request.request_id,
                        conds=conditionals,
                        text_tokens=text_tokens,
                        seed=seed,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p,
                        top_k=top_k,
                    )
                elif multilingual:
                    (
                        _text,
                        _language_id,
                        _speaker,
                        _exaggeration,
                        cfg_weight,
                        temperature,
                        repetition_penalty,
                        top_p,
                        min_p,
                    ) = params
                    row = runner.begin(
                        request.request_id,
                        conds=conditionals,
                        text_tokens=text_tokens,
                        cfg_weight=cfg_weight,
                        seed=seed,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p,
                        min_p=min_p,
                    )
                else:
                    _text, _speaker, _exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p = params
                    row = runner.begin(
                        request.request_id,
                        conds=conditionals,
                        text_tokens=text_tokens,
                        cfg_weight=cfg_weight,
                        seed=seed,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p,
                        min_p=min_p,
                    )
            except Exception as error:  # noqa: BLE001 - errors this request alone
                results[index] = self._build_error_result(request, error)
                continue
            if row is None:
                # Pool refusal / disabled arena: honest sequential fallback.
                results[index] = sequential([params], [request])[0]

        # Phase 12-8: a pure continuation step decodes a burst of tokens per
        # engine step instead of one token per asyncio round-trip. An admission
        # step (a row that just ran `_prefill`) stays at exactly one token so
        # newly admitted rows return immediately, preserving the step-1
        # admission/abort/deadline/poisoned-row semantics. The burst stops as
        # soon as any row finishes or errors, so that row's KV-cache slots are
        # released (via `_take_ar_row`/`runner.pop`) without waiting through
        # extra token steps.
        burst = 1 if new_indices else max(1, int(getattr(runner, "step_tokens", 1) or 1))
        outcomes: dict = {}
        for _token_step in range(burst):
            active = [
                request.request_id
                for request in requests_to_batch
                if runner.has_row(request.request_id) and not _request_finished(request)
            ]
            if not active:
                break
            step_outcomes = runner.step(active)
            for outcome in step_outcomes:
                outcomes[outcome.request_id] = outcome
            if any(outcome.finished or outcome.error is not None for outcome in step_outcomes):
                break

        for index, request in enumerate(requests_to_batch):
            if results[index] is not None:
                continue
            if _request_finished(request):
                # Aborted/deadline-expired during this step: 9-4 already released
                # the row; make sure nothing lingers and emit no client output.
                if runner.has_row(request.request_id):
                    runner.pop(request.request_id)
                runner.pop_pending_vocode(request.request_id)
                continue
            outcome = outcomes.get(request.request_id)
            try:
                if outcome is None:
                    pending = runner.peek_pending_vocode(request.request_id)
                    if pending is not None:
                        # Already finished T3 in an earlier step; still waiting
                        # for the batch's active rows before vocoding.
                        results[index] = ExecutorResult(
                            request.request_id,
                            request.request_data,
                            UnfinishedRequestOutput(
                                request_id=request.request_id, progress=dict(pending[3])
                            ),
                        )
                        continue
                    raise RuntimeError(f"AR row for request {request.request_id} disappeared")
                if outcome.error is not None:
                    runner.pop(request.request_id)
                    results[index] = self._build_error_result(request, outcome.error)
                elif outcome.finished:
                    try:
                        tokens, conds, seed = self._take_ar_row(runner, request.request_id)
                    except Exception as error:  # noqa: BLE001 - errors this request alone
                        if runner.has_row(request.request_id):
                            runner.pop(request.request_id)
                        results[index] = self._build_error_result(request, error)
                        continue
                    # Hold the vocode; its KV slot is already released by
                    # `_take_ar_row`, so active rows are never stalled.
                    runner.put_pending_vocode(
                        request.request_id, (tokens, conds, seed, dict(outcome.progress))
                    )
                    results[index] = ExecutorResult(
                        request.request_id,
                        request.request_data,
                        UnfinishedRequestOutput(
                            request_id=request.request_id, progress=dict(outcome.progress)
                        ),
                    )
                else:
                    results[index] = ExecutorResult(
                        request.request_id,
                        request.request_data,
                        UnfinishedRequestOutput(
                            request_id=request.request_id, progress=dict(outcome.progress)
                        ),
                    )
            except Exception as error:  # noqa: BLE001 - errors this request alone
                runner.pop(request.request_id)
                results[index] = self._build_error_result(request, error)

        # Phase 12-8: defer S3Gen until the batch's active rows finish decoding,
        # then vocode every held row together (one padded `inference_batched`).
        still_decoding = any(
            runner.has_row(request.request_id) and not _request_finished(request)
            for request in requests_to_batch
        )
        if not still_decoding:
            finished: list[tuple] = []
            for index, request in enumerate(requests_to_batch):
                pending = runner.pop_pending_vocode(request.request_id)
                if pending is None:
                    continue
                tokens, conds, seed, progress = pending
                finished.append((index, request, tokens, conds, seed, progress))
            if finished:
                self._finalize_ar_rows(finished, results, turbo=turbo)
        # A request aborted between scheduling and execution produces no result.
        return [result for result in results if result is not None]

    def _take_ar_row(self, runner, request_id: str):
        """Copy a finished row's token stream/conditionals out and free its slots."""
        row = runner.row(request_id)
        if row is None:
            raise RuntimeError(f"no AR row for request {request_id}")
        tokens = runner.final_tokens(request_id)
        conds = row.conds
        seed = getattr(row, "seed", None)
        # Release the slot as soon as the tokens are copied out; S3Gen needs none.
        runner.pop(request_id)
        if tokens is None or tokens.numel() == 0:
            raise ValueError("AR decode produced no valid speech tokens")
        return tokens, conds, seed

    def _finalize_ar_rows(self, finished, results, *, turbo: bool) -> None:
        """Vocode the rows that finished this step (Phase 9-5 padded S3Gen hop).

        Non-turbo rows are grouped into length-similar sub-batches on the token
        axis and vocoded with one padded :func:`inference_batched` forward per
        group. A single-row group, an unsupported graph (turbo's meanflow hop is
        the documented skip) or any batch failure falls back to the per-row
        upstream call, so a poisoned row errors alone (8-3 deviation-6).
        """
        from harmonyspeech.modeling.models.chatterbox.s3gen_batch import (
            S3GenRow,
            inference_batched,
            supports_batched_s3gen,
        )

        s3gen = getattr(self.model, "s3gen", None)
        if turbo or not supports_batched_s3gen(s3gen):
            for row in finished:
                self._finish_ar_row(results, *row[:5], row[5], turbo=turbo)
            return

        lengths = [int(row[2].numel()) for row in finished]
        batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
        budget = getattr(self.model_config, "batch_token_budget", None) or default_token_budget(
            lengths, batch_size
        )
        groups = plan_batches(lengths, batch_size, budget)

        for group in groups:
            group_rows = [finished[position] for position in group]
            if len(group_rows) == 1:
                self._finish_ar_row(results, *group_rows[0][:5], group_rows[0][5], turbo=turbo)
                continue
            try:
                rows = [
                    S3GenRow(tokens=tokens, ref_dict=conds.gen, seed=seed)
                    for _index, _request, tokens, conds, seed, _progress in group_rows
                ]
                with torch.no_grad():
                    waveforms = inference_batched(s3gen, rows)
            except Exception as error:  # noqa: BLE001 - degrade this group to the loop
                logger.warning(
                    f"Chatterbox S3Gen padded batch failed ({error}); falling back to per-row vocode"
                )
                for row in group_rows:
                    self._finish_ar_row(results, *row[:5], row[5], turbo=turbo)
                continue
            for row, waveform in zip(group_rows, waveforms, strict=True):
                index, request, _tokens, _conds, _seed, progress = row
                try:
                    artifact = self._artifact_from_waveform(
                        waveform, watermark=not _watermark_suppressed(request)
                    )
                    result = self._build_result(request, "", TextToSpeechRequestOutput)
                    result.result_data.audio_artifact = artifact
                    result.result_data.progress = progress
                    results[index] = result
                except Exception as error:  # noqa: BLE001 - errors this request alone
                    results[index] = self._build_error_result(request, error)

    def _finish_ar_row(self, results, index, request, tokens, conds, seed, progress, *, turbo):
        """Per-row upstream vocode: the fast path and the fallback."""
        try:
            artifact = self._vocode_ar_row(
                tokens, conds, seed, turbo=turbo, watermark=not _watermark_suppressed(request)
            )
            result = self._build_result(request, "", TextToSpeechRequestOutput)
            result.result_data.audio_artifact = artifact
            result.result_data.progress = progress
            results[index] = result
        except Exception as error:  # noqa: BLE001 - errors this request alone
            results[index] = self._build_error_result(request, error)

    @contextlib.contextmanager
    def _seeded_rng(self, seed):
        """Seed the global RNG for one upstream vocode, then restore it.

        The padded hop draws each row's noise from a ``torch.Generator`` seeded
        with the row's seed; seeding the global RNG the same way makes the
        per-row upstream fallback produce the *same* audio, so a batch failure
        is a performance fallback, not an audio change.
        """
        if seed is None:
            yield
            return
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(int(seed))
            yield
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _vocode_ar_row(self, tokens, conds, seed, *, turbo: bool, watermark: bool = True):
        """Turn one row's token stream into audio through the upstream S3Gen."""
        from harmonyspeech.modeling.models.chatterbox.s3gen_precision import s3gen_autocast

        s3gen = self.model.s3gen
        precision = getattr(s3gen, "_hse_s3gen_precision", "float32")
        with self._seeded_rng(seed), s3gen_autocast(self.device, precision=precision) as autocast:
            if turbo:
                from chatterbox.models.s3gen.const import S3GEN_SIL

                silence = torch.tensor(
                    [S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=torch.long, device=self.device
                )
                speech_tokens = torch.cat([tokens.to(self.device), silence])
                wav, _ = s3gen.inference(
                    speech_tokens=speech_tokens, ref_dict=conds.gen, n_cfm_timesteps=2
                )
            else:
                wav, _ = s3gen.inference(speech_tokens=tokens.to(self.device), ref_dict=conds.gen)
            # Phase 11-2: blanket autocast around the upstream call (the loader's
            # seam already wraps `s3gen.inference`; this is the same scope, kept
            # explicit at the site the experiment names). The waveform must reach
            # the artifact/watermark path as float32.
            if autocast is not None:
                wav = wav.float()
        return self._artifact_from_waveform(wav, watermark=watermark)

    def _artifact_from_waveform(self, wav, *, watermark: bool = True):
        """Watermark a waveform and wrap it in an ``AudioArtifact``.

        ``watermark=False`` is the Phase 12-2 long-form chunk path: the serving
        layer watermarks the stitched buffer exactly once instead of every
        chunk (which would be crossfaded into a muddled mark).
        """
        wav = wav.squeeze(0).detach().cpu().numpy()
        if watermark:
            watermarker = getattr(self.model, "watermarker", None)
            if watermarker is not None:
                wav = watermarker.apply_watermark(wav, sample_rate=self.model.sr)
        return _audio_artifact_from_tensor(torch.from_numpy(wav).unsqueeze(0), self.model.sr)

    @contextlib.contextmanager
    def _without_watermark(self):
        """Temporarily replace the model's watermarker with an identity one.

        Used only for a sequential (non-AR) long-form chunk request: upstream
        ``generate()`` applies the watermark itself, so the serving layer cannot
        suppress it otherwise. The swap covers one synthesis call and is
        restored in ``finally``; the runner executes one request at a time on
        this path, so no other request observes it.
        """
        model = getattr(self, "model", None)
        original = getattr(model, "watermarker", None) if model is not None else None
        if original is None:
            yield
            return
        model.watermarker = _NullWatermarker()
        try:
            yield
        finally:
            model.watermarker = original

    @torch.inference_mode()
    def execute_model(self, requests_to_batch: list[EngineRequest]) -> list[ExecutorResult]:
        """
        Executes a group of batched requests against the model which is loaded
        :param requests_to_batch:
        :return:
        """
        if not requests_to_batch:
            return []

        # One result per request id per step (Phase 5-1). Reset before any
        # partition runs so the guard is scoped to this engine step.
        self._step_result_ids = set()

        # Split mixed-type batches before preparing inputs. Some prepare
        # functions (e.g. the Chatterbox ones) return inputs for only one of
        # the request types present, which used to drop or mispair the other
        # requests in the batch (Phase 0-3).
        partitions = self._partition_by_request_type(requests_to_batch)
        if len(partitions) > 1:
            model_type = getattr(self.model_config, "model_type", "?")
            logger.debug(
                f"Splitting mixed batch of {len(requests_to_batch)} requests for {model_type} into "
                f"{len(partitions)} type-homogeneous batches: {[t.__name__ for t in partitions]}"
            )
            outputs: list[ExecutorResult] = []
            for partition in partitions.values():
                outputs.extend(self._execute_partition(partition))
            return outputs

        return self._execute_partition(requests_to_batch)

    @staticmethod
    def _partition_by_request_type(requests_to_batch: list[EngineRequest]) -> dict:
        """Group requests by their request-input class, preserving order."""
        partitions: dict = {}
        for request in requests_to_batch:
            partitions.setdefault(type(request.request_data), []).append(request)
        return partitions

    def _execute_partition(self, requests_to_batch: list[EngineRequest]) -> list[ExecutorResult]:
        """Prepare inputs for one type-homogeneous batch and dispatch it.

        Phase 12-8: continuing AR rows skip ``prepare_inputs`` entirely (their
        prefill is already on the runner) — the handler reads only the newly
        admitted inputs. On the decode steady state this avoids re-tokenising and
        a per-step ``ThreadPoolExecutor`` for every continuing request.
        """
        runner = getattr(self, "_ar_runner", None)
        ar_active = runner is not None and getattr(runner, "enabled", False)

        def _continuing_ar(request) -> bool:
            if not ar_active:
                return False
            request_id = request.request_id
            return runner.has_row(request_id) or runner.has_pending_vocode(request_id)

        # Wrap prepare_inputs so that input-validation failures (e.g. wrong
        # request type, bad payload encoding, unsupported option) are converted to
        # per-request error results instead of raising an unhandled exception
        # that kills the async engine's background loop.
        if ar_active:
            new_requests = [req for req in requests_to_batch if not _continuing_ar(req)]
            if new_requests:
                try:
                    new_inputs = prepare_inputs(self.model_config, new_requests)
                except Exception as e:
                    logger.opt(exception=e).error(
                        f"Failed to prepare inputs for model "
                        f"{getattr(self.model_config, 'model_type', '?')}: {e}"
                    )
                    return [self._build_error_result(req, e) for req in requests_to_batch]
            else:
                # Every request in the partition is a continuing AR row: do not
                # call prepare_inputs at all (the decode steady state).
                new_inputs = []
            new_iter = iter(new_inputs)
            inputs = [
                None if _continuing_ar(request) else next(new_iter)
                for request in requests_to_batch
            ]
        else:
            try:
                inputs = prepare_inputs(self.model_config, requests_to_batch)
            except Exception as e:
                logger.opt(exception=e).error(
                    f"Failed to prepare inputs for model "
                    f"{getattr(self.model_config, 'model_type', '?')}: {e}"
                )
                return [self._build_error_result(req, e) for req in requests_to_batch]

        try:
            assert_batch_aligned(
                f"{getattr(self.model_config, 'model_type', '?')} prepare_inputs", inputs, requests_to_batch
            )
        except EngineBatchError as e:
            logger.opt(exception=e).error("Batch alignment invariant violated during input preparation")
            return [self._build_error_result(req, e) for req in requests_to_batch]

        # A handler that detects an inconsistent batch raises EngineBatchError;
        # convert it to per-request errors (Phase 5-1) rather than letting it
        # escape into the engine loop.
        try:
            return self._dispatch_handler(inputs, requests_to_batch)
        except EngineBatchError as e:
            logger.opt(exception=e).error("Batch invariant violated during model execution")
            return [self._build_error_result(req, e) for req in requests_to_batch]

    def _dispatch_handler(self, inputs, requests_to_batch: list[EngineRequest]) -> list[ExecutorResult]:
        """Run the handler matching this executor's model type."""
        model_type = getattr(self.model_config, "model_type", None)
        if model_type == "HarmonySpeechEncoder":
            return self._execute_harmonyspeech_encoder(inputs, requests_to_batch)
        elif model_type == "HarmonySpeechSynthesizer":
            return self._execute_harmonyspeech_synthesizer(inputs, requests_to_batch)
        elif model_type == "HarmonySpeechVocoder":
            return self._execute_harmonyspeech_vocoder(inputs, requests_to_batch)
        elif model_type in ["OpenVoiceV1ToneConverterEncoder", "OpenVoiceV2ToneConverterEncoder"]:
            return self._execute_openvoice_tone_converter_encoder(inputs, requests_to_batch)
        elif model_type in ["OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"]:
            return self._execute_openvoice_tone_converter(inputs, requests_to_batch)
        elif model_type == "OpenVoiceV1Synthesizer":
            return self._execute_openvoice_synthesizer(inputs, requests_to_batch)
        elif model_type == "MeloTTSSynthesizer":
            return self._execute_melotts_synthesizer(inputs, requests_to_batch)
        elif model_type == "FasterWhisper":
            return self._execute_faster_whisper(inputs, requests_to_batch)
        elif model_type == "VoiceFixerRestorer":
            return self._execute_voicefixer_restorer(inputs, requests_to_batch)
        elif model_type == "VoiceFixerVocoder":
            return self._execute_voicefixer_vocoder(inputs, requests_to_batch)
        elif model_type == "SileroVAD":
            return self._execute_silero_vad(inputs, requests_to_batch)
        elif model_type == "KittenTTSSynthesizer":
            return self._execute_kittentts_synthesizer(inputs, requests_to_batch)
        elif model_type in ["ChatterboxTTS", "ChatterboxTurboTTS", "ChatterboxMultilingualTTS"]:
            # A single executor can serve both embedding and TTS requests.
            # Batches are type-homogeneous here, so one check is sufficient.
            if requests_to_batch and isinstance(requests_to_batch[0].request_data, SpeechEmbeddingRequestInput):
                return self._execute_chatterbox_embedding(inputs, requests_to_batch)
            elif model_type == "ChatterboxTTS":
                return self._execute_chatterbox_tts(inputs, requests_to_batch)
            elif model_type == "ChatterboxTurboTTS":
                return self._execute_chatterbox_turbo_tts(inputs, requests_to_batch)
            else:
                return self._execute_chatterbox_multilingual_tts(inputs, requests_to_batch)
        elif model_type == "ChatterboxEmbedding":
            return self._execute_chatterbox_embedding(inputs, requests_to_batch)
        elif model_type == "ChatterboxVC":
            return self._execute_chatterbox_vc(inputs, requests_to_batch)
        else:
            raise NotImplementedError(f"Model {model_type} is not supported")

    def _register_result_id(self, request_id: str, *, raise_on_duplicate: bool = True) -> bool:
        """Register a result being built for ``request_id`` in the current step.

        Returns ``False`` when the id was already registered. With
        ``raise_on_duplicate`` (the success path) a duplicate raises
        :class:`EngineBatchError`: building two results for one request inside a
        single step means the batch pairing is broken, and silently returning
        one of them would hide it (Phase 5-1).

        The guard is scoped to an engine step: :meth:`execute_model` opens the
        scope, and a handler invoked directly (unit tests, tooling) has no scope
        so nothing is tracked.
        """
        ids = getattr(self, "_step_result_ids", None)
        if ids is None:
            return True
        if request_id in ids:
            message = f"duplicate result for request {request_id} within one engine step"
            if raise_on_duplicate:
                raise EngineBatchError(message)
            logger.warning(message)
            return False
        ids.add(request_id)
        return True

    def _build_error_result(self, initial_request, exception: Exception, *, log_exception: bool = True):
        """Build an error ExecutorResult for a single request that failed during inference.
        Wraps in ExecutorResult so _process_model_outputs / check_forward_processing can handle it
        without crashing. The serving layer then checks finish_reason == 'error' on result_data
        and returns an HTTP error to the caller.

        ``log_exception=False`` is used for expected cooperative cancellations
        (an aborted Whisper request): the request errors, but the server log
        should not carry a traceback for a client-initiated abort.
        """
        request_id = initial_request.request_id
        self._register_result_id(request_id, raise_on_duplicate=False)
        metrics = initial_request.metrics
        metrics.finished_time = time.time()
        # Log full stack trace so the server-side cause is always visible in logs.
        if log_exception:
            logger.opt(exception=exception).error(
                f"Inference error for request {request_id}: {exception}"
            )
        else:
            logger.info(f"Request {request_id} did not complete: {exception}")
        error_output = RequestOutput(
            request_id=request_id, finish_reason="error", error=str(exception), metrics=metrics
        )
        return ExecutorResult(request_id=request_id, input_data=initial_request.request_data, result_data=error_output)

    def _build_result(self, initial_request, inference_result, result_cls):
        request_id = initial_request.request_id
        # Request-id uniqueness within a step (Phase 5-1): raises EngineBatchError
        # if this request already produced a result, which the partition wrapper
        # converts into per-request errors.
        self._register_result_id(request_id)
        metrics = initial_request.metrics
        metrics.finished_time = time.time()

        # Build kwargs based on result_cls type
        kwargs = {"request_id": request_id, "finish_reason": "stop", "metrics": metrics}

        # TextToSpeechRequestOutput requires text and output parameters
        if result_cls == TextToSpeechRequestOutput:
            input_text = ""
            if hasattr(initial_request.request_data, "input_text"):
                input_text = initial_request.request_data.input_text
            kwargs["text"] = input_text
            kwargs["output"] = inference_result
        else:
            kwargs["output"] = inference_result

        result = ExecutorResult(
            request_id=request_id, input_data=initial_request.request_data, result_data=result_cls(**kwargs)
        )
        return result

    def _execute_harmonyspeech_encoder(self, inputs, requests_to_batch):
        assert_batch_aligned("_execute_harmonyspeech_encoder", inputs, requests_to_batch)

        def partial_frames(audio_artifact):
            """Numpy mel frontend per clip (the model consumes its fixed-length partials)."""
            # Numpy preprocessing runs on the artifact's waveform (CPU-side); the
            # move is explicit, never implicit.
            source = _bind_to_device(audio_artifact, "cpu")
            preprocessed = preprocess_wav(source.numpy, source_sr=source.sample_rate)
            return get_input_frames(preprocessed)

        def embed_utterance(audio_artifact, initial_request):
            """Per-item loop: the ``len == 1`` fast path and the fallback."""
            input_frames = partial_frames(audio_artifact)
            utterances_tensor = torch.from_numpy(input_frames).to(self.device)
            partial_embeds = self.model(utterances=utterances_tensor).detach().cpu().numpy()
            return _finalize_encoder_embedding(partial_embeds)

        # Cross-request cache (Phase 7-2): resolve hits for the whole batch
        # first (guard once at entry — zero cost when no cache is attached),
        # compute only the misses below; outputs stay in request order.
        cache = getattr(self, "embedding_cache", None)
        slots = self._resolve_cache_slots(
            cache, requests_to_batch, lambda i: self._embedding_cache_lookup(inputs[i], {})
        )

        # ------------------------------------------------------------------
        # Padded-batch path (Phase 8-2). Rows are (request, partial-utterance)
        # pairs: the vendored ``SpeakerEncoder`` already takes ``[B, frames,
        # mels]`` and partials are fixed-length by construction, so the frame
        # axis is the row axis — no change to the numpy mel frontend and no
        # waveform padding. Only cache misses enter the batch; hits never reach
        # the model.
        # ------------------------------------------------------------------
        batched: dict[int, SpeakerArtifact] = {}
        if batched_inference_enabled(self.model_config) and len(inputs) > 1:
            misses = [index for index in range(len(inputs)) if slots[index][1] is None]
            try:
                rows: list[tuple[int, np.ndarray]] = [
                    (index, frames) for index in misses for frames in partial_frames(inputs[index])
                ]
                if rows:
                    partials: dict[int, list[np.ndarray]] = {index: [] for index in misses}
                    plan_lengths = [int(row[1].shape[0]) for row in rows]
                    batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
                    budget = self.model_config.batch_token_budget or default_token_budget(
                        plan_lengths, batch_size
                    )
                    groups = plan_batches(plan_lengths, batch_size, budget)
                    for group in groups:
                        group_rows = [rows[index] for index in group]
                        with torch.no_grad():
                            padded, padded_lengths = pad_feature_batch(
                                [
                                    torch.from_numpy(np.ascontiguousarray(row[1].T, dtype=np.float32))
                                    for row in group_rows
                                ],
                                device=self.device,
                                return_lengths=True,
                            )
                            embeds = (
                                self.model.forward_batched(padded.transpose(1, 2).contiguous(), padded_lengths)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        for position, absolute in enumerate(group):
                            partials[rows[absolute][0]].append(embeds[position])
                    for index in misses:
                        artifact = _finalize_encoder_embedding(np.stack(partials[index]))
                        batched[index] = artifact
                        self._store_embedding_cache_entry(slots[index][0], artifact)
                    logger.debug(
                        f"HarmonySpeech encoder: {len(rows)} partial(s) from {len(misses)} request(s) "
                        f"forwarded in {len(groups)} padded sub-batch(es)"
                    )
            except Exception as e:  # noqa: BLE001 - degrade to the per-item loop
                logger.warning(
                    f"HarmonySpeech encoder batched embedding failed ({e}); falling back to the per-request loop"
                )
                batched.clear()

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                key, cached = slots[i]
                if cached is not None:
                    embedding = cached
                elif i in batched:
                    embedding = batched[i]
                else:
                    embedding = embed_utterance(x, initial_request)
                    self._store_embedding_cache_entry(key, embedding)
                result = self._build_result(initial_request, "", SpeechEmbeddingRequestOutput)
                result.result_data.embedding_artifact = embedding
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_harmonyspeech_synthesizer(self, inputs, requests_to_batch):
        # Phase 8-4 audited the vendored forward and kept this a loop: the
        # HarmonySpeech synthesizer is ForwardTacotron (three SeriesPredictors, a
        # CBHG prenet, a duration-driven LengthRegulator and a CBHG postnet, each
        # with a bidirectional GRU/LSTM over the padded axis), not the
        # SynthesizerTrn the plan expected. See the "Vendored forward audit" in
        # `.current_work/engine-modernization/8-4-HarmonySpeechVoiceFixerBatching.md`
        # for the measurements and the verdict.
        assert_batch_aligned("_execute_harmonyspeech_synthesizer", inputs, requests_to_batch)

        def synthesize_text(input_params, initial_request):
            # Get inputs
            chars, speaker_embeddings, speed_modifier, pitch_function, energy_function = input_params

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeddings).float().to(self.device)

            kwargs = {
                "x": chars,
                "spk_emb": speaker_embeddings,
                "alpha": speed_modifier,
                "pitch_function": pitch_function,
                "energy_function": energy_function,
            }
            _, mels, _, _, _ = self.model.generate(**kwargs)
            mels = mels.detach().cpu().numpy()
            # Combine Spectograms from generation
            specs = []
            for m in mels:
                specs.append(m)
            breaks = [subspec.shape[1] for subspec in specs]
            full_spectogram = np.concatenate(specs, axis=1)

            # Typed mel artifact (with the vocoder's breaks) replaces the
            # wire JSON envelope; nothing is serialized on this hop.
            return MelArtifact.from_numpy(
                full_spectogram,
                sample_rate=16000,
                hop_length=200,
                breaks=breaks,
            )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                mel_artifact = synthesize_text(x, initial_request)
                result = self._build_result(initial_request, "", SpeechSynthesisRequestOutput)
                result.result_data.mel_artifact = mel_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_harmonyspeech_vocoder(self, inputs, requests_to_batch):
        assert_batch_aligned("_execute_harmonyspeech_vocoder", inputs, requests_to_batch)

        def finish_waveform(wav, breaks):
            """Post-decode steps, identical on both paths.

            Breaks (inter-chunk silence) apply **after** decode, so batching only
            has to hand this function the request's own decoded samples and its
            own ``breaks`` list — the pairing never leaves the request.
            """
            # Add breaks if defined
            if breaks is not None:
                # Add breaks - FIXME: Make this parameterized
                b_ends = np.cumsum(np.array(breaks) * 200)  # hop_size 200
                b_starts = np.concatenate(([0], b_ends[:-1]))
                wavs = [wav[start:end] for start, end in zip(b_starts, b_ends, strict=False)]
                syn_breaks = [np.zeros(int(0.15 * 16000))] * len(breaks)  # Sample Rate 16000
                wav = np.concatenate(
                    [i for w, b in zip(wavs, syn_breaks, strict=False) for i in (w, b)]
                )

            # Normalize
            wav = wav / np.abs(wav).max() * 0.97

            return AudioArtifact.from_numpy(wav, 16000)

        def mel_frames(mel_artifact):
            """Per-request device mel on the frame axis (the batch's row unit)."""
            mel_tensor = _bind_to_device(mel_artifact, self.device).mel.squeeze(0)
            return mel_tensor / 4.0  # Fixme: make this configurable

        def vocode_mel(input_params, initial_request):
            """Per-item loop: the ``len == 1`` fast path and the fallback."""
            mel_artifact, breaks = input_params
            kwargs = {"c": mel_frames(mel_artifact).T}
            wav = self.model.inference(**kwargs).detach().cpu().numpy()
            wav = wav.squeeze(1)
            return finish_waveform(wav, breaks)

        # ------------------------------------------------------------------
        # Padded-batch path (Phase 8-4). Rows are requests on the mel-frame axis:
        # the vocoder is a MelGAN decoder, so a padded row's frames must not reach
        # the valid tail (`inference_batched` masks every stage, 8-3 lesson).
        # Per-request ``breaks`` and normalization stay in ``finish_waveform``.
        # ------------------------------------------------------------------
        batched: dict[int, AudioArtifact] = {}
        if batched_inference_enabled(self.model_config) and len(inputs) > 1:
            rows: list[tuple[int, torch.Tensor]] = []
            for index, (mel_artifact, _breaks) in enumerate(inputs):
                try:
                    rows.append((index, mel_frames(mel_artifact)))
                except Exception as e:  # noqa: BLE001 - the loop reports this request alone
                    logger.warning(
                        f"HarmonySpeech vocoder: mel frontend failed for request "
                        f"{requests_to_batch[index].request_id} ({e}); per-item loop will report it"
                    )
            if rows:
                frame_lengths = [int(mel.shape[-1]) for _index, mel in rows]
                batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
                budget = self.model_config.batch_token_budget or default_token_budget(
                    frame_lengths, batch_size
                )
                groups = plan_batches(frame_lengths, batch_size, budget)
                forwarded = 0
                for group in groups:
                    group_rows = [rows[position] for position in group]
                    try:
                        with torch.no_grad():
                            padded, padded_lengths = pad_feature_batch(
                                [mel for _index, mel in group_rows],
                                device=self.device,
                                return_lengths=True,
                            )
                            waveform, sample_lengths = self.model.inference_batched(
                                padded, padded_lengths
                            )
                            trimmed = trim_decoder_output(waveform, sample_lengths, 1)
                    except Exception as e:  # noqa: BLE001 - degrade this sub-batch to the loop
                        logger.warning(
                            f"HarmonySpeech vocoder batched forward failed ({e}); "
                            f"falling back to the per-request loop for {len(group_rows)} request(s)"
                        )
                        continue
                    forwarded += len(group_rows)
                    for position, _absolute in enumerate(group):
                        index = group_rows[position][0]
                        try:
                            wav = trimmed[position][0].detach().cpu().float().numpy()
                            batched[index] = finish_waveform(wav, inputs[index][1])
                        except Exception as e:  # noqa: BLE001 - the loop reports this request alone
                            logger.warning(
                                f"HarmonySpeech vocoder: could not finish request "
                                f"{requests_to_batch[index].request_id} ({e}); "
                                f"per-item loop will report it"
                            )
                if batched:
                    logger.debug(
                        f"HarmonySpeech vocoder: {forwarded} request(s) forwarded in "
                        f"{len(groups)} padded sub-batch(es)"
                    )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = batched.get(i)
                if audio_artifact is None:
                    audio_artifact = vocode_mel(x, initial_request)
                result = self._build_result(initial_request, "", VocodeRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_openvoice_tone_converter_encoder(self, inputs, requests_to_batch):
        # Get model flavour if applicable
        flavour = get_model_flavour(self.model_config)
        # Load config
        hf_config = get_model_config(
            self.model_config.model, self.model_config.model_type, self.model_config.revision, flavour
        )

        # The encoder batch below covers the reference encoder; the per-segment
        # STFT frontend stays per row (it reflect-pads every clip on its own).
        assert_batch_aligned("_execute_openvoice_tone_converter_encoder", inputs, requests_to_batch)

        def segment_spectrogram(segment):
            """Per-segment STFT: the historical frontend, unchanged."""
            y = _bind_to_device(segment, self.device).waveform
            return spectrogram_torch(
                y,
                hf_config.data.filter_length,
                hf_config.data.sampling_rate,
                hf_config.data.hop_length,
                hf_config.data.win_length,
                center=False,
            ).to(self.device)

        def run_converter_encoder(vad_audio_segments, initial_request):
            """Per-item loop: the ``len == 1`` fast path and the fallback.

            A reference clip the VAD could not detect as speech yields **zero**
            segments (a synthetic tone, a silent clip, a wrong sample rate). The
            encoder cannot condition on nothing: without this guard
            ``torch.stack([])`` raised ``RuntimeError: stack expects a
            non-empty TensorList``, an opaque 500 that told the client nothing.
            Raise a diagnosable message instead, which the serving layer turns
            into a 400 (Phase 12-4 finding, 2026-11-09).
            """
            if not vad_audio_segments:
                raise ValueError(
                    "reference clip contains no detectable speech (VAD segments empty)"
                )
            gs = []
            for segment in vad_audio_segments:
                y = segment_spectrogram(segment)
                with torch.no_grad():
                    g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                    gs.append(g.detach())
            gs = torch.stack(gs).mean(0)

            # Typed embedding artifact (no torch.save/text-encoding on the hop).
            return SpeakerArtifact(embedding=gs.cpu(), backend="openvoice")

        # Cross-request cache (Phase 7-2). The key hashes the request's decoded
        # input audio (not the VAD-sliced segments) plus the VAD source, which
        # together determine the segment set the encoder embeds.
        cache = getattr(self, "embedding_cache", None)

        def _lookup(index):
            entry = requests_to_batch[index]
            audio = get_audio_artifact(as_request_input(entry), entry, "input_audio")
            return self._embedding_cache_lookup(audio, self._openvoice_encoder_params(entry))

        slots = self._resolve_cache_slots(cache, requests_to_batch, _lookup)

        # ------------------------------------------------------------------
        # Padded-batch path (Phase 8-2). Rows are (request, VAD segment) pairs —
        # a request may own several segments, like 5-2's (request, sentence)
        # rows. The reference encoder takes mel frames, so rows are padded on the
        # frame axis and forwarded for their real frame counts; each request's
        # segment embeddings are then stacked and averaged in segment order, the
        # same arithmetic the loop performs. Only cache misses enter the batch.
        # ------------------------------------------------------------------
        batched: dict[int, SpeakerArtifact] = {}
        if batched_inference_enabled(self.model_config) and len(inputs) > 1:
            misses = [index for index in range(len(inputs)) if slots[index][1] is None]
            try:
                rows: list[tuple[int, AudioArtifact]] = [
                    (index, segment) for index in misses for segment in inputs[index]
                ]
                if rows:
                    segment_gs: dict[int, list[torch.Tensor]] = {index: [] for index in misses}
                    # The sample count is the segment length on a per-row basis
                    # (frames = f(samples) monotonic), so planning on it keeps the
                    # same length-similar grouping without a second STFT.
                    plan_lengths = [int(segment.waveform.numel()) for _, segment in rows]
                    batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
                    budget = self.model_config.batch_token_budget or default_token_budget(
                        plan_lengths, batch_size
                    )
                    groups = plan_batches(plan_lengths, batch_size, budget)
                    for group in groups:
                        group_rows = [rows[index] for index in group]
                        specs = [segment_spectrogram(segment) for _row, segment in group_rows]
                        with torch.no_grad():
                            padded, padded_lengths = pad_feature_batch(
                                [spec[0] for spec in specs],
                                device=self.device,
                                return_lengths=True,
                            )
                            g = self.model.ref_enc.forward_batched(
                                padded.transpose(1, 2).contiguous(), padded_lengths
                            )
                        for position, absolute in enumerate(group):
                            segment_gs[rows[absolute][0]].append(
                                g[position].detach().unsqueeze(-1).unsqueeze(0)
                            )
                    for index in misses:
                        if not segment_gs[index]:
                            # No segment survived the VAD filters: the loop below
                            # raises the honest "no detectable speech" ValueError
                            # (Phase 12-4) exactly as it did before batching existed.
                            continue
                        gs = torch.stack(segment_gs[index]).mean(0)
                        artifact = SpeakerArtifact(embedding=gs.cpu(), backend="openvoice")
                        batched[index] = artifact
                        self._store_embedding_cache_entry(slots[index][0], artifact)
                    logger.debug(
                        f"OpenVoice tone-converter encoder: {len(rows)} segment(s) from {len(misses)} "
                        f"request(s) forwarded in {len(groups)} padded sub-batch(es)"
                    )
            except Exception as e:  # noqa: BLE001 - degrade to the per-item loop
                logger.warning(
                    f"OpenVoice tone-converter encoder batched embedding failed ({e}); "
                    f"falling back to the per-request loop"
                )
                batched.clear()

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                key, cached = slots[i]
                if cached is not None:
                    embedding = cached
                elif i in batched:
                    embedding = batched[i]
                else:
                    embedding = run_converter_encoder(x, initial_request)
                    self._store_embedding_cache_entry(key, embedding)
                result = self._build_result(initial_request, "", SpeechEmbeddingRequestOutput)
                result.result_data.embedding_artifact = embedding
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_openvoice_tone_converter(self, inputs, requests_to_batch):
        # Get model flavour if applicable
        flavour = get_model_flavour(self.model_config)
        # Load config
        hf_config = get_model_config(
            self.model_config.model, self.model_config.model_type, self.model_config.revision, flavour
        )
        sample_rate = int(hf_config.data.sampling_rate)
        # A VITS decoder emits ``prod(upsample_rates)`` samples per mel frame;
        # for both tone converters that equals the configured hop length. Reading
        # it off the model keeps the unpad step tied to the model's own output
        # semantics (the solo path returns the whole decoder output) rather than
        # to a coincidental config value.
        samples_per_frame = int(hf_config.data.hop_length)
        decoder_rates = getattr(getattr(self.model, "dec", None), "upsample_rates", None)
        if decoder_rates:
            samples_per_frame = 1
            for rate in decoder_rates:
                samples_per_frame *= int(rate)

        assert_batch_aligned("_execute_openvoice_tone_converter", inputs, requests_to_batch)

        def speaker_embeddings(input_params):
            """Device-local source/target embeddings, validated per request."""
            _audio_artifact, target_speaker, source_speaker = input_params

            # Both embedding blobs decoded through the safe loaders at admission
            # (``weights_only=True``), so a hostile pickle fails closed before
            # reaching this handler. Device moves are explicit here.
            if target_speaker is None or target_speaker.embedding is None:
                raise ValueError("input_embedding/target_embedding: missing embedding tensor")
            if source_speaker is None or source_speaker.embedding is None:
                raise ValueError("source_embedding: missing embedding tensor")
            target_embedding = _bind_to_device(target_speaker, self.device).embedding
            src_embedding = _bind_to_device(source_speaker, self.device).embedding
            return src_embedding, target_embedding

        def mel_frontend(audio_artifact):
            """Per-clip mel frontend: the historical frontend, unchanged.

            ``spectrogram_torch`` reflect-pads every clip before the STFT, so a
            padded waveform batch would change the shorter rows' edge frames
            (8-2 lesson). The batch is built on the mel frames instead.
            """
            y = _bind_to_device(audio_artifact, self.device).waveform.float()
            return spectrogram_torch(
                y,
                hf_config.data.filter_length,
                hf_config.data.sampling_rate,
                hf_config.data.hop_length,
                hf_config.data.win_length,
                center=False,
            ).to(self.device)

        def run_converter_encoder(input_params, initial_request):
            """Per-item loop: the ``len == 1`` fast path and the fallback."""
            audio_artifact = input_params[0]
            src_embedding, target_embedding = speaker_embeddings(input_params)
            spec = mel_frontend(audio_artifact)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)

            with torch.no_grad():
                output_audio = self.model.voice_conversion(
                    spec, spec_lengths, sid_src=src_embedding, sid_tgt=target_embedding, tau=0.3
                )
                output_audio = output_audio[0][0, 0].data.cpu().float().numpy()

            return AudioArtifact.from_numpy(output_audio, sample_rate)

        # ------------------------------------------------------------------
        # Padded-batch path (Phase 8-3). One row per request: the mel frontend
        # stays per clip (reflect padding, see above), the rows are padded on
        # the mel-frame axis, and every length-similar sub-batch is one
        # ``voice_conversion_batched`` forward. Outputs are unpadded in the
        # sample domain with the model's own frame counts (``y_lengths *
        # prod(upsample_rates)``), which is exactly the solo path's full
        # decoder output. A failing sub-batch falls back to the loop for its
        # requests only.
        # ------------------------------------------------------------------
        batched: dict[int, AudioArtifact] = {}
        if batched_inference_enabled(self.model_config) and len(inputs) > 1:
            rows: list[tuple[int, torch.Tensor]] = []
            for index, input_params in enumerate(inputs):
                try:
                    rows.append((index, mel_frontend(input_params[0])))
                except Exception as e:  # noqa: BLE001 - the loop reports this request alone
                    logger.warning(
                        f"OpenVoice tone converter: mel frontend failed for request "
                        f"{requests_to_batch[index].request_id} ({e}); per-item loop will report it"
                    )
            if rows:
                frame_lengths = [int(spec.size(-1)) for _index, spec in rows]
                batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
                budget = self.model_config.batch_token_budget or default_token_budget(
                    frame_lengths, batch_size, max_budget=DEFAULT_MEL_FRAME_BATCH_BUDGET
                )
                groups = plan_batches(frame_lengths, batch_size, budget)
                forwarded = 0
                for group in groups:
                    group_rows = [rows[position] for position in group]
                    try:
                        with torch.no_grad():
                            padded, padded_lengths = pad_feature_batch(
                                [spec[0] for _index, spec in group_rows],
                                device=self.device,
                                return_lengths=True,
                            )
                            embeddings = [
                                speaker_embeddings(inputs[index]) for index, _spec in group_rows
                            ]
                            sid_src = torch.cat([source for source, _target in embeddings], dim=0)
                            sid_tgt = torch.cat([target for _source, target in embeddings], dim=0)
                            output_audio, _y_mask, _ = self.model.voice_conversion_batched(
                                padded,
                                padded_lengths,
                                sid_src=sid_src,
                                sid_tgt=sid_tgt,
                                tau=0.3,
                            )
                            trimmed = trim_decoder_output(
                                output_audio, padded_lengths, samples_per_frame
                            )
                    except Exception as e:  # noqa: BLE001 - degrade this sub-batch to the loop
                        logger.warning(
                            f"OpenVoice tone converter batched forward failed ({e}); "
                            f"falling back to the per-request loop for {len(group_rows)} request(s)"
                        )
                        continue
                    forwarded += len(group_rows)
                    for position, _absolute in enumerate(group):
                        index = group_rows[position][0]
                        batched[index] = AudioArtifact.from_numpy(
                            trimmed[position][0].detach().cpu().float().numpy(), sample_rate
                        )
                if batched:
                    logger.debug(
                        f"OpenVoice tone converter: {forwarded} request(s) forwarded in "
                        f"{len(groups)} padded sub-batch(es)"
                    )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = batched.get(i)
                if audio_artifact is None:
                    audio_artifact = run_converter_encoder(x, initial_request)
                result = self._build_result(initial_request, "", VoiceConversionRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _whisper_lane_pool(self) -> WhisperLanePool:
        """Return this runner's Whisper lane pool, rebuilding it if M changed.

        The pool is owned by the runner (not created per step), so the worker
        threads — and their per-thread pipelines — survive across engine steps
        instead of being torn down and re-created. A model reload or a config
        change swaps it.
        """
        workers = max(1, int(getattr(self.model_config, "num_workers", 1) or 1))
        model = getattr(self, "model", None)
        pool = getattr(self, "_whisper_pool", None)
        if pool is None or pool.num_workers != workers or pool.model is not model:
            if pool is not None:
                pool.close()
            pool = WhisperLanePool(model, num_workers=workers)
            self._whisper_pool = pool
        return pool

    def _whisper_window_batcher(self) -> WhisperWindowBatcher:
        """Return this runner's cross-request window batcher (Phase 10-2).

        Stateless across steps, so it is simply rebuilt when the loaded model
        changes (model reload / config swap). Kept on the runner so tests can
        patch ``model_runner_base.WhisperWindowBatcher``. The cap comes from the
        per-model ``whisper_window_batch_size`` key (Phase 11-0, default 8),
        falling back to the module constant for configs built before it existed.
        """
        model = getattr(self, "model", None)
        batcher = getattr(self, "_whisper_batcher_obj", None)
        window_batch_size = getattr(
            self.model_config, "whisper_window_batch_size", WHISPER_WINDOW_BATCH_SIZE
        )
        if (
            batcher is None
            or batcher.model is not model
            or batcher.window_batch_size != max(1, int(window_batch_size or WHISPER_WINDOW_BATCH_SIZE))
        ):
            batcher = WhisperWindowBatcher(model, window_batch_size=window_batch_size)
            self._whisper_batcher_obj = batcher
        return batcher

    def whisper_lane_stats(self) -> dict:
        """Phase 10-1 lane accounting for ``/metrics`` (see ``get_whisper_lane_stats``)."""
        pool = getattr(self, "_whisper_pool", None)
        batch_fallbacks = int(getattr(self, "_whisper_batch_fallbacks", 0) or 0)
        if pool is None:
            return {
                "workers": max(1, int(getattr(self.model_config, "num_workers", 1) or 1)),
                "in_flight": 0,
                "queue_wait": {"sum": 0.0, "count": 0, "max": 0.0, "buckets": []},
                "ct2": _ct2_queue_state(getattr(self, "model", None)),
                "batch_fallbacks": batch_fallbacks,
            }
        stats = dict(pool.stats())
        stats["batch_fallbacks"] = batch_fallbacks
        return stats

    def _record_whisper_batch_fallback(self, request_count: int, error: Exception) -> None:
        """Count + log a window-batcher failure that triggered the sequential fallback."""
        self._whisper_batch_fallbacks = int(getattr(self, "_whisper_batch_fallbacks", 0) or 0) + 1
        logger.opt(exception=error).warning(
            f"Whisper window batcher failed for {request_count} request(s) ({error}); "
            "falling back to per-request pipeline.transcribe for this step"
        )

    def _whisper_batched_pairs(self, inputs, requests_to_batch):
        """Run the cross-request window batcher, normalized to the lane-pool contract."""
        batcher = self._whisper_window_batcher()
        pairs = []
        for payload, error in batcher.transcribe_batch(inputs, requests_to_batch):
            if error is not None:
                pairs.append((None, error))
                continue
            pairs.append(((json.dumps(payload), VadArtifact.from_payload(payload)), None))
        return pairs

    def _build_whisper_outputs(self, requests_to_batch, pairs):
        """Turn ``(payload, error)`` pairs into one result per request, in input order."""
        outputs = []
        for initial_request, (payload, error) in zip(requests_to_batch, pairs, strict=True):
            if error is not None:
                if isinstance(error, WhisperRequestAbortedError):
                    result = self._build_error_result(initial_request, error, log_exception=False)
                else:
                    result = self._build_error_result(initial_request, error)
                outputs.append(result)
                continue
            try:
                inference_result, vad_artifact = payload
                result = self._build_result(initial_request, inference_result, SpeechTranscriptionRequestOutput)
                result.result_data.vad_artifact = vad_artifact
            except Exception as e:  # noqa: BLE001 - pairing/result-building failure for this request only
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_faster_whisper(self, inputs, requests_to_batch):
        """Transcribe (and segment, for the pipeline's VAD step) per request.

        Two execution paths share this handler (Phase 10-2):

        * **Cross-request window batch** — when a step holds more than one
          request and ``batched_inference`` is enabled. ``WhisperWindowBatcher``
          collects every admitted request's ≤30 s windows, tags each row with
          ``(request, window_index, offset)`` and issues one CT2
          ``encode(batch)`` + one ``generate(encoder_output, prompts, …)`` per
          formed batch (≤ ``whisper_window_batch_size`` windows, default 8,
          owner-approved initial). CT2 owns the KV cache and
          beam search; this handler only forms batches and stitches segments
          back per request. Language detection reuses the batch's encoder
          output (no second encoder pass).
        * **Per-request pipeline** — the ``len == 1`` fast path, an explicit
          ``batched_inference: false``, and the fallback when the batcher raises
          (batch-level CT2 failure). A step of N requests is dispatched to the
          runner's M-worker pool (M = ``num_workers`` = CT2 ``inter_threads``;
          same-device workers share weights), each worker consuming its own
          segment generator to the end (Phase 10-1).

        Failure isolation: a request whose window collection/decoding fails is
        returned as that request's error and never de-batches its neighbours.
        A batch-level failure falls back to the per-request path for the whole
        step, where the poisoned request errors alone — never a user-visible
        error for the neighbours (Phase 8-3 deviation-6 discipline).

        Cancellation is cooperative: the lane pool checks between segments, the
        window batcher between formed batches (faster-whisper has no
        mid-``generate`` abort), so an aborted request errors on its own while
        the rest of the step completes.
        """
        assert_batch_aligned("_execute_faster_whisper", inputs, requests_to_batch)

        if len(inputs) > 1 and whisper_batched_inference_enabled(self.model_config):
            try:
                return self._build_whisper_outputs(
                    requests_to_batch, self._whisper_batched_pairs(inputs, requests_to_batch)
                )
            except Exception as error:  # noqa: BLE001 - never user-visible; sequential fallback
                self._record_whisper_batch_fallback(len(requests_to_batch), error)

        pool = self._whisper_lane_pool()
        pairs = pool.run(inputs, requests_to_batch, pool.transcribe)
        return self._build_whisper_outputs(requests_to_batch, pairs)

    def _execute_openvoice_synthesizer(self, inputs, requests_to_batch):
        assert_batch_aligned("_execute_openvoice_synthesizer", inputs, requests_to_batch)
        # Get model flavour if applicable
        flavour = get_model_flavour(self.model_config)
        # Load config
        hf_config = get_model_config(
            self.model_config.model, self.model_config.model_type, self.model_config.revision, flavour
        )
        sample_rate = int(hf_config.data.sampling_rate)
        # VITS hop: one decoder frame == hop_length samples. Used to unpad the
        # batched decoder output using the model's own per-sample y_lengths.
        hop_length = int(hf_config.data.hop_length)

        def _assemble(audio_segments, speed_modifier):
            """Concatenate per-sentence waveforms with the historical 50 ms gap."""
            audio_segments_out = []
            gap = int((sample_rate * 0.05) / speed_modifier)
            for segment_data in audio_segments:
                audio_segments_out += np.asarray(segment_data).reshape(-1).tolist()
                audio_segments_out += [0] * gap
            return np.array(audio_segments_out).astype(np.float32)

        def synthesize_text(input_params, initial_request):
            """Per-sentence loop: the ``len(inputs) == 1`` fast path / fallback."""
            audio_segment_list = []
            text_normalized, speaker_id, speed_modifier, sampling = input_params
            for text_element in text_normalized:
                # Inference
                with torch.no_grad():
                    x_tst = text_element.unsqueeze(0).to(self.device)
                    x_tst_lengths = torch.LongTensor([text_element.size(0)]).to(self.device)
                    sid = torch.LongTensor([speaker_id]).to(self.device)
                    audio = (
                        self.model.infer(
                            x_tst,
                            x_tst_lengths,
                            sid=sid,
                            noise_scale=sampling["noise_scale"],
                            noise_scale_w=sampling["noise_scale_w"],
                            sdp_ratio=sampling["sdp_ratio"],
                            length_scale=1.0 / speed_modifier,
                        )[0][0, 0]
                        .data.cpu()
                        .float()
                        .numpy()
                    )
                audio_segment_list.append(audio)

            return AudioArtifact.from_numpy(_assemble(audio_segment_list, speed_modifier), sample_rate)

        # ------------------------------------------------------------------
        # Padded-batch path (Phase 5-2). A step is flattened into (request,
        # sentence) rows and forwarded as one padded [N, T_max] batch per
        # length-similar sub-batch; outputs are unpadded with the model's own
        # per-sample frame counts, then reassembled per request in input order.
        # ------------------------------------------------------------------
        if batched_inference_enabled(self.model_config) and len(inputs) > 1:
            rows = [
                (index, element, speaker_id, speed_modifier, sampling)
                for index, (text_normalized, speaker_id, speed_modifier, sampling) in enumerate(inputs)
                for element in text_normalized
            ]
            if rows:
                try:
                    segment_audio: list[list[np.ndarray]] = [[] for _ in inputs]
                    lengths = [int(row[1].numel()) for row in rows]
                    batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
                    budget = self.model_config.batch_token_budget or default_token_budget(
                        lengths, batch_size, max_budget=DEFAULT_PHONE_BATCH_BUDGET
                    )
                    groups = plan_batches(lengths, batch_size, budget)
                    for group in groups:
                        group_rows = [rows[index] for index in group]
                        with torch.no_grad():
                            padded, padded_lengths = pad_phone_batch(
                                [row[1] for row in group_rows], pad_id=0, device=self.device
                            )
                            sid = torch.tensor(
                                [row[2] for row in group_rows], dtype=torch.long, device=self.device
                            )
                            waveforms, y_lengths, _, _, _ = self.model.infer_batched(
                                padded,
                                padded_lengths,
                                sid=sid,
                                noise_scale=collapse_per_sample(row[4]["noise_scale"] for row in group_rows),
                                noise_scale_w=collapse_per_sample(row[4]["noise_scale_w"] for row in group_rows),
                                sdp_ratio=collapse_per_sample(row[4]["sdp_ratio"] for row in group_rows),
                                length_scale=collapse_per_sample(
                                    1.0 / row[3] for row in group_rows
                                ),
                            )
                            trimmed = trim_decoder_output(waveforms, y_lengths, hop_length)
                        for position, absolute in enumerate(group):
                            segment_audio[rows[absolute][0]].append(
                                trimmed[position][0].detach().cpu().float().numpy()
                            )
                    logger.debug(
                        f"OpenVoice V1: {len(rows)} sentence(s) from {len(inputs)} request(s) "
                        f"forwarded in {len(groups)} padded sub-batch(es)"
                    )
                    return [
                        self._build_audio_result(
                            requests_to_batch[index],
                            AudioArtifact.from_numpy(_assemble(segments, inputs[index][2]), sample_rate),
                        )
                        for index, segments in enumerate(segment_audio)
                    ]
                except Exception as e:  # noqa: BLE001 - degrade to the per-item loop
                    logger.warning(
                        f"OpenVoice V1 batched synthesis failed ({e}); falling back to the per-request loop"
                    )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = synthesize_text(x, initial_request)
                result = self._build_result(initial_request, "", SpeechSynthesisRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _build_audio_result(self, initial_request, audio_artifact):
        """Build a finished synthesis result carrying ``audio_artifact``."""
        result = self._build_result(initial_request, "", SpeechSynthesisRequestOutput)
        result.result_data.audio_artifact = audio_artifact
        return result


    def _execute_melotts_synthesizer(self, inputs, requests_to_batch):
        assert_batch_aligned("_execute_melotts_synthesizer", inputs, requests_to_batch)
        # Get model flavour if applicable
        flavour = get_model_flavour(self.model_config)
        # Load config
        hf_config = get_model_config(
            self.model_config.model, self.model_config.model_type, self.model_config.revision, flavour
        )
        sample_rate = int(hf_config.data.sampling_rate)
        # VITS hop: one decoder frame == hop_length samples (MeloTTS English v3: 512).
        hop_length = int(hf_config.data.hop_length)
        # Phase 4-1: prepared BERT features arrive as float32 host tensors; the
        # text encoder's projections follow the model dtype (float16 on GPU).
        model_dtype = next(self.model.parameters()).dtype

        def _assemble(audio_segments, speed_modifier):
            """Concatenate per-sentence waveforms with the historical 50 ms gap."""
            audio_segments_out = []
            gap = int((sample_rate * 0.05) / speed_modifier)
            for segment_data in audio_segments:
                audio_segments_out += np.asarray(segment_data).reshape(-1).tolist()
                audio_segments_out += [0] * gap
            return np.array(audio_segments_out).astype(np.float32)

        def synthesize_text(input_params, initial_request):
            """Per-sentence loop: the ``len(inputs) == 1`` fast path / fallback."""
            audio_segment_list = []
            inference_items, speaker_id, speed_modifier, sampling = input_params
            for items in inference_items:
                # rebuild items from tuple
                bert, ja_bert, phones, tones, lang_ids = items

                # Inference
                with torch.no_grad():
                    x_tst = phones.to(self.device).unsqueeze(0)
                    tones = tones.to(self.device).unsqueeze(0)
                    lang_ids = lang_ids.to(self.device).unsqueeze(0)
                    bert = bert.to(self.device, dtype=model_dtype).unsqueeze(0)
                    ja_bert = ja_bert.to(self.device, dtype=model_dtype).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
                    del phones
                    speakers = torch.LongTensor([speaker_id]).to(self.device)
                    audio = (
                        self.model.infer(
                            x_tst,
                            x_tst_lengths,
                            speakers,
                            tones,
                            lang_ids,
                            bert,
                            ja_bert,
                            sdp_ratio=sampling["sdp_ratio"],
                            noise_scale=sampling["noise_scale"],
                            noise_scale_w=sampling["noise_scale_w"],
                            length_scale=1.0 / speed_modifier,
                        )[0][0, 0]
                        .data.cpu()
                        .float()
                        .numpy()
                    )
                    del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                audio_segment_list.append(audio)

            return AudioArtifact.from_numpy(_assemble(audio_segment_list, speed_modifier), sample_rate)

        # ------------------------------------------------------------------
        # Padded-batch path (Phase 5-2): flatten (request, sentence) rows,
        # pad phones/tones/language ids and the BERT feature matrices, one
        # padded forward per length-similar sub-batch, unpad by y_lengths.
        # ------------------------------------------------------------------
        if batched_inference_enabled(self.model_config) and len(inputs) > 1:
            rows = [
                (index, item, speaker_id, speed_modifier, sampling)
                for index, (inference_items, speaker_id, speed_modifier, sampling) in enumerate(inputs)
                for item in inference_items
            ]
            if rows:
                try:
                    segment_audio: list[list[np.ndarray]] = [[] for _ in inputs]
                    lengths = [int(row[1][2].numel()) for row in rows]
                    batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
                    budget = self.model_config.batch_token_budget or default_token_budget(
                        lengths, batch_size, max_budget=DEFAULT_PHONE_BATCH_BUDGET
                    )
                    groups = plan_batches(lengths, batch_size, budget)
                    for group in groups:
                        group_rows = [rows[index] for index in group]
                        with torch.no_grad():
                            padded, padded_lengths = pad_phone_batch(
                                [row[1][2] for row in group_rows], pad_id=0, device=self.device
                            )
                            padded_tones, _ = pad_phone_batch(
                                [row[1][3] for row in group_rows], pad_id=0, device=self.device
                            )
                            padded_lang, _ = pad_phone_batch(
                                [row[1][4] for row in group_rows], pad_id=0, device=self.device
                            )
                            padded_bert = pad_feature_batch(
                                [row[1][0].to(dtype=model_dtype) for row in group_rows], device=self.device
                            )
                            padded_ja_bert = pad_feature_batch(
                                [row[1][1].to(dtype=model_dtype) for row in group_rows], device=self.device
                            )
                            speakers = torch.tensor(
                                [row[2] for row in group_rows], dtype=torch.long, device=self.device
                            )
                            waveforms, y_lengths, _, _, _ = self.model.infer_batched(
                                padded,
                                padded_lengths,
                                speakers,
                                padded_tones,
                                padded_lang,
                                padded_bert,
                                padded_ja_bert,
                                sdp_ratio=collapse_per_sample(row[4]["sdp_ratio"] for row in group_rows),
                                noise_scale=collapse_per_sample(row[4]["noise_scale"] for row in group_rows),
                                noise_scale_w=collapse_per_sample(row[4]["noise_scale_w"] for row in group_rows),
                                length_scale=collapse_per_sample(1.0 / row[3] for row in group_rows),
                            )
                            trimmed = trim_decoder_output(waveforms, y_lengths, hop_length)
                        for position, absolute in enumerate(group):
                            segment_audio[rows[absolute][0]].append(
                                trimmed[position][0].detach().cpu().float().numpy()
                            )
                    logger.debug(
                        f"MeloTTS: {len(rows)} sentence(s) from {len(inputs)} request(s) "
                        f"forwarded in {len(groups)} padded sub-batch(es)"
                    )
                    return [
                        self._build_audio_result(
                            requests_to_batch[index],
                            AudioArtifact.from_numpy(_assemble(segments, inputs[index][2]), sample_rate),
                        )
                        for index, segments in enumerate(segment_audio)
                    ]
                except Exception as e:  # noqa: BLE001 - degrade to the per-item loop
                    logger.warning(
                        f"MeloTTS batched synthesis failed ({e}); falling back to the per-request loop"
                    )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = synthesize_text(x, initial_request)
                result = self._build_result(initial_request, "", SpeechSynthesisRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs


    def _execute_voicefixer_restorer(self, inputs, requests_to_batch):
        """Execute VoiceFixerRestorer model to enhance audio quality."""
        assert_batch_aligned("_execute_voicefixer_restorer", inputs, requests_to_batch)
        # The vendored forward processes a clip in 30 s slices and concatenates the
        # per-slice mels, so the batched row unit is the ``(request, slice)`` pair
        # (the same shape as 5-2's ``(request, sentence)`` rows).
        segment_samples = int(getattr(self.model, "sample_rate", 44100)) * 30

        def audio_segments(audio_artifact):
            """Device waveform slices, ``[1, 1, samples]`` each."""
            audio_tensor = _bind_to_device(audio_artifact, self.device).waveform
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(1)
            return [
                audio_tensor[..., start : start + segment_samples]
                for start in range(0, int(audio_tensor.shape[-1]), segment_samples)
            ]

        def mel_frontend(segment):
            """Per-slice mel frontend: the historical STFT/mel path, per clip.

            It stays per row because the centered STFT reflect-pads every clip on
            its own (the 8-2/8-3 frontend rule); the batch is built on the frames.
            """
            return self.model.preprocess(segment)

        def mel_artifact(segment_mels):
            """Fold a request's per-slice mels exactly like ``model.forward``.

            The solo path collects ``[1, 1, frames, bins]`` tensors per slice and
            concatenates them along the historical axis (``dim=3``) when there is
            more than one; a single slice is returned as-is. Both shapes are
            preserved here verbatim.
            """
            enhanced = (
                segment_mels[0] if len(segment_mels) == 1 else torch.cat(segment_mels, dim=3)
            )
            # VoiceFixer emits [batch, channels, frames, bins]; the vocoder only
            # consumes channel 0, so store the canonical [1, bins, frames].
            mel = enhanced.detach().cpu().float()
            if mel.dim() == 4:
                mel = mel[:, 0].transpose(-1, -2)
            elif mel.dim() == 3:
                mel = mel.transpose(-1, -2)
            else:
                raise ValueError(f"VoiceFixerRestorer returned an unexpected mel shape {tuple(mel.shape)}")
            return MelArtifact(
                mel=mel.contiguous(),
                sample_rate=44100,
                hop_length=441,
            )

        def restore_audio(audio_artifact, initial_request):
            """Per-item loop: the ``len == 1`` fast path and the fallback."""
            # [1, samples] -> [batch, channels, samples]
            audio_tensor = _bind_to_device(audio_artifact, self.device).waveform
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(1)

            # Run the VoiceFixerRestorer model
            with torch.no_grad():
                enhanced_mel = self.model(audio_tensor)
            return mel_artifact([enhanced_mel])

        # ------------------------------------------------------------------
        # Padded-batch path (Phase 8-4). Rows are ``(request, 30 s slice)`` pairs
        # on the mel-frame axis: the UNet's conv/pool stack would otherwise let a
        # padded row's padding reach its valid tail (`forward_batched` masks every
        # stage). Per-request outputs are reassembled in slice order.
        # ------------------------------------------------------------------
        batched: dict[int, MelArtifact] = {}
        if batched_inference_enabled(self.model_config) and len(inputs) > 1:
            rows: list[tuple[int, torch.Tensor]] = []
            for index, audio_artifact in enumerate(inputs):
                try:
                    for segment in audio_segments(audio_artifact):
                        rows.append((index, mel_frontend(segment)))
                except Exception as e:  # noqa: BLE001 - the loop reports this request alone
                    logger.warning(
                        f"VoiceFixer restorer: mel frontend failed for request "
                        f"{requests_to_batch[index].request_id} ({e}); per-item loop will report it"
                    )
            if rows:
                frame_lengths = [int(mel.shape[2]) for _index, mel in rows]
                batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
                budget = self.model_config.batch_token_budget or default_token_budget(
                    frame_lengths, batch_size
                )
                groups = plan_batches(frame_lengths, batch_size, budget)
                per_request: dict[int, list[torch.Tensor]] = {index: [] for index, _ in rows}
                forwarded = 0
                for group in groups:
                    group_rows = [rows[position] for position in group]
                    try:
                        with torch.no_grad():
                            padded, padded_lengths = _pad_mel_batch(
                                [mel for _index, mel in group_rows], self.device
                            )
                            restored = self.model.forward_batched(padded, padded_lengths)
                            trimmed = trim_feature_output(restored, padded_lengths, dim=2)
                    except Exception as e:  # noqa: BLE001 - degrade this sub-batch to the loop
                        logger.warning(
                            f"VoiceFixer restorer batched forward failed ({e}); "
                            f"falling back to the per-request loop for {len(group_rows)} request(s)"
                        )
                        continue
                    forwarded += len(group_rows)
                    for position, _absolute in enumerate(group):
                        index = group_rows[position][0]
                        per_request[index].append(trimmed[position].unsqueeze(0))
                for index, slices in per_request.items():
                    if len(slices) == len(audio_segments(inputs[index])):
                        try:
                            batched[index] = mel_artifact(slices)
                        except Exception as e:  # noqa: BLE001 - the loop reports this request alone
                            logger.warning(
                                f"VoiceFixer restorer: could not reassemble request "
                                f"{requests_to_batch[index].request_id} ({e}); "
                                f"per-item loop will report it"
                            )
                if batched:
                    logger.debug(
                        f"VoiceFixer restorer: {forwarded} slice(s) from {len(inputs)} request(s) "
                        f"forwarded in {len(groups)} padded sub-batch(es)"
                    )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                artifact = batched.get(i)
                if artifact is None:
                    artifact = restore_audio(x, initial_request)
                result = self._build_result(initial_request, "", AudioConversionRequestOutput)
                result.result_data.mel_artifact = artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_voicefixer_vocoder(self, inputs, requests_to_batch):
        """Execute VoiceFixerVocoder model to convert mel spectrogram to audio."""
        assert_batch_aligned("_execute_voicefixer_vocoder", inputs, requests_to_batch)

        def mel_frames(mel_artifact):
            """Device mel on the frame axis, ``[1, bins, frames]`` (artifact form)."""
            return _bind_to_device(mel_artifact, self.device).mel

        def normalize(audio_np):
            """Per-request peak normalization: the historical arithmetic."""
            if np.abs(audio_np).max() > 0:
                audio_np = audio_np / np.abs(audio_np).max() * 0.97
            return audio_np

        def vocode_mel(mel_artifact, initial_request):
            """Per-item loop: the ``len == 1`` fast path and the fallback."""
            # VoiceFixer's forward expects [batch, 1, frames, bins].
            mel_tensor = mel_frames(mel_artifact).transpose(-1, -2).unsqueeze(1)

            # Run the VoiceFixerVocoder model
            with torch.no_grad():
                audio_output = self.model(mel_tensor)

            # Convert to numpy and normalize
            audio_np = audio_output.detach().cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

            return AudioArtifact.from_numpy(normalize(audio_np), 44100)

        # ------------------------------------------------------------------
        # Padded-batch path (Phase 8-4). The mel-to-condition frontend stays per
        # request (`tr_pre`'s tail length depends on the row's own frame count);
        # the vocoder generator runs one masked forward per sub-batch, and the
        # peak normalization stays **per request** — a batch max would rescale a
        # quiet clip by its neighbour's loudest sample.
        # ------------------------------------------------------------------
        batched: dict[int, AudioArtifact] = {}
        if batched_inference_enabled(self.model_config) and len(inputs) > 1:
            rows: list[tuple[int, torch.Tensor]] = []
            for index, mel_artifact in enumerate(inputs):
                try:
                    rows.append((index, mel_frames(mel_artifact).squeeze(0)))
                except Exception as e:  # noqa: BLE001 - the loop reports this request alone
                    logger.warning(
                        f"VoiceFixer vocoder: mel frontend failed for request "
                        f"{requests_to_batch[index].request_id} ({e}); per-item loop will report it"
                    )
            if rows:
                frame_lengths = [int(mel.shape[-1]) for _index, mel in rows]
                batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
                budget = self.model_config.batch_token_budget or default_token_budget(
                    frame_lengths, batch_size
                )
                groups = plan_batches(frame_lengths, batch_size, budget)
                forwarded = 0
                for group in groups:
                    group_rows = [rows[position] for position in group]
                    try:
                        with torch.no_grad():
                            padded, padded_lengths = pad_feature_batch(
                                [mel for _index, mel in group_rows],
                                device=self.device,
                                return_lengths=True,
                            )
                            mel_batch = padded.transpose(-1, -2).unsqueeze(1)
                            audio, sample_lengths = self.model.forward_batched(
                                mel_batch, padded_lengths
                            )
                            trimmed = trim_decoder_output(audio, sample_lengths, 1)
                    except Exception as e:  # noqa: BLE001 - degrade this sub-batch to the loop
                        logger.warning(
                            f"VoiceFixer vocoder batched forward failed ({e}); "
                            f"falling back to the per-request loop for {len(group_rows)} request(s)"
                        )
                        continue
                    forwarded += len(group_rows)
                    for position, _absolute in enumerate(group):
                        index = group_rows[position][0]
                        audio_np = trimmed[position][0].detach().cpu().float().numpy()
                        batched[index] = AudioArtifact.from_numpy(normalize(audio_np), 44100)
                if batched:
                    logger.debug(
                        f"VoiceFixer vocoder: {forwarded} request(s) forwarded in "
                        f"{len(groups)} padded sub-batch(es)"
                    )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = batched.get(i)
                if audio_artifact is None:
                    audio_artifact = vocode_mel(x, initial_request)
                result = self._build_result(initial_request, "", AudioConversionRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_silero_vad(self, inputs, requests_to_batch):
        """Execute Silero VAD model to detect voice activity.

        Phase 8-5 verdict: **sequential, no lanes.** Two independent reasons,
        both measured on the installed packages:

        * Batching is impossible — ``get_speech_timestamps`` is a single-stream
          API: one 512-sample window at a time through ``model(chunk, sr)``,
          producing a probability list that the post-processor turns into
          segments. A batch axis would mean re-implementing that windowing loop
          and the upstream ONNX graph's state protocol.
        * Parallel lanes are worse than useless here. Unlike a one-shot graph,
          the VAD is *many tiny* inferences with Python glue between them, and
          the installed ``silero_vad.OnnxWrapper`` keeps the streaming LSTM state
          on the **instance** (``_state`` / ``_context``) and calls
          ``reset_states()`` at the start of every ``get_speech_timestamps`` —
          concurrent calls on the shared model corrupt each other (measured: 7 of
          16 concurrent results differed from their sequential answer). Lanes on
          per-lane wrappers avoid the corruption but still measured 1.10x / 0.95x
          / 0.92x at 4 / 8 / 16 lanes on 16 x 30 s clips: the GIL-bound Python
          loop dominates, exactly as expected for a step this cheap.

        The per-request loop is therefore the honest implementation, and
        ``max_batch_size`` stays an admission/queue-depth setting. Numbers and the
        verdict are recorded in ``docs/batching.md``.
        """
        assert_batch_aligned("_execute_silero_vad", inputs, requests_to_batch)

        def run_vad(input_params, initial_request):
            audio_artifact, vad_params = input_params

            # Silero runs on CPU tensors; the move is explicit.
            audio_tensor = _bind_to_device(audio_artifact, "cpu").waveform.squeeze(0)

            # Extract parameters from the input object
            threshold = vad_params["threshold"]
            min_speech_duration_ms = vad_params["min_speech_duration_ms"]
            min_silence_duration_ms = vad_params["min_silence_duration_ms"]
            speech_pad_ms = vad_params["speech_pad_ms"]
            return_seconds = vad_params["return_seconds"]
            get_timestamps = vad_params["get_timestamps"]

            # Run Silero VAD with dynamic parameters
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=return_seconds,
            )

            # Build response based on whether timestamps are requested
            if get_timestamps:
                response = {"speech_activity": len(timestamps) > 0, "segments": timestamps}
            else:
                response = {"speech_activity": len(timestamps) > 0}

            # The artifact always carries whisper-schema segments (Phase 3-4):
            # the response shape stays request-dependent, but a pipeline VAD
            # step needs the segments regardless of the endpoint's flags.
            return json.dumps(response), VadArtifact.from_silero_timestamps(
                timestamps, return_seconds=return_seconds
            )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                inference_result, vad_artifact = run_vad(x, initial_request)
                result = self._build_result(initial_request, inference_result, DetectVoiceActivityRequestOutput)
                result.result_data.vad_artifact = vad_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_kittentts_synthesizer(self, inputs, requests_to_batch):
        """Synthesize the step's requests as parallel lanes (KittenTTS cannot batch)."""
        assert_batch_aligned("_execute_kittentts_synthesizer", inputs, requests_to_batch)

        def synthesize_text(input_params):
            input_text, voice, speed_modifier = input_params

            # Run KittenTTS generation
            # Returns numpy float32 array at 24000 Hz
            audio_array = self.model.generate(text=input_text, voice=voice, speed=speed_modifier, clean_text=True)

            # Flatten to 1D if needed (KittenTTS may return 2D)
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()

            return AudioArtifact.from_numpy(audio_array, self.model.sample_rate)

        # ------------------------------------------------------------------
        # Parallel lanes (Phase 8-5). KittenTTS cannot batch: all four cached
        # variants export a static batch dimension of 1 (``input_ids [1, len]``,
        # ``style [1, 256]``, ``speed [1]``). A step therefore runs one request
        # per lane — ``max_batch_size`` *is* the lane count, no new config key —
        # over one shared ORT session (``InferenceSession.run`` is thread-safe
        # and releases the GIL, and the graph is one-shot, so a lane is a single
        # long C++ call and the lanes scale). The phonemizer inside
        # ``generate`` serialises itself (espeak is not thread-safe).
        # ------------------------------------------------------------------
        lanes = lane_count(len(inputs), getattr(self.model_config, "max_batch_size", 1))
        lane_results = run_in_lanes(synthesize_text, inputs, lanes, thread_name_prefix="kitten-lane")
        if lanes > 1:
            logger.debug(f"KittenTTS: {len(inputs)} request(s) synthesised in {lanes} lane(s)")

        # Result assembly stays on the calling thread (``_build_result`` owns the
        # per-step duplicate-id guard) and in input order, so a failed lane only
        # ever becomes its own request's error result.
        outputs = []
        for i, (audio_artifact, error) in enumerate(lane_results):
            initial_request = requests_to_batch[i]
            if error is not None:
                outputs.append(self._build_error_result(initial_request, error))
                continue
            try:
                result = self._build_result(initial_request, "", TextToSpeechRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    # ------------------------------------------------------------------
    # Shared-weight invariant (Phase 4 dependency)
    # ------------------------------------------------------------------
    # Per-request state lives on the *wrapper executor object*
    # (``self.model.conds`` / ``self.model.ref_dict``) — never on shared
    # backbone submodules (S3Gen / VoiceEncoder). Executors that share backbones
    # must keep distinct wrapper instances or distinct per-request state slots.
    # Every handler below mutates the wrapper only, and restores the baseline
    # captured at load time in a ``finally`` block so no request can ever
    # inherit the previous request's voice.
    # ------------------------------------------------------------------

    def _execute_chatterbox_tts(self, inputs, requests_to_batch):
        """Execute ChatterboxTTS: AR batch runner when eligible, else the loop."""
        assert_batch_aligned("_execute_chatterbox_tts", inputs, requests_to_batch)
        runner = self._get_ar_runner()
        if runner is None or not runner.enabled:
            return self._execute_chatterbox_tts_sequential(inputs, requests_to_batch)
        return self._execute_chatterbox_tts_ar(runner, inputs, requests_to_batch, turbo=False)

    def _execute_chatterbox_tts_sequential(self, inputs, requests_to_batch):
        """Per-request loop: the ``len == 1`` fast path and the fallback."""
        assert_batch_aligned("_execute_chatterbox_tts_sequential", inputs, requests_to_batch)

        def synthesize(input_params, initial_request):
            text, speaker, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p = input_params

            conditionals = _chatterbox_conditionals(speaker, self.device)
            if conditionals is not None:
                # Pre-computed conditionals passed by reference from the embed step
                self.model.conds = conditionals
            elif speaker is not None and speaker.reference_audio is not None:
                # Clone from the already-decoded reference artifact (no wire encoding/tempfile)
                self.model.prepare_conditionals(
                    _wav_buffer(speaker.reference_audio), exaggeration=exaggeration
                )
            else:
                # No per-request conditioning: use the voice loaded with the model
                # (Chatterbox bundles `conds.pt`). A deep copy keeps the baseline
                # immutable — upstream `generate` mutates `conds.t3.emotion_adv`.
                baseline = (getattr(self, "_baseline_state", None) or {}).get("conds")
                if baseline is None and getattr(self.model, "conds", None) is None:
                    raise ValueError("voice cloning requires input_audio or a precomputed embedding/conditionals")
                self.model.conds = copy.deepcopy(baseline)

            # Phase 4-1: match conditional tensors to the T3 dtype (fp16 rollout).
            self._align_chatterbox_conditionals(exaggeration)

            suppression = (
                self._without_watermark()
                if _watermark_suppressed(initial_request)
                else contextlib.nullcontext()
            )
            with suppression:
                wav = self.model.generate(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    min_p=min_p,
                )

            return _audio_artifact_from_tensor(wav, self.model.sr)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = synthesize(x, initial_request)
                result = self._build_result(initial_request, "", TextToSpeechRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            finally:
                self._restore_chatterbox_state()
            outputs.append(result)
        return outputs

    def _execute_chatterbox_turbo_tts(self, inputs, requests_to_batch):
        """Execute ChatterboxTurboTTS: AR batch runner when eligible, else the loop."""
        assert_batch_aligned("_execute_chatterbox_turbo_tts", inputs, requests_to_batch)
        runner = self._get_ar_runner()
        if runner is None or not runner.enabled:
            return self._execute_chatterbox_turbo_tts_sequential(inputs, requests_to_batch)
        return self._execute_chatterbox_tts_ar(runner, inputs, requests_to_batch, turbo=True)

    def _execute_chatterbox_turbo_tts_sequential(self, inputs, requests_to_batch):
        """Per-request loop: the ``len == 1`` fast path and the fallback."""
        assert_batch_aligned("_execute_chatterbox_turbo_tts_sequential", inputs, requests_to_batch)

        def synthesize(input_params, initial_request):
            text, speaker, temperature, repetition_penalty, top_p, top_k, norm_loudness = input_params

            conditionals = _chatterbox_conditionals(speaker, self.device)
            if conditionals is not None:
                self.model.conds = conditionals
            elif speaker is not None and speaker.reference_audio is not None:
                self.model.prepare_conditionals(
                    _wav_buffer(speaker.reference_audio), norm_loudness=norm_loudness
                )
            else:
                baseline = (getattr(self, "_baseline_state", None) or {}).get("conds")
                if baseline is None and getattr(self.model, "conds", None) is None:
                    raise ValueError("voice cloning requires input_audio or a precomputed embedding/conditionals")
                self.model.conds = copy.deepcopy(baseline)

            # Phase 4-1: match conditional tensors to the T3 dtype (fp16 rollout).
            self._align_chatterbox_conditionals()

            suppression = (
                self._without_watermark()
                if _watermark_suppressed(initial_request)
                else contextlib.nullcontext()
            )
            with suppression:
                wav = self.model.generate(
                    text,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    top_k=top_k,
                    norm_loudness=norm_loudness,
                )

            return _audio_artifact_from_tensor(wav, self.model.sr)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = synthesize(x, initial_request)
                result = self._build_result(initial_request, "", TextToSpeechRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            finally:
                self._restore_chatterbox_state()
            outputs.append(result)
        return outputs

    def _execute_chatterbox_multilingual_tts(self, inputs, requests_to_batch):
        """Execute ChatterboxMultilingualTTS: AR runner when eligible, else the loop."""
        assert_batch_aligned("_execute_chatterbox_multilingual_tts", inputs, requests_to_batch)
        runner = self._get_ar_runner()
        if runner is None or not runner.enabled:
            return self._execute_chatterbox_multilingual_tts_sequential(inputs, requests_to_batch)
        return self._execute_chatterbox_tts_ar(
            runner, inputs, requests_to_batch, turbo=False, multilingual=True
        )

    def _execute_chatterbox_multilingual_tts_sequential(self, inputs, requests_to_batch):
        """Per-request loop: the ``len == 1`` fast path and the fallback."""
        assert_batch_aligned("_execute_chatterbox_multilingual_tts_sequential", inputs, requests_to_batch)

        def synthesize(input_params, initial_request):
            text, language_id, speaker, exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p = (
                input_params
            )

            conditionals = _chatterbox_conditionals(speaker, self.device)
            if conditionals is not None:
                self.model.conds = conditionals
            elif speaker is not None and speaker.reference_audio is not None:
                self.model.prepare_conditionals(
                    _wav_buffer(speaker.reference_audio), exaggeration=exaggeration
                )
            else:
                baseline = (getattr(self, "_baseline_state", None) or {}).get("conds")
                if baseline is None and getattr(self.model, "conds", None) is None:
                    raise ValueError("voice cloning requires input_audio or a precomputed embedding/conditionals")
                self.model.conds = copy.deepcopy(baseline)

            # Phase 4-1: match conditional tensors to the T3 dtype (fp16 rollout).
            self._align_chatterbox_conditionals(exaggeration)

            suppression = (
                self._without_watermark()
                if _watermark_suppressed(initial_request)
                else contextlib.nullcontext()
            )
            # Phase 12-7: upstream's multilingual `generate` forgets the OOV
            # filter English applies (`speech_tokens < 6561`). Wrap only this
            # wrapper's S3Gen seam; the shared module is restored in `finally`.
            s3gen = getattr(self.model, "s3gen", None)
            filtered = _FilteredS3Gen(s3gen) if s3gen is not None else None
            if filtered is not None:
                self.model.s3gen = filtered
            try:
                with suppression:
                    wav = self.model.generate(
                        text,
                        language_id=language_id,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p,
                        min_p=min_p,
                    )
            finally:
                if filtered is not None:
                    self.model.s3gen = s3gen

            return _audio_artifact_from_tensor(wav, self.model.sr)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = synthesize(x, initial_request)
                result = self._build_result(initial_request, "", TextToSpeechRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            finally:
                self._restore_chatterbox_state()
            outputs.append(result)
        return outputs

    def _execute_chatterbox_embedding(self, inputs, requests_to_batch):
        """Execute ChatterboxEmbedding to compute voice Conditionals from audio.

        Architecture note (REQ-PERF-02):
        The result is a :class:`SpeakerArtifact` carrying the live ``Conditionals``
        object, forwarded to the TTS/VC step by reference. External embed clients
        still receive the 0-1-safe wire payload — that serialization happens
        only in the serving layer (``wire.to_wire_embedding``), never here.

        The cross-request cache (Phase 7-2) sits at this step: repeated
        reference audio is served as a per-request deep copy of the stored
        ``Conditionals`` and skips ``prepare_conditionals`` entirely. The
        ``exaggeration`` option is part of the cache key (it bakes into
        ``conditionals.t3.emotion_adv``).

        Payload format (wire only): ``{"t3": {<T3Cond fields>}, "gen": {<S3Gen ref
        fields>}}`` — a plain dict of tensors/primitives
        (``common.serialization.conditionals_to_payload``), reloaded with
        ``weights_only=True`` on the consumer side.
        """
        assert_batch_aligned("_execute_chatterbox_embedding", inputs, requests_to_batch)

        def compute_embedding(audio_artifact, initial_request):
            audio_buf = _wav_buffer(_bind_to_device(audio_artifact, "cpu"))
            try:
                self.model.prepare_conditionals(audio_buf)
            except AssertionError as e:
                # Upstream library uses AssertionError for input validation (e.g. "Audio prompt must be longer than 5 seconds!")
                # Re-raise as ValueError so the engine can return a 400 Bad Request instead of crashing.
                raise ValueError(str(e)) from e
            conditionals = self.model.conds
            return SpeakerArtifact(conditionals=conditionals, backend="chatterbox")

        # Cross-request cache (Phase 7-2): resolve hits before any compute;
        # served conditionals are deep copies, so the baseline-restore wrappers
        # below are untouched (the cache sits above them, not inside).
        cache = getattr(self, "embedding_cache", None)
        slots = self._resolve_cache_slots(
            cache,
            requests_to_batch,
            lambda i: self._embedding_cache_lookup(
                inputs[i], self._chatterbox_embedding_params(requests_to_batch[i])
            ),
        )

        outputs = []
        for i, audio_artifact in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                key, cached = slots[i]
                if cached is not None:
                    embedding = cached
                else:
                    embedding = compute_embedding(audio_artifact, initial_request)
                    self._store_embedding_cache_entry(key, embedding)
                result = self._build_result(initial_request, "", SpeechEmbeddingRequestOutput)
                result.result_data.embedding_artifact = embedding
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            finally:
                self._restore_chatterbox_state()
            outputs.append(result)
        return outputs

    def _execute_chatterbox_vc(self, inputs, requests_to_batch):
        """Execute ChatterboxVC to perform voice conversion.

        The target-voice preparation (``set_target_voice`` on the decoded
        target audio) goes through the cross-request cache (Phase 7-2): in a
        VC flood the *target* clip is the repeated input. The cached value is
        the ``ref_dict`` tensor dict wrapped in a ``SpeakerArtifact``; a hit
        assigns the per-request deep copy to ``model.ref_dict`` and skips the
        audio pass entirely. Pre-computed conditionals always win over the
        cache (an explicit embedding beats any inference).
        """
        assert_batch_aligned("_execute_chatterbox_vc", inputs, requests_to_batch)

        # Cross-request cache (Phase 7-2): only the "target audio" path computes
        # anything cacheable; precomputed-conditionals and missing-target
        # requests keep a (None, None) slot.
        cache = getattr(self, "embedding_cache", None)

        def _lookup(index):
            _source, target_speaker, target_audio = inputs[index]
            if target_speaker is not None or target_audio is None:
                return None, None
            return self._embedding_cache_lookup(target_audio, {})

        slots = self._resolve_cache_slots(cache, requests_to_batch, _lookup)

        # Phase 9-5 batched the VC vocode hop; Phase 11-1 adds the batched source
        # tokenizer, so a multi-request step has zero serial model forwards. The
        # shipped width is 8; an explicit `batched_inference: false` keeps the
        # per-item loop.
        if len(inputs) > 1 and batched_inference_enabled(self.model_config):
            batched = self._try_batched_chatterbox_vc(inputs, requests_to_batch, slots)
            if batched is not None:
                return batched

        def convert(input_params, initial_request, index):
            source_artifact, target_speaker, target_audio_artifact = input_params
            key, cached = slots[index]

            conditionals = _chatterbox_conditionals(target_speaker, self.device)
            if conditionals is not None:
                # Pre-computed Conditionals — extract .gen dict and assign to model.ref_dict
                self.model.ref_dict = conditionals.gen
            elif cached is not None:
                # Cache hit on the target voice: the served dict is this
                # request's own deep copy (Phase 0-2 isolation).
                self.model.ref_dict = cached.conditionals
            elif target_audio_artifact is not None:
                # Use the already-decoded target audio artifact to set the target voice
                self.model.set_target_voice(_wav_buffer(_bind_to_device(target_audio_artifact, "cpu")))
                self._store_embedding_cache_entry(
                    key,
                    SpeakerArtifact(conditionals=self.model.ref_dict, backend="chatterbox"),
                )
            else:
                raise ValueError("voice conversion requires target_audio or target_embedding")

            # Phase 11-1: a seeded VC request seeds the vocode noise exactly like
            # the batched path's per-row generator, so a batch failure is a
            # performance fallback, not an audio change.
            seed = self._request_seed(initial_request)
            with self._seeded_rng(seed):
                wav = self.model.generate(
                    audio=_wav_buffer(_bind_to_device(source_artifact, "cpu"))
                )

            return _audio_artifact_from_tensor(wav, self.model.sr)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = convert(x, initial_request, i)
                result = self._build_result(initial_request, "", VoiceConversionRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            finally:
                self._restore_chatterbox_state()
            outputs.append(result)
        return outputs

    @staticmethod
    def _request_seed(request) -> int | None:
        """The request's optional seed (``generation_options.seed``)."""
        options = getattr(request.request_data, "generation_options", None)
        if options is None:
            return None
        return getattr(options, "seed", None)

    def _chatterbox_vc_clip(self, source_artifact):
        """Decode a VC source clip to 16 kHz mono, exactly as upstream.

        The per-row ``librosa`` decode is deliberately kept (Phase 11-1 fences
        decode hygiene out of scope: ``librosa``'s resampler differs numerically
        from ``torchaudio``'s, so changing it would break token-level
        bit-identity with the sequential fallback).
        """
        import librosa
        from chatterbox.models.s3tokenizer import S3_SR

        audio_16, _ = librosa.load(
            _wav_buffer(_bind_to_device(source_artifact, "cpu")), sr=S3_SR
        )
        return torch.from_numpy(audio_16).float()

    def _try_batched_chatterbox_vc(self, inputs, requests_to_batch, slots):
        """Batch a multi-request VC step; ``None`` = use the loop.

        Resolves each request's target ``ref_dict`` (precomputed / cache /
        ``set_target_voice``), decodes each source clip (per-row ``librosa``
        decode kept), tokenizes every clip in **one** :func:`tokenize_batched`
        pass and vocodes the rows in length-similar sub-batches. Any
        preparation or batch failure returns ``None`` so the caller falls back
        to the per-item loop, where a poisoned request errors alone.
        """
        from harmonyspeech.modeling.models.chatterbox.s3gen_batch import (
            S3GenRow,
            inference_batched,
            supports_batched_s3gen,
            supports_batched_tokenizer,
            tokenize_batched,
        )

        s3gen = getattr(self.model, "s3gen", None)
        tokenizer = getattr(s3gen, "tokenizer", None)
        if not supports_batched_s3gen(s3gen) or not supports_batched_tokenizer(tokenizer):
            return None

        prepared: list[tuple[int, dict, int | None]] = []
        clips: list[torch.Tensor] = []
        try:
            for index, (source_artifact, target_speaker, target_audio_artifact) in enumerate(inputs):
                key, cached = slots[index]
                conditionals = _chatterbox_conditionals(target_speaker, self.device)
                if conditionals is not None:
                    ref_dict = conditionals.gen
                elif cached is not None:
                    ref_dict = cached.conditionals
                elif target_audio_artifact is not None:
                    self.model.set_target_voice(
                        _wav_buffer(_bind_to_device(target_audio_artifact, "cpu"))
                    )
                    ref_dict = dict(self.model.ref_dict)
                    self._store_embedding_cache_entry(
                        key,
                        SpeakerArtifact(conditionals=self.model.ref_dict, backend="chatterbox"),
                    )
                else:
                    raise ValueError("voice conversion requires target_audio or target_embedding")
                prepared.append(
                    (index, ref_dict, self._request_seed(requests_to_batch[index]))
                )
                clips.append(self._chatterbox_vc_clip(source_artifact))
            # One batched source-tokenizer pass for the whole step (Phase 11-1).
            tokens = tokenize_batched(tokenizer, clips)
        except Exception as error:  # noqa: BLE001 - degrade the whole step to the loop
            logger.warning(f"Chatterbox VC batch preparation failed ({error}); using the per-item loop")
            return None
        finally:
            # ``set_target_voice`` mutated ``model.ref_dict``; the rows hold their
            # own references, and the loop below re-resolves per request.
            self._restore_chatterbox_state()

        lengths = [int(token.numel()) for token in tokens]
        batch_size = max(1, int(getattr(self.model_config, "max_batch_size", 1)))
        budget = getattr(self.model_config, "batch_token_budget", None) or default_token_budget(
            lengths, batch_size
        )
        results: list = [None] * len(inputs)
        for group in plan_batches(lengths, batch_size, budget):
            try:
                rows = [
                    S3GenRow(
                        tokens=tokens[position],
                        ref_dict=prepared[position][1],
                        seed=prepared[position][2],
                    )
                    for position in group
                ]
                with torch.no_grad():
                    waveforms = inference_batched(s3gen, rows)
            except Exception as error:  # noqa: BLE001 - whole-step fallback
                logger.warning(f"Chatterbox VC padded batch failed ({error}); using the per-item loop")
                return None
            for position, waveform in zip(group, waveforms, strict=True):
                index = prepared[position][0]
                try:
                    artifact = self._artifact_from_waveform(
                        waveform,
                        watermark=not _watermark_suppressed(requests_to_batch[index]),
                    )
                    result = self._build_result(
                        requests_to_batch[index], "", VoiceConversionRequestOutput
                    )
                    result.result_data.audio_artifact = artifact
                    results[index] = result
                except Exception as error:  # noqa: BLE001 - errors this request alone
                    results[index] = self._build_error_result(requests_to_batch[index], error)
        return [result for result in results if result is not None]
