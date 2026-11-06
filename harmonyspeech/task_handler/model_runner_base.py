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
  path and the fallback.
"""

import copy
import io
import json
import time
from dataclasses import asdict

import numpy as np
import torch
from faster_whisper import BatchedInferencePipeline
from loguru import logger
from silero_vad import get_speech_timestamps

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
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
    VocodeRequestOutput,
    VoiceConversionRequestOutput,
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult

# Re-exported for tests and callers that used the Phase 0 helper names; the
# implementations live in ``common.serialization`` (weights_only=True loaders).
from harmonyspeech.common.serialization import conditionals_to_payload as _conditionals_to_payload  # noqa: F401
from harmonyspeech.common.serialization import load_embedding_tensor as _load_embedding_tensor  # noqa: F401
from harmonyspeech.modeling.loader import get_model, get_model_config, get_model_flavour
from harmonyspeech.modeling.models.harmonyspeech.common import preprocess_wav
from harmonyspeech.modeling.models.harmonyspeech.encoder.inputs import get_input_frames
from harmonyspeech.modeling.models.openvoice.mel_processing import spectrogram_torch
from harmonyspeech.task_handler.batching import (
    batched_inference_enabled,
    collapse_per_sample,
    default_token_budget,
    pad_feature_batch,
    pad_phone_batch,
    plan_batches,
    trim_decoder_output,
)
from harmonyspeech.task_handler.inputs import (
    as_request_input,
    get_audio_artifact,
    get_vad_artifact,
    prepare_inputs,
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


class ModelRunnerBase:
    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        is_driver_worker: bool = False,
        *args,
        weight_registry=None,
        embedding_cache=None,
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
        # Cache identity (name/type/dtype/repo + load_version); built on every
        # model load by `_refresh_model_identity`. None until the first load.
        self.model_identity: str | None = None

        self.device_config = device_config if device_config is not None else DeviceConfig()
        self.device = self.device_config.device

        self.model = None
        self.model_memory_usage = 0

        # Baseline Chatterbox conditioning state, captured right after the
        # model is loaded (see `_capture_baseline_state`). None means "not
        # captured yet"; `_restore_chatterbox_state` then treats the baseline
        # as "no conditionals".
        self._baseline_state: dict | None = None

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
        """Prepare inputs for one type-homogeneous batch and dispatch it."""
        # Wrap prepare_inputs so that input-validation failures (e.g. wrong
        # request type, bad payload encoding, unsupported option) are converted to
        # per-request error results instead of raising an unhandled exception
        # that kills the async engine's background loop.
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

    def _build_error_result(self, initial_request, exception: Exception):
        """Build an error ExecutorResult for a single request that failed during inference.
        Wraps in ExecutorResult so _process_model_outputs / check_forward_processing can handle it
        without crashing. The serving layer then checks finish_reason == 'error' on result_data
        and returns an HTTP error to the caller.
        """
        request_id = initial_request.request_id
        self._register_result_id(request_id, raise_on_duplicate=False)
        metrics = initial_request.metrics
        metrics.finished_time = time.time()
        # Log full stack trace so the server-side cause is always visible in logs.
        logger.opt(exception=exception).error(
            f"Inference error for request {request_id}: {exception}"
        )
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
        # FIXME: This is not properly batched
        assert_batch_aligned("_execute_harmonyspeech_encoder", inputs, requests_to_batch)

        def embed_utterance(audio_artifact, initial_request):
            # Numpy preprocessing runs on the artifact's waveform (CPU-side); the
            # move is explicit, never implicit.
            source = _bind_to_device(audio_artifact, "cpu")
            preprocessed = preprocess_wav(source.numpy, source_sr=source.sample_rate)
            input_frames = get_input_frames(preprocessed)
            utterances_tensor = torch.from_numpy(input_frames).to(self.device)
            partial_embeds = self.model(utterances=utterances_tensor).detach().cpu().numpy()
            # Compute the utterance embedding from the partial embeddings
            raw_embed = np.mean(partial_embeds, axis=0)
            embed = raw_embed / np.linalg.norm(raw_embed, 2)
            return SpeakerArtifact(
                embedding=torch.from_numpy(np.ascontiguousarray(embed, dtype=np.float32)),
                backend="harmonyspeech",
            )

        # Cross-request cache (Phase 7-2): resolve hits for the whole batch
        # first (guard once at entry — zero cost when no cache is attached),
        # compute only the misses below; outputs stay in request order.
        cache = getattr(self, "embedding_cache", None)
        slots = self._resolve_cache_slots(
            cache, requests_to_batch, lambda i: self._embedding_cache_lookup(inputs[i], {})
        )

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                key, cached = slots[i]
                if cached is not None:
                    embedding = cached
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
        # FIXME: This is not properly batched
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
        # FIXME: This is not properly batched
        assert_batch_aligned("_execute_harmonyspeech_vocoder", inputs, requests_to_batch)

        def vocode_mel(input_params, initial_request):
            mel_artifact, breaks = input_params
            # Explicit move; the artifact carries its source device.
            mel_tensor = _bind_to_device(mel_artifact, self.device).mel.squeeze(0)
            mel = mel_tensor / 4.0  # Fixme: make this configurable
            kwargs = {"c": mel.T}
            wav = self.model.inference(**kwargs).detach().cpu().numpy()
            wav = wav.squeeze(1)

            # Add breaks if defined
            if breaks is not None:
                # Add breaks - FIXME: Make this parameterized
                b_ends = np.cumsum(np.array(breaks) * 200)  # hop_size 200
                b_starts = np.concatenate(([0], b_ends[:-1]))
                wavs = [wav[start:end] for start, end in zip(b_starts, b_ends)]
                syn_breaks = [np.zeros(int(0.15 * 16000))] * len(breaks)  # Sample Rate 16000
                wav = np.concatenate([i for w, b in zip(wavs, syn_breaks) for i in (w, b)])

            # Normalize
            wav = wav / np.abs(wav).max() * 0.97

            return AudioArtifact.from_numpy(wav, 16000)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
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

        # FIXME: This is not properly batched
        assert_batch_aligned("_execute_openvoice_tone_converter_encoder", inputs, requests_to_batch)

        def run_converter_encoder(vad_audio_segments, initial_request):
            gs = []
            for segment in vad_audio_segments:
                y = _bind_to_device(segment, self.device).waveform
                y = spectrogram_torch(
                    y,
                    hf_config.data.filter_length,
                    hf_config.data.sampling_rate,
                    hf_config.data.hop_length,
                    hf_config.data.win_length,
                    center=False,
                ).to(self.device)
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

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                key, cached = slots[i]
                if cached is not None:
                    embedding = cached
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

        # FIXME: This is not properly batched
        assert_batch_aligned("_execute_openvoice_tone_converter", inputs, requests_to_batch)

        def run_converter_encoder(input_params, initial_request):
            audio_artifact, target_speaker, source_speaker = input_params

            # Both embedding blobs decoded through the safe loaders at admission
            # (``weights_only=True``), so a hostile pickle fails closed before
            # reaching this handler. Device moves are explicit here.
            if target_speaker is None or target_speaker.embedding is None:
                raise ValueError("input_embedding/target_embedding: missing embedding tensor")
            if source_speaker is None or source_speaker.embedding is None:
                raise ValueError("source_embedding: missing embedding tensor")
            target_embedding = _bind_to_device(target_speaker, self.device).embedding
            src_embedding = _bind_to_device(source_speaker, self.device).embedding

            y = _bind_to_device(audio_artifact, self.device).waveform.float()
            spec = spectrogram_torch(
                y,
                hf_config.data.filter_length,
                hf_config.data.sampling_rate,
                hf_config.data.hop_length,
                hf_config.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)

            with torch.no_grad():
                output_audio = self.model.voice_conversion(
                    spec, spec_lengths, sid_src=src_embedding, sid_tgt=target_embedding, tau=0.3
                )
                output_audio = output_audio[0][0, 0].data.cpu().float().numpy()

            return AudioArtifact.from_numpy(output_audio, hf_config.data.sampling_rate)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = run_converter_encoder(x, initial_request)
                result = self._build_result(initial_request, "", VoiceConversionRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_faster_whisper(self, inputs, requests_to_batch):
        assert_batch_aligned("_execute_faster_whisper", inputs, requests_to_batch)

        def run_batched_transcribe(audio_artifact, initial_request):
            # faster-whisper takes a mono numpy array; the artifact already holds
            # the boundary-decoded 16 kHz waveform.
            audio_ref = _bind_to_device(audio_artifact, "cpu").numpy
            batched_model = BatchedInferencePipeline(model=self.model)
            segments, info = batched_model.transcribe(audio_ref, batch_size=16)
            segment_data = []
            text = ""
            for segment in segments:
                text += segment.text
                segment_data.append(asdict(segment))

            response = {"text": text, "segments": segment_data, "info": asdict(info)}
            return json.dumps(response), VadArtifact.from_payload(response)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                inference_result, vad_artifact = run_batched_transcribe(x, initial_request)
                result = self._build_result(initial_request, inference_result, SpeechTranscriptionRequestOutput)
                result.result_data.vad_artifact = vad_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

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
                    budget = self.model_config.batch_token_budget or default_token_budget(lengths, batch_size)
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
                    budget = self.model_config.batch_token_budget or default_token_budget(lengths, batch_size)
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

        def restore_audio(audio_artifact, initial_request):
            # [1, samples] -> [batch, channels, samples]
            audio_tensor = _bind_to_device(audio_artifact, self.device).waveform
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(1)

            # Run the VoiceFixerRestorer model
            with torch.no_grad():
                enhanced_mel = self.model(audio_tensor)

            # VoiceFixer emits [batch, channels, frames, bins]; the vocoder only
            # consumes channel 0, so store the canonical [1, bins, frames].
            mel = enhanced_mel.detach().cpu().float()
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

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                mel_artifact = restore_audio(x, initial_request)
                result = self._build_result(initial_request, "", AudioConversionRequestOutput)
                result.result_data.mel_artifact = mel_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_voicefixer_vocoder(self, inputs, requests_to_batch):
        """Execute VoiceFixerVocoder model to convert mel spectrogram to audio."""
        assert_batch_aligned("_execute_voicefixer_vocoder", inputs, requests_to_batch)

        def vocode_mel(mel_artifact, initial_request):
            # VoiceFixer's forward expects [batch, 1, frames, bins].
            mel = _bind_to_device(mel_artifact, self.device).mel
            mel_tensor = mel.transpose(-1, -2).unsqueeze(1)

            # Run the VoiceFixerVocoder model
            with torch.no_grad():
                audio_output = self.model(mel_tensor)

            # Convert to numpy and normalize
            audio_np = audio_output.detach().cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

            # Normalize audio
            if np.abs(audio_np).max() > 0:
                audio_np = audio_np / np.abs(audio_np).max() * 0.97

            return AudioArtifact.from_numpy(audio_np, 44100)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = vocode_mel(x, initial_request)
                result = self._build_result(initial_request, "", AudioConversionRequestOutput)
                result.result_data.audio_artifact = audio_artifact
            except Exception as e:
                result = self._build_error_result(initial_request, e)
            outputs.append(result)
        return outputs

    def _execute_silero_vad(self, inputs, requests_to_batch):
        """Execute Silero VAD model to detect voice activity."""
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
        """Execute KittenTTSSynthesizer to generate speech audio from text."""
        assert_batch_aligned("_execute_kittentts_synthesizer", inputs, requests_to_batch)

        def synthesize_text(input_params, initial_request):
            input_text, voice, speed_modifier = input_params

            # Run KittenTTS generation
            # Returns numpy float32 array at 24000 Hz
            audio_array = self.model.generate(text=input_text, voice=voice, speed=speed_modifier, clean_text=True)

            # Flatten to 1D if needed (KittenTTS may return 2D)
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()

            return AudioArtifact.from_numpy(audio_array, self.model.sample_rate)

        outputs = []
        for i, x in enumerate(inputs):
            initial_request = requests_to_batch[i]
            try:
                audio_artifact = synthesize_text(x, initial_request)
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
        """Execute ChatterboxTTS to generate speech audio from text."""
        assert_batch_aligned("_execute_chatterbox_tts", inputs, requests_to_batch)

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
        """Execute ChatterboxTurboTTS to generate speech audio from text."""
        assert_batch_aligned("_execute_chatterbox_turbo_tts", inputs, requests_to_batch)

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
        """Execute ChatterboxMultilingualTTS to generate speech audio from text."""
        assert_batch_aligned("_execute_chatterbox_multilingual_tts", inputs, requests_to_batch)

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

            wav = self.model.generate(audio=_wav_buffer(_bind_to_device(source_artifact, "cpu")))

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
