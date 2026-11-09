import base64
import contextlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.audio_codec import AudioDecodeError, decode_audio
from harmonyspeech.common.metrics import RequestMetrics
from harmonyspeech.common.serialization import load_conditionals_payload, load_embedding_tensor
from harmonyspeech.endpoints.openai.protocol import (
    AudioConversionRequest,
    DetectVoiceActivityRequest,
    EmbedSpeakerRequest,
    SpeechTranscribeRequest,
    SynthesizeAudioRequest,
    TextToSpeechRequest,
    VocodeAudioRequest,
    VoiceConversionRequest,
)

#: Request fields that carry an audio container (base64 on the wire).
AUDIO_FIELDS: tuple[str, ...] = ("input_audio", "source_audio", "target_audio")

#: Request fields that carry a serialized embedding.
EMBEDDING_FIELDS: tuple[str, ...] = ("input_embedding", "target_embedding")

__all__ = [
    "AUDIO_FIELDS",
    "EMBEDDING_FIELDS",
    "artifact_store",
    "as_request_input",
    "decode_audio_field",
    "decode_conditionals_field",
    "decode_embedding_bytes",
    "decode_embedding_field",
    "decode_float32_field",
    "decode_json_field",
    "decode_mel_field",
    "get_audio_artifact",
    "get_mel_artifact",
    "get_speaker_artifact",
    "get_vad_artifact",
    "prefill_audio_artifacts",
]


def as_request_input(entry: Any):
    """Return the ``RequestInput`` for a prepare entry.

    Prepare functions accept either an ``EngineRequest`` (the engine path, which
    carries the artifact store) or a bare request input (tests and direct
    callers), so both spellings work. The check is an explicit type test rather
    than ``hasattr`` because test doubles auto-create arbitrary attributes.
    """
    from harmonyspeech.common.request import EngineRequest

    if isinstance(entry, EngineRequest):
        return entry.request_data
    return entry


def artifact_store(entry: Any) -> dict | None:
    """Return the artifact store of a prepare entry, if it has one."""
    store = getattr(entry, "artifacts", None)
    return store if isinstance(store, dict) else None


def decode_audio_field(payload: str | None, field_name: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode a base64 audio request field through the boundary codec.

    Any base64 or codec failure becomes a ``ValueError`` naming the request
    field, which the engine turns into a 400-style per-request error result
    instead of crashing the pipeline.
    """
    if not payload:
        raise ValueError(f"{field_name}: missing audio payload")
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise ValueError(f"{field_name}: invalid base64 audio payload: {exc}") from exc
    try:
        return decode_audio(raw, target_sr=target_sr)
    except AudioDecodeError as exc:
        raise ValueError(f"{field_name}: {exc}") from exc


def decode_json_field(payload: str | None, field_name: str) -> Any:
    if payload is None:
        raise ValueError(f"{field_name}: missing payload")
    try:
        return json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}: invalid JSON payload: {exc}") from exc


def decode_b64_json_field(payload: str | None, field_name: str) -> Any:
    """Decode ``base64(JSON)`` (the legacy double-encoded intermediate format)."""
    raw = _decode_b64_bytes(payload, field_name)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{field_name}: invalid base64(JSON) payload: {exc}") from exc


def _decode_b64_bytes(payload: str | None, field_name: str) -> bytes:
    if not payload:
        raise ValueError(f"{field_name}: missing payload")
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise ValueError(f"{field_name}: invalid base64 payload: {exc}") from exc


def decode_embedding_bytes(data: bytes | memoryview, field_name: str = "embedding") -> torch.Tensor:
    """Load a torch-saved embedding from raw bytes with ``weights_only=True``."""
    import io

    return load_embedding_tensor(io.BytesIO(bytes(data)), field_name)


def decode_embedding_field(payload: str | None, field_name: str) -> torch.Tensor:
    """Decode a base64 embedding (torch.save tensor) safely."""
    return decode_embedding_bytes(_decode_b64_bytes(payload, field_name), field_name)


def decode_float32_field(payload: str | None, field_name: str) -> torch.Tensor:
    """Decode a base64 raw-float32 embedding (HarmonySpeech encoder format)."""
    raw = _decode_b64_bytes(payload, field_name)
    array = np.frombuffer(raw, dtype=np.float32).copy()
    return torch.from_numpy(array)


def decode_conditionals_field(payload: str | None, field_name: str):
    """Decode a base64 Chatterbox conditionals blob safely (0-1 contract)."""
    return load_conditionals_payload(_decode_b64_bytes(payload, field_name), field_name)


def decode_mel_field(payload: str | None, field_name: str) -> np.ndarray:
    """Decode the legacy ``base64(JSON(list[float]))`` mel representation."""
    data = decode_b64_json_field(payload, field_name)
    try:
        return np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}: mel payload is not a numeric array: {exc}") from exc


# ---------------------------------------------------------------------------
# Precedence getters: artifact wins over the base64 field, decode once, cache
# ---------------------------------------------------------------------------


def get_audio_artifact(
    request: Any,
    entry: Any = None,
    field: str = "input_audio",
    target_sr: int | None = None,
) -> AudioArtifact:
    """Return the audio artifact for ``field``.

    Precedence: ``entry.artifacts[field]`` -> decode the base64 field once and
    cache it -> ``ValueError`` naming the field. The native-rate artifact is what
    gets cached; a step asking for a different ``target_sr`` receives a transient
    :meth:`AudioArtifact.resampled` view (never cached, so memory stays flat).
    """
    store = artifact_store(entry)
    existing = store.get(field) if store is not None else None
    if isinstance(existing, AudioArtifact):
        return existing.resampled(target_sr) if target_sr else existing

    payload = getattr(request, field, None)
    if not payload:
        raise ValueError(f"{field}: missing audio payload")
    waveform, sample_rate = decode_audio_field(payload, field)
    artifact = AudioArtifact.from_numpy(waveform, sample_rate)
    if store is not None:
        store[field] = artifact
    return artifact.resampled(target_sr) if target_sr else artifact


def get_speaker_artifact(
    request: Any,
    entry: Any = None,
    field: str = "input_embedding",
    backend: str = "openvoice",
) -> SpeakerArtifact:
    """Return the speaker artifact for ``field`` (embedding or conditionals).

    ``backend`` selects the payload encoding for the legacy base64 field:
    ``"openvoice"`` / ``"harmonyspeech"`` for torch-saved / raw-float32
    embeddings and ``"chatterbox"`` for the safe conditionals payload.
    """
    store = artifact_store(entry)
    existing = store.get(field) if store is not None else None
    if isinstance(existing, SpeakerArtifact):
        return existing

    payload = getattr(request, field, None)
    if payload is None or payload == "":
        raise ValueError(f"{field}: missing embedding payload")
    if backend == "chatterbox":
        artifact = SpeakerArtifact(
            conditionals=decode_conditionals_field(payload, field),
            backend="chatterbox",
        )
    elif backend == "harmonyspeech":
        artifact = SpeakerArtifact(
            embedding=decode_float32_field(payload, field),
            backend="harmonyspeech",
        )
    else:
        artifact = SpeakerArtifact(
            embedding=decode_embedding_field(payload, field),
            backend="openvoice",
        )
    if store is not None:
        store[field] = artifact
    return artifact


def get_vad_artifact(request: Any, entry: Any = None, field: str = "input_vad_data") -> VadArtifact:
    """Return the VAD/transcription artifact for ``field`` (JSON on the wire)."""
    store = artifact_store(entry)
    existing = store.get(field) if store is not None else None
    if isinstance(existing, VadArtifact):
        return existing

    payload = getattr(request, field, None)
    if payload is None:
        raise ValueError(f"{field}: missing VAD payload")
    artifact = VadArtifact.from_payload(decode_json_field(payload, field))
    if store is not None:
        store[field] = artifact
    return artifact


def get_mel_artifact(
    request: Any,
    entry: Any = None,
    field: str = "input_mel_spectrogram",
    sample_rate: int = 44100,
    hop_length: int = 441,
) -> MelArtifact:
    """Return the mel artifact for ``field``, decoding the legacy JSON form once."""
    store = artifact_store(entry)
    existing = store.get(field) if store is not None else None
    if isinstance(existing, MelArtifact):
        return existing

    payload = getattr(request, field, None)
    if not payload:
        raise ValueError(f"{field}: missing mel payload")
    array = decode_mel_field(payload, field)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        pass
    else:
        raise ValueError(f"{field}: expected a 2-D or 3-D mel array, got shape {array.shape}")
    artifact = MelArtifact.from_numpy(array, sample_rate=sample_rate, hop_length=hop_length)
    if store is not None:
        store[field] = artifact
    return artifact


def prefill_audio_artifacts(request: Any, artifacts: dict | None = None) -> dict:
    """Decode inbound audio/VAD/mel fields into artifacts exactly once (admission).

    Runs at request admission so no pipeline step ever re-reads a base64 audio
    field. Best effort for fields a family might interpret differently (the
    HarmonySpeech vocoder's ``input_audio`` is JSON, not audio): undecodable
    payloads are left for the step that owns them to reject with its typed
    error, which preserves the pre-Phase-2 error behaviour.
    """
    artifacts = artifacts if artifacts is not None else {}
    for field in AUDIO_FIELDS:
        payload = getattr(request, field, None)
        if not isinstance(payload, str) or not payload or field in artifacts:
            continue
        try:
            waveform, sample_rate = decode_audio_field(payload, field)
        except ValueError:
            continue
        artifacts[field] = AudioArtifact.from_numpy(waveform, sample_rate)

    vad_payload = getattr(request, "input_vad_data", None)
    if isinstance(vad_payload, str) and vad_payload and "input_vad_data" not in artifacts:
        with contextlib.suppress(ValueError):
            artifacts["input_vad_data"] = VadArtifact.from_payload(
                decode_json_field(vad_payload, "input_vad_data")
            )

    mel_payload = getattr(request, "input_mel_spectrogram", None)
    if isinstance(mel_payload, str) and mel_payload and "input_mel_spectrogram" not in artifacts:
        try:
            array = decode_mel_field(mel_payload, "input_mel_spectrogram")
        except ValueError:
            array = None
        if array is not None and array.ndim == 2:
            artifacts["input_mel_spectrogram"] = MelArtifact.from_numpy(
                array[None, ...], sample_rate=44100, hop_length=441
            )
    return artifacts



@dataclass
class TextToSpeechGenerationOptions:
    seed: int | None
    style: int | None
    speed: float | None
    pitch: float | None
    energy: float | None
    # Chatterbox-specific fields (None = use model default in prepare function)
    exaggeration: float | None = None
    cfg_weight: float | None = None
    temperature: float | None = None
    repetition_penalty: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    norm_loudness: bool | None = None
    # OpenVoice V1 / MeloTTS sampling parameters (None = family default).
    noise_scale: float | None = None
    noise_scale_w: float | None = None
    sdp_ratio: float | None = None
    # Long-form audio (Phase 12-1 fields, wired by the serving layer in 12-2).
    # Defaults keep every existing caller on the single-shot path.
    split_text: bool = False
    chunk_size: int | None = None
    crossfade_ms: int | None = None
    sentence_pause_ms: int | None = None
    # Phase B output filters (Phase 12-3). Default off = byte-identical output.
    remove_dc_offset: bool = False
    trim_silence: bool = False
    fix_internal_silence: bool = False


class TextToSpeechAudioOutputOptions:
    format: str | None = "wav"
    sample_rate: int | None = None
    stream: bool | None = False


class RequestInput:
    """
    The base class for model input data
    """

    def __init__(self, request_id: str, requested_model: str, model: str, metrics: RequestMetrics | None = None):
        self.request_id = request_id
        self.requested_model = requested_model
        self.model = model
        self.metrics = metrics

    # Artifact getters (Phase 2-2). They accept the owning ``EngineRequest`` so
    # artifacts cached on it take precedence over the base64 fields; callers
    # without an EngineRequest still get a one-off decode.
    def get_audio_artifact(self, engine_request=None, field: str = "input_audio", target_sr: int | None = None):
        return get_audio_artifact(self, engine_request, field, target_sr)

    def get_speaker_artifact(self, engine_request=None, field: str = "input_embedding", backend: str = "openvoice"):
        return get_speaker_artifact(self, engine_request, field, backend)

    def get_vad_artifact(self, engine_request=None, field: str = "input_vad_data"):
        return get_vad_artifact(self, engine_request, field)

    def get_mel_artifact(
        self, engine_request=None, field: str = "input_mel_spectrogram", sample_rate: int = 44100, hop_length: int = 441
    ):
        return get_mel_artifact(self, engine_request, field, sample_rate, hop_length)


class AudioConversionRequestInput(RequestInput):
    """
    The input data for an Audio Conversion Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        source_audio: str,
        input_mel_spectrogram: str | None = None,
        output_options: TextToSpeechAudioOutputOptions | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.source_audio = source_audio
        self.input_mel_spectrogram = input_mel_spectrogram
        self.output_options = output_options

    @classmethod
    def from_openai(cls, request_id: str, request: "AudioConversionRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            source_audio=getattr(request, "source_audio", ""),
            input_mel_spectrogram=getattr(request, "input_mel_spectrogram", None),
            output_options=getattr(request, "output_options", None),
        )


class VoiceConversionRequestInput(RequestInput):
    """
    The input data for a Voice Conversion Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        source_audio: str,
        target_audio: str | None,
        target_embedding: str | None,
        generation_options: TextToSpeechGenerationOptions | None,
        output_options: TextToSpeechAudioOutputOptions | None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.source_audio = source_audio
        self.target_audio = target_audio
        self.target_embedding = target_embedding
        self.generation_options = generation_options
        self.output_options = output_options

    @classmethod
    def from_openai(cls, request_id: str, request: "VoiceConversionRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            source_audio=getattr(request, "source_audio", ""),
            target_audio=getattr(request, "target_audio", None),
            target_embedding=getattr(request, "target_embedding", None),
            generation_options=getattr(request, "generation_options", None),
            output_options=getattr(request, "output_options", None),
        )


class TextToSpeechRequestInput(RequestInput):
    """
    The input data for a Text-to-Speech Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_text: str,
        mode: str,
        language_id: str | None = None,
        voice_id: str | None = None,
        input_audio: str | None = None,
        input_vad_mode: str | None = None,
        input_vad_data: str | None = None,
        input_embedding: str | None = None,
        generation_options: TextToSpeechGenerationOptions | None = None,
        output_options: TextToSpeechAudioOutputOptions | None = None,
        post_generation_filters: list[AudioConversionRequestInput] | None = None,
        metrics: RequestMetrics | None = None,
        # Internal, non-wire flag (Phase 12-2): a long-form chunk sub-request
        # asks the runner not to watermark its own slice, because the serving
        # layer watermarks the stitched buffer exactly once. Never set from the
        # wire; only the serving chunker sets it on the requests it builds.
        suppress_watermark: bool = False,
        # Internal, non-wire per-request deadline override (Phase 12-2
        # follow-up, 2026-11-09). ``None`` means "inherit the engine's
        # configured ``request_timeout_s``"; it does NOT mean "no timeout". A
        # value > 0 replaces the engine-wide timeout for this request; ``0``
        # disables the deadline exactly like an engine-wide 0. Only the serving
        # chunker sets it (a chunk may legitimately wait in the queue for its
        # siblings ahead of it, which is not a synthesis failure).
        request_timeout_s: float | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_text = input_text
        self.mode = mode
        self.language_id = language_id
        self.voice_id = voice_id
        self.input_audio = input_audio
        self.input_vad_mode = input_vad_mode
        self.input_vad_data = input_vad_data
        self.input_embedding = input_embedding
        self.generation_options = generation_options
        self.output_options = output_options
        self.post_generation_filters = post_generation_filters
        self.suppress_watermark = bool(suppress_watermark)
        self.request_timeout_s = request_timeout_s

    @classmethod
    def from_openai(cls, request_id: str, request: "TextToSpeechRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_text=getattr(request, "input", ""),
            mode=getattr(request, "mode", ""),
            language_id=getattr(request, "language", None),
            voice_id=getattr(request, "voice", None),
            input_audio=getattr(request, "input_audio", None),
            input_vad_mode=getattr(request, "input_vad_mode", None),
            input_vad_data=getattr(request, "input_vad_data", None),
            input_embedding=getattr(request, "input_embedding", None),
            generation_options=getattr(request, "generation_options", None),
            output_options=getattr(request, "output_options", None),
            post_generation_filters=getattr(request, "post_generation_filters", None),
        )


class SpeechEmbeddingRequestInput(RequestInput):
    """
    The input data for a Speech Embedding Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_audio: str | None = None,
        input_vad_mode: str | None = None,
        input_vad_data: str | None = None,
        generation_options: TextToSpeechGenerationOptions | None = None,
        metrics: RequestMetrics | None = None,
        # Internal, non-wire per-request deadline override (Phase 12-2
        # follow-up): the one-time reference-embedding sub-request of a
        # long-form clone request queues behind other admitted work too, so the
        # serving layer stamps the same extended deadline on it. ``None`` means
        # "inherit the engine's ``request_timeout_s``" (not "no timeout").
        request_timeout_s: float | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_audio = input_audio
        self.input_vad_mode = input_vad_mode
        self.input_vad_data = input_vad_data
        # Optional carrier for the originating TTS request's options (Phase
        # 12-2): the serving layer's one-time reference-embedding sub-request
        # passes them so family-specific embedding parameters (e.g. Chatterbox
        # ``exaggeration``) stay identical to the non-chunked pipeline. Not a
        # wire field; external embed requests leave it unset.
        self.generation_options = generation_options
        self.request_timeout_s = request_timeout_s

    @classmethod
    def from_openai(cls, request_id: str, request: "EmbedSpeakerRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_audio=getattr(request, "input_audio", None),
            input_vad_mode=getattr(request, "input_vad_mode", None),
            input_vad_data=getattr(request, "input_vad_data", None),
        )


class SynthesisRequestInput(RequestInput):
    """
    The input data for a Speech Synthesis Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_text: str = "",
        language_id: str | None = None,
        voice_id: str | None = None,
        input_embedding: str = "",
        generation_options: TextToSpeechGenerationOptions | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_text = input_text
        self.language_id = language_id
        self.voice_id = voice_id
        self.input_embedding = input_embedding
        self.generation_options = generation_options

    @classmethod
    def from_openai(cls, request_id: str, request: "SynthesizeAudioRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_text=getattr(request, "input", ""),
            language_id=getattr(request, "language", None),
            voice_id=getattr(request, "voice", None),
            input_embedding=getattr(request, "input_embedding", ""),
            generation_options=getattr(request, "generation_options", None),
        )


class VocodeRequestInput(RequestInput):
    """
    The input data for a Vocoding Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_audio: str | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_audio = input_audio

    @classmethod
    def from_openai(cls, request_id: str, request: "VocodeAudioRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_audio=getattr(request, "input_audio", None),
        )


class SpeechTranscribeRequestInput(RequestInput):
    """
    The input data for a Speech Transcribe Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_audio: str | None = None,
        get_language: bool | None = False,
        get_timestamps: bool | None = False,
        task: str = "transcribe",
        language: str | None = None,
        prompt: str | None = None,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_audio = input_audio
        self.get_language = get_language
        self.get_timestamps = get_timestamps
        # ``task`` is faster-whisper's transcription task ("transcribe" or
        # "translate"); the handler turns it into the matching prompt token
        # (<|transcribe|> / <|translate|>) in both execution paths (Phase 10-4).
        self.task = task or "transcribe"
        # Source-language hint (never a translation target) and optional
        # initial prompt, both forwarded to faster-whisper when set.
        self.language = language
        self.prompt = prompt

    @classmethod
    def from_openai(cls, request_id: str, request: "SpeechTranscribeRequest", task: str = "transcribe"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_audio=getattr(request, "input_audio", None),
            get_language=getattr(request, "get_language", False),
            get_timestamps=getattr(request, "get_timestamps", False),
            task=task,
            language=getattr(request, "language", None),
            prompt=getattr(request, "prompt", None),
        )


class DetectVoiceActivityRequestInput(RequestInput):
    """
    The input data for a VAD Request
    """

    def __init__(
        self,
        request_id: str,
        requested_model: str,
        model: str,
        input_audio: str | None = None,
        get_timestamps: bool | None = False,
        threshold: float | None = 0.5,
        min_speech_duration_ms: int | None = 250,
        min_silence_duration_ms: int | None = 100,
        speech_pad_ms: int | None = 30,
        return_seconds: bool | None = False,
        metrics: RequestMetrics | None = None,
    ):
        super().__init__(request_id=request_id, requested_model=requested_model, model=model, metrics=metrics)
        self.input_audio = input_audio
        self.get_timestamps = get_timestamps
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.return_seconds = return_seconds

    @classmethod
    def from_openai(cls, request_id: str, request: "DetectVoiceActivityRequest"):
        return cls(
            request_id=request_id,
            requested_model=getattr(request, "model", ""),
            model=getattr(request, "model", ""),
            input_audio=getattr(request, "input_audio", None),
            get_timestamps=getattr(request, "get_timestamps", False),
            threshold=getattr(request, "threshold", 0.5),
            min_speech_duration_ms=getattr(request, "min_speech_duration_ms", 250),
            min_silence_duration_ms=getattr(request, "min_silence_duration_ms", 100),
            speech_pad_ms=getattr(request, "speech_pad_ms", 30),
            return_seconds=getattr(request, "return_seconds", False),
        )
