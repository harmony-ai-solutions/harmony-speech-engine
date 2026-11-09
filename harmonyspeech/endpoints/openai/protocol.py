# Protocol Definition for OpenAI-Like Endpoint
# Inspiration taken from PygmalionAI / Aphrodite Engine & OpenAI API definitions

import time
from typing import Literal, Self

from pydantic import BaseModel, Field

from harmonyspeech.common.utils import random_uuid


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: str | None = None
    code: int


class VoiceOptions(BaseModel):
    voice: str = "default"
    styles: list[str] | None = Field(default_factory=list)


class LanguageOptions(BaseModel):
    language: str = "default"
    voices: list[VoiceOptions] | None = Field(default_factory=list)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "harmony-ai-solutions"
    languages: list[LanguageOptions] | None = Field(default_factory=list)
    root: str | None = None
    parent: str | None = None
    # Additive metadata generated from the pipeline registry (Phase 3-3):
    # capabilities the pipeline serves ("tts", "embed", "vc", "stt", "vad",
    # "restore") and, for toolchain cards, the configured member executor names.
    capabilities: list[str] | None = None
    models: list[str] | None = None
    # permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class VoiceDescriptor(BaseModel):
    """One selectable voice in the voices catalog (Phase 13-3).

    ``source`` is the honesty label: ``"native"`` is a preset owned by the
    configured model (its ``voices:`` keys), ``"standard"`` is a vendored
    standard voice pack identity. A standard entry carries no ``languages``
    (pack clips are language-agnostic references); native entries carry the
    configured language, or ``"default"`` for models that expose voices
    without a language.
    """

    name: str
    source: Literal["native", "standard"]
    languages: list[str] | None = None


class ModelVoiceCatalogEntry(BaseModel):
    """The voices listing for one model or toolchain id (Phase 13-3).

    ``object`` mirrors ``/v1/audio/speech/models``: ``"model"`` for a
    configured executor, ``"toolchain"`` for a pipeline alias that is not also
    a configured model name. ``model_type`` is null for a toolchain entry.

    ``cloning_capable`` means a standard voice pack name is meaningful for this
    id in some direction (TTS ``voice`` or VC ``target_voice``). The capability
    flags describe the model; ``voices`` only contains ``standard`` entries
    while the pack is enabled on the server.

    ``vc_target_kind`` keeps the two VC capability kinds distinct:
    ``"target_audio"`` (the executor consumes the clip itself) or
    ``"target_embedding"`` (the serving layer embeds the clip first); null when
    the id is not a voice-conversion target.
    """

    model: str
    object: Literal["model", "toolchain"] = "model"
    model_type: str | None = None
    cloning_capable: bool = False
    vc_target_kind: Literal["target_audio", "target_embedding"] | None = None
    voices: list[VoiceDescriptor] = Field(default_factory=list)


class VoiceCatalogList(BaseModel):
    """``GET /v1/audio/voices`` response (``ModelList``-consistent)."""

    object: str = "list"
    data: list[ModelVoiceCatalogEntry] = Field(default_factory=list)


class ResponseFormat(BaseModel):
    # type must be "json_object" or "text"
    type: Literal["text", "json_object"]


class GenerationOptions(BaseModel):
    seed: int | None = None
    style: int | None = None
    speed: float | None = None
    pitch: float | None = None
    energy: float | None = None
    # Chatterbox-specific fields
    exaggeration: float | None = Field(
        None,
        description="(Chatterbox TTS/Multilingual) Emotion exaggeration factor. Higher values amplify emotional expressiveness. Default: 0.5",
    )
    cfg_weight: float | None = Field(
        None,
        description="(Chatterbox TTS/Multilingual) Classifier-free guidance weight. Controls how closely the model follows the voice prompt. Default: 0.5",
    )
    temperature: float | None = Field(
        None,
        description="(Chatterbox TTS/Turbo/Multilingual) Sampling temperature. Higher values increase randomness. Default: 0.8",
    )
    repetition_penalty: float | None = Field(
        None,
        description="(Chatterbox TTS/Turbo/Multilingual) Repetition penalty applied during sampling. Values > 1.0 reduce repetition. Default: 1.2",
    )
    top_p: float | None = Field(
        None,
        description="(Chatterbox TTS/Turbo/Multilingual) Top-p (nucleus) sampling probability. Default TTS: 1.0, Turbo: 0.95",
    )
    min_p: float | None = Field(
        None,
        description="(Chatterbox TTS/Multilingual only) Minimum probability threshold for sampling. Default: 0.05. Not supported by Turbo variant.",
    )
    top_k: int | None = Field(
        None,
        description="(Chatterbox Turbo only) Top-k sampling - number of candidates kept. Default: 1000. Not supported by standard TTS or Multilingual variants.",
    )
    norm_loudness: bool | None = Field(
        None,
        description="(Chatterbox Turbo only) Apply pyloudnorm loudness normalization to output. Default: True. Not supported by standard TTS or Multilingual variants.",
    )
    # OpenVoice V1 / MeloTTS sampling parameters. Exposed because both models
    # normally sample; `noise_scale: 0` (with `noise_scale_w: 0`) makes synthesis
    # reproducible, which the batching equivalence gate relies on.
    noise_scale: float | None = Field(
        None,
        description="(OpenVoice V1 / MeloTTS) Waveform sampling noise. Defaults: OpenVoice V1 0.667, MeloTTS 0.6. 0 disables sampling noise.",
    )
    noise_scale_w: float | None = Field(
        None,
        description="(OpenVoice V1 / MeloTTS) Duration-predictor noise. Defaults: OpenVoice V1 0.6, MeloTTS 0.8. 0 disables it.",
    )
    sdp_ratio: float | None = Field(
        None,
        description="(OpenVoice V1 / MeloTTS) Weight of the stochastic duration predictor vs the deterministic one. Default: 0.2.",
    )
    # Long-form audio (Phase 12-2): sentence-aware splitting, concurrent chunk
    # synthesis and ordered stitching. Defaults keep the single-shot path.
    split_text: bool = Field(
        False,
        description=(
            "Split long input text into sentence-aware chunks before synthesis and stitch the "
            "result. A request that yields 0 or 1 chunk takes the single-shot path unchanged."
        ),
    )
    chunk_size: int | None = Field(
        None,
        description=(
            "Character budget per synthesis chunk when `split_text` is enabled (also enables "
            "chunking on its own). When `split_text` is true and this is null, the server "
            "budget is 300 characters, except for Chatterbox Turbo (150) and Chatterbox "
            "Multilingual (75), which use tighter budgets so their long-form output stays "
            "faithful to the text (measured 2026-11-09). An explicit value always wins."
        ),
    )
    crossfade_ms: int | None = Field(
        None,
        description=(
            "Equal-power crossfade length between stitched chunks in milliseconds (default 0). "
            "The fade consumes part of `sentence_pause_ms` rather than adding to it."
        ),
    )
    sentence_pause_ms: int | None = Field(
        None,
        description=(
            "Silence inserted between stitched sentence chunks in milliseconds (default 0). "
            "The audible gap is `sentence_pause_ms - crossfade_ms` (clamped at zero)."
        ),
    )
    # Phase B output filters (Phase 12-3). All default off: with every flag
    # false the serving layer does not touch the samples, so output is
    # byte-identical to the pre-12 path.
    remove_dc_offset: bool = Field(
        False,
        description=(
            "Remove the constant (DC) offset from the generated audio. In the chunked path the "
            "mean is removed per chunk before stitching. Default: false."
        ),
    )
    trim_silence: bool = Field(
        False,
        description=(
            "Trim leading and trailing silence (energy-gated at 40 dB below the loudest frame, "
            "keeping 50 ms of padding). Applied to the final stitched buffer. Default: false."
        ),
    )
    fix_internal_silence: bool = Field(
        False,
        description=(
            "Collapse internal pauses longer than 700 ms to at most 300 ms. Leading/trailing "
            "silence is not touched (see `trim_silence`). Applied to the final stitched buffer. "
            "Default: false."
        ),
    )


class AudioOutputOptions(BaseModel):
    format: str | None = Field(
        default="wav",
        description=(
            "Output audio container/codec: wav (default), flac, ogg, mp3, opus, pcm (raw s16le) "
            "or aac (only when ffmpeg is available on the server). Unsupported values are rejected with HTTP 400."
        ),
    )
    sample_rate: int | None = Field(
        default=None,
        description=(
            "Resample the output to this rate in Hz. Supported: 8000, 16000, 22050, 24000, 32000, 44100, 48000. "
            "When omitted the model's native rate is kept."
        ),
    )
    stream: bool | None = False


class BaseRequest(BaseModel):
    model: str = Field(default="", description="the name of the model")


class AudioConversionRequest(BaseRequest):
    """
    AudioConversionRequest
    Used to convert the style or content of an audio in a specific way.
    Depending on model selection, the caller may need to provide additional params.
    """

    source_audio: str = Field(default="", description="Binary audio data to be processed, encoded in base64")
    output_options: AudioOutputOptions | None = Field(
        default=None, description="Options for returning the generated audio, see documentation for possible values"
    )
    pre_processing_filters: list[Self] | None = Field(
        default_factory=list, description="List of Preprocessing filters to apply to the generated audio."
    )


class VoiceConversionRequest(AudioConversionRequest):
    """
    VoiceConversionRequest
    Used to convert the tone or nature of a voice in a specific way.
    Depending on model selection, the caller may need to provide additional params.
    """

    target_audio: str | None = Field(
        default=None,
        description="Binary audio data of the reference speaker for converting the source, encoded in base64",
    )
    target_voice: str | None = Field(
        default=None,
        description=(
            "Name of a standard voice pack speaker (for example 'Abigail') to use as the "
            "conversion target. Resolves to the shipped reference clip; cannot be combined "
            "with target_audio or target_embedding. On target-audio models (for example "
            "'chatterbox_vc') the clip is the target audio; on OpenVoice tone converters the "
            "engine embeds the clip once through the pipeline encoder and uses the result as "
            "the target embedding."
        ),
    )
    target_embedding: str | None = Field(
        default=None,
        description="Binary embedding of the reference speaker for converting the source, encoded in base64. Faster than providing a target audio file",
    )
    generation_options: GenerationOptions | None = Field(
        default=None, description="Options for generating a speech request, see documentation for possible values"
    )


class TextToSpeechRequest(BaseRequest):
    """
    TextToSpeechRequest
    Used to trigger a speech generation for the provided text input using the specified model.
    Depending on model selection, the caller may need to provide additional params.
    Based on OpenAI TTS API; extended for Harmony Speech Engine features.
    """

    input: str = Field(default="", description="the text to synthesize")
    mode: str = Field(default="", description="the operation mode for the model to process the request")
    language: str | None = Field(
        default=None,
        description="language to synthesize. Only if the model supports language IDs. Please refer to the model's documentation for which values are availiable.",
    )
    voice: str | None = Field(
        default=None,
        description="ID of the voice to synthesize. Only if the model supports voice IDs. Please refer to the model's documentation for which values are availiable.",
    )
    input_audio: str | None = Field(
        default=None,
        description="Binary audio data of the reference speaker for synthesizing the text, encoded in base64",
    )
    input_vad_mode: str | None = Field(
        default=None,
        description="VAD mode to use in case the provided TTS or VC framework requires VAD information for processing",
    )
    input_vad_data: str | None = Field(
        default=None, description="result data from VAD step to be processed in follow-up TTS or VC steps"
    )
    input_embedding: str | None = Field(
        default=None,
        description="Binary embedding of the reference speaker for synthesizing the text, encoded in base64. Faster than providing a target audio file",
    )
    generation_options: GenerationOptions | None = Field(
        default=None, description="Options for generating a speech request, see documentation for possible values"
    )
    output_options: AudioOutputOptions | None = Field(
        default=None, description="Options for returning the generated audio, see documentation for possible values"
    )
    pre_processing_filters: list[AudioConversionRequest] | None = Field(
        default_factory=list, description="List of Preprocessing filters to apply to the generated audio."
    )
    post_generation_filters: list[AudioConversionRequest] | None = Field(
        default_factory=list, description="List of Post-Generation filters to apply to the generated audio."
    )


class BaseResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"{random_uuid()}")
    model: str = Field(default="", description="the name of the model")
    created: int = Field(default_factory=lambda: int(time.time()))


class AudioDataResponse(BaseResponse):
    """
    AudioDataResponse
    Result Audio Data
    """

    data: str = Field(default="", description="Audio Bytes in requested format of the initial request")
    sample_rate: int | None = Field(
        default=None,
        description=(
            "Effective sample rate of `data` in Hz. Set when the engine re-encoded or resampled the output "
            "(for example `pcm` output, or any explicit sample_rate); omitted for unchanged WAV passthrough."
        ),
    )


class TextToSpeechResponse(AudioDataResponse):
    """
    TextToSpeechResponse
    Extends AudioDataResponse with a request specific ID
    """

    id: str = Field(default_factory=lambda: f"tts-{random_uuid()}")


class VoiceConversionResponse(AudioDataResponse):
    """
    VoiceConversionResponse
    Extends AudioDataResponse with a request specific ID
    """

    id: str = Field(default_factory=lambda: f"vc-{random_uuid()}")


class AudioConversionResponse(AudioDataResponse):
    """
    AudioConversionResponse
    Extends AudioDataResponse with a request specific ID
    """

    id: str = Field(default_factory=lambda: f"ac-{random_uuid()}")


class SpeechTranscribeRequest(BaseRequest):
    """
    SpeechTranscribeRequest
    Used to apply ASR on a provided audio file
    Depending on model selection, the caller may need to provide additional params.
    Based on OpenAI STT API; extended for Harmony Speech Engine features.
    """

    input_audio: str = Field(
        default="",
        description="Binary audio data of the reference speaker for synthesizing the text, encoded in base64",
    )
    language: str | None = Field(
        default=None,
        description=(
            "Source-language hint (ISO 639-1, e.g. 'en', 'de'). When omitted the model detects it. "
            "For translations this is only a hint about the spoken language — the output is always translated "
            "to English; it is never a target language."
        ),
    )
    prompt: str | None = Field(
        default=None,
        description="Optional text used as an initial prompt to guide the transcription/translation (vocabulary, style).",
    )
    response_format: str = Field(
        default="json",
        description=(
            "Response format: 'json' (default, the SpeechToTextResponse object), 'text' (plain transcript), "
            "'srt' or 'vtt' (subtitle documents rendered from the segment timestamps). "
            "The subtitle formats are served as text/plain. Unknown values are rejected with HTTP 400."
        ),
    )
    get_language: bool | None = Field(
        default=False, description="whether to return the source language tag. Check model description if supported."
    )
    get_timestamps: bool | None = Field(
        default=False, description="whether to return the word timestamps. Check model description if supported."
    )


class SpeechTranslationRequest(SpeechTranscribeRequest):
    """
    SpeechTranslationRequest
    Used to translate speech in a provided audio file into English text.
    Mirrors :class:`SpeechTranscribeRequest`; the serving layer runs the same
    pipeline with ``task="translate"``. ``language`` is a source-language hint
    only (the output language is always English, matching the OpenAI API).
    """


class SpeechToTextResponse(BaseResponse):
    """
    SpeechToTextResponse
    Result text determined from audio and optional language tag
    """

    id: str = Field(default_factory=lambda: f"stt-{random_uuid()}")
    text: str = Field(default="", description="text retrieved from the input audio")
    language: str | None = Field(default=None, description="tag of the detected language")
    timestamps: list[dict] | None = Field(default_factory=list, description="list of the detected timestamps")


class DetectVoiceActivityRequest(BaseRequest):
    """
    DetectVoiceActivityRequest
    Used to check for human speech in a provided audio file
    Depending on model selection, the caller may need to provide additional params.
    """

    input_audio: str = Field(
        default="",
        description="Binary audio data of the reference speaker for synthesizing the text, encoded in base64",
    )
    get_timestamps: bool | None = Field(
        default=False, description="whether to return the word timestamps. Check model description if supported."
    )
    # Dynamic Silero VAD parameters
    threshold: float | None = Field(
        default=0.5, description="Speech threshold. Probabilities above this value are considered as speech."
    )
    min_speech_duration_ms: int | None = Field(
        default=250, description="Final speech chunks shorter than this duration (in ms) are discarded."
    )
    min_silence_duration_ms: int | None = Field(
        default=100, description="Minimum silence duration (in ms) before separating speech chunks."
    )
    speech_pad_ms: int | None = Field(
        default=30, description="Speech chunks are padded by this duration (in ms) on each side."
    )
    return_seconds: bool | None = Field(
        default=False, description="Whether to return timestamps in seconds (default: samples)."
    )


class DetectVoiceActivityResponse(BaseResponse):
    """
    DetectVoiceActivityResponse
    Result text determined from audio and optional language tag
    """

    id: str = Field(default_factory=lambda: f"stt-{random_uuid()}")
    speech_activity: bool = Field(default=False, description="whether speech activity was detected")
    timestamps: list[dict] | None = Field(default_factory=list, description="list of the detected timestamps")


class EmbedSpeakerRequest(BaseRequest):
    """
    EmbedSpeakerRequest
    Used to create a Speaker Embedding form a provided audio, which can later be re-used for Text-to-Speech or
    Voice Conversion functionality.
    Please refer to the documentation whether an embedding is compatible between different models.
    """

    input_audio: str = Field(default="", description="Binary audio data to be processed, encoded in base64")


class EmbedSpeakerResponse(BaseResponse):
    """
    EmbedSpeakerResponse
    Result Speaker Embedding
    """

    id: str = Field(default_factory=lambda: f"embed-{random_uuid()}")
    data: str = Field(
        default="",
        description="Speaker embedding data for the audio provided in the initial request, encoded in base64",
    )


class SynthesizeAudioRequest(BaseRequest):
    """
    SynthesizeAudioRequest
    Used to run a synthesizer model for a provided input audio
    """

    input: str = Field(default="", description="the text to synthesize")
    language: str = Field(
        default="",
        description="language to synthesize. Only if the model supports language IDs. Please refer to the model's documentation for which values are availiable.",
    )
    voice: str | None = Field(
        default=None,
        description="ID of the voice to synthesize. Only if the model supports voice IDs. Please refer to the model's documentation for which values are availiable.",
    )
    input_embedding: str | None = Field(
        default=None,
        description="Binary embedding of the reference speaker for synthesizing the text, encoded in base64. Faster than providing a target audio file",
    )
    generation_options: GenerationOptions | None = Field(
        default=None, description="Options for generating a speech request, see documentation for possible values"
    )


class SynthesizeAudioResponse(AudioDataResponse):
    """
    SynthesizeAudioResponse
    Result Audio file after synthesis step
    """

    id: str = Field(default_factory=lambda: f"synthesis-{random_uuid()}")


class VocodeAudioRequest(BaseRequest):
    """
    VocodeAudioRequest
    Used to run a vocoder model over a provided input audio
    """

    input_audio: str = Field(default="", description="Binary audio data to be processed, encoded in base64")


class VocodeAudioResponse(AudioDataResponse):
    """
    VocodeAudioResponse
    Result Audio file after vocoding step
    """

    id: str = Field(default_factory=lambda: f"vocode-{random_uuid()}")
