"""Long-form TTS chunking + stitching through the real engine loop (Phase 12-2).

Every test here drives the *real* ``OpenAIServingTextToSpeech`` and the real
``AsyncHarmonySpeech`` background loop, scheduler and runner dispatch. Model
weights are replaced by tiny deterministic stubs (the Phase 8-x precedent), so
the suite is hermetic and CPU-only.

What is pinned:

* concurrent chunk submission + ordered stitch (chunk-index order, never
  completion order);
* clone mode embeds exactly once for an N-chunk request;
* the watermark runs exactly once per HTTP request, on the stitched buffer;
* a failing chunk fails the whole request (no partial audio with a 200);
* per-chunk seed derivation is deterministic and reaches the model;
* native-speed families are not double-stretched; non-native families are;
* the 0/1-chunk fast path is byte-identical to the un-chunked path;
* every family (Chatterbox en/turbo/multilingual, hs1, Kitten x4, Melo,
  OpenVoice V1/V2) produces a valid WAV whose duration grows with the chunk
  count when a sentence pause is requested.
"""

import asyncio
import base64
import dataclasses
import threading
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    DetectVoiceActivityRequestInput,
    SpeechEmbeddingRequestInput,
    SpeechTranscribeRequestInput,
    SynthesisRequestInput,
    TextToSpeechRequestInput,
    VocodeRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.endpoints.openai.protocol import (
    ErrorResponse,
    GenerationOptions,
    TextToSpeechRequest,
    TextToSpeechResponse,
    VoiceConversionRequest,
    VoiceConversionResponse,
)
from harmonyspeech.endpoints.openai.serving_text_to_speech import (
    NATIVE_SPEED_MODEL_TYPES,
    STANDALONE_EMBED_MODEL_TYPES,
    OpenAIServingTextToSpeech,
)
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

SAMPLE_RATE = 8000
#: Samples produced per input character by the stub synthesizers. Chunk texts
#: are long enough that a chunk always dwarfs the crossfade window.
SAMPLES_PER_CHAR = 8
PIPELINE_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Stub watermarker / models
# ---------------------------------------------------------------------------


class CountingWatermarker:
    def __init__(self):
        self.calls = 0

    def apply_watermark(self, wav, sample_rate=None):  # noqa: ARG002 - upstream signature
        self.calls += 1
        return np.asarray(wav, dtype=np.float32)


class ChatterboxStubModel(torch.nn.Module):
    """Upstream-shaped Chatterbox stand-in (watermarks inside ``generate``)."""

    def __init__(self, *, model_type="ChatterboxTTS", fail_on=None):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.model_type = model_type
        self.sr = SAMPLE_RATE
        self.watermarker = CountingWatermarker()
        self.conds = types.SimpleNamespace(name="baseline")
        self.prepare_calls = 0
        self.generate_calls: list[str] = []
        self.fail_on = fail_on

    def prepare_conditionals(self, audio_buf, **kwargs):  # noqa: ARG002 - upstream signature
        self.prepare_calls += 1
        self.conds = types.SimpleNamespace(name="conditionals")

    def generate(self, text, **kwargs):  # noqa: ARG002 - family signature superset
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError(f"synthetic failure for {text!r}")
        self.generate_calls.append(text)
        samples = max(1, len(text)) * SAMPLES_PER_CHAR
        # Distinct value per chunk so the stitch-order test can see which chunk
        # landed where (PCM-16 quantization is ~3e-5).
        value = (ord(text[0]) % 50 + 1) / 100.0
        wav = np.full((samples,), value, dtype=np.float32)
        wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)


class KittenStubModel(torch.nn.Module):
    """KittenTTS stand-in; honours ``speed`` natively (length / speed)."""

    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.sample_rate = SAMPLE_RATE
        self.generate_calls: list[str] = []

    def generate(self, text, voice="Jasper", speed=1.0, clean_text=True):  # noqa: ARG002
        self.generate_calls.append(text)
        samples = int(max(1, len(text)) * SAMPLES_PER_CHAR / max(float(speed), 1e-6))
        return np.full((samples,), 0.1, dtype=np.float32)


class KittenTextStubModel(KittenStubModel):
    """KittenTTS stand-in whose audio *encodes the chunk text*.

    KittenTTS runs through parallel lanes (Phase 8-5), so a chunk→result
    crossing would swap whole blocks of the stitched buffer while keeping the
    total duration plausible. A constant-valued stub cannot see that; this one
    gives each chunk a distinct sample value (derived from its first character)
    and records the thread that produced it, so the pairing and the lane width
    are both observable.
    """

    def __init__(self):
        super().__init__()
        self.thread_ids: set[int] = set()

    def generate(self, text, voice="Jasper", speed=1.0, clean_text=True):  # noqa: ARG002
        self.generate_calls.append(text)
        self.thread_ids.add(threading.get_ident())
        samples = max(1, len(text)) * SAMPLES_PER_CHAR
        value = (ord(text[0]) % 50 + 1) / 100.0
        return np.full((samples,), value, dtype=np.float32)


class MeloStubModel(torch.nn.Module):
    """MeloTTS stand-in: native speed via the phones length the handler passes."""

    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def infer(self, x_tst, x_tst_lengths, speakers=None, tones=None, lang_ids=None, bert=None, ja_bert=None, **kwargs):
        length = int(x_tst_lengths[0])
        speed = 1.0 / float(kwargs.get("length_scale", 1.0))
        samples = int(length * SAMPLES_PER_CHAR / max(speed, 1e-6))
        return (torch.full((1, 1, samples), 0.1),)


class OpenVoiceStubModel(torch.nn.Module):
    """OpenVoice V1 synthesizer stand-in (native speed via ``length_scale``)."""

    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def infer(self, x_tst, x_tst_lengths, sid=None, **kwargs):
        length = int(x_tst_lengths[0])
        speed = 1.0 / float(kwargs.get("length_scale", 1.0))
        samples = int(length * SAMPLES_PER_CHAR / max(speed, 1e-6))
        return (torch.full((1, 1, samples), 0.1),)


class HarmonySpeechSynthStubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def generate(self, x=None, spk_emb=None, alpha=1.0, pitch_function=None, energy_function=None, **kwargs):
        frames = max(1, int(x.numel()))
        return None, torch.full((1, 80, frames), 0.1), None, None, None


class HarmonySpeechVocoderStubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def inference(self, c=None, **kwargs):
        frames = int(c.shape[0])
        # Match the real ``inference`` contract: (samples, channels).
        return torch.full((frames * 4, 1), 0.1)


class HarmonySpeechEncoderStubModel(torch.nn.Module):
    """HarmonySpeech encoder stand-in: counts real embedding forwards."""

    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.forward_calls = 0

    def forward(self, utterances=None, **kwargs):  # noqa: ARG002 - upstream signature
        self.forward_calls += 1
        return torch.ones(int(utterances.shape[0]), 16)

    def forward_batched(self, x, lengths):  # noqa: ARG002 - padded-batch path
        self.forward_calls += 1
        return torch.ones(int(x.shape[0]), 16)


class _RefEncStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        #: Counts real embedding computations (a 7-2 cache hit skips the forward).
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return torch.ones(x.shape[0], 4, 1)

    def forward_batched(self, x, lengths):  # noqa: ARG002 - padded-batch path
        self.forward_calls += 1
        return torch.ones(x.shape[0], 4, 1)


class OpenVoiceEncoderStubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ref_enc = _RefEncStub()


class ToneConverterStubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def voice_conversion(self, y, y_lengths, sid_src=None, sid_tgt=None, tau=0.3):  # noqa: ARG002
        frames = int(y_lengths[0])
        return (torch.full((1, 1, frames * 4), 0.1),)


# ---------------------------------------------------------------------------
# Engine harness
# ---------------------------------------------------------------------------


class StubExecutor:
    def __init__(self, runner, delay_s=0.0):
        self.runner = runner
        self.model_config = getattr(runner, "model_config", None)
        self.batch_sizes: list[int] = []
        #: Deterministic per-request synthesis time (seconds). Non-zero models
        #: a slow "loop" executor under load for the deadline regression test.
        self.delay_s = float(delay_s)

    def execute_model(self, requests_to_batch):
        self.batch_sizes.append(len(requests_to_batch))
        return self.runner.execute_model(requests_to_batch)

    async def execute_model_async(self, requests_to_batch):
        self.batch_sizes.append(len(requests_to_batch))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self.runner.execute_model(requests_to_batch)

    def check_health(self):
        return {"status": "ok"}

    def release_request(self, request_id):  # noqa: ARG002
        return None


def _model_config(name, model_type, **extra):
    return ModelConfig(
        name=name,
        model=f"stub/{name}",
        model_type=model_type,
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        batched_inference=False,
        **extra,
    )


def _runner(config, model):
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner._baseline_state = {
        "conds": getattr(model, "conds", None),
        "ref_dict": getattr(model, "ref_dict", None),
    }
    runner._ar_runner = None
    runner.embedding_cache = None
    runner.model_identity = None
    runner.model_config = config
    return runner


def _fake_hf_config(model_type):
    if model_type in ("MeloTTSSynthesizer", "OpenVoiceV1Synthesizer"):
        return types.SimpleNamespace(
            data=types.SimpleNamespace(sampling_rate=SAMPLE_RATE, hop_length=8),
            speakers={"default": 0},
        )
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            sampling_rate=SAMPLE_RATE,
            hop_length=8,
            filter_length=16,
            win_length=16,
        ),
    )


# ---------------------------------------------------------------------------
# Type-strict input preparation
#
# The Phase 12-2 stubs used to accept *any* request type for every model, which
# is why the 2026-11-09 OpenVoice V2 chunked-clone failure escaped: the serving
# dispatcher sent a ``SpeechEmbeddingRequestInput`` to a MeloTTS executor and the
# real ``prepare_inputs`` rejected it, while the stub happily produced inputs.
# ``_ACCEPTED_REQUEST_TYPES`` mirrors the acceptance sets in
# ``harmonyspeech/task_handler/inputs.py`` (the source of truth) so the stub
# fails exactly where the real pipeline fails.
# ---------------------------------------------------------------------------

_ACCEPTED_REQUEST_TYPES: dict[str, tuple[type, ...]] = {
    "HarmonySpeechEncoder": (TextToSpeechRequestInput, SpeechEmbeddingRequestInput),
    "HarmonySpeechSynthesizer": (TextToSpeechRequestInput, SynthesisRequestInput),
    "HarmonySpeechVocoder": (TextToSpeechRequestInput, VocodeRequestInput),
    "OpenVoiceV1ToneConverter": (TextToSpeechRequestInput, VoiceConversionRequestInput),
    "OpenVoiceV2ToneConverter": (TextToSpeechRequestInput, VoiceConversionRequestInput),
    "OpenVoiceV1ToneConverterEncoder": (
        SpeechEmbeddingRequestInput,
        TextToSpeechRequestInput,
        VoiceConversionRequestInput,
    ),
    "OpenVoiceV2ToneConverterEncoder": (
        SpeechEmbeddingRequestInput,
        TextToSpeechRequestInput,
        VoiceConversionRequestInput,
    ),
    "FasterWhisper": (
        SpeechEmbeddingRequestInput,
        TextToSpeechRequestInput,
        SpeechTranscribeRequestInput,
        DetectVoiceActivityRequestInput,
    ),
    "OpenVoiceV1Synthesizer": (TextToSpeechRequestInput, SynthesisRequestInput),
    "MeloTTSSynthesizer": (TextToSpeechRequestInput, SynthesisRequestInput),
    "KittenTTSSynthesizer": (TextToSpeechRequestInput, SynthesisRequestInput),
    "ChatterboxTTS": (
        SpeechEmbeddingRequestInput,
        TextToSpeechRequestInput,
        SynthesisRequestInput,
    ),
    "ChatterboxTurboTTS": (
        SpeechEmbeddingRequestInput,
        TextToSpeechRequestInput,
        SynthesisRequestInput,
    ),
    "ChatterboxMultilingualTTS": (
        SpeechEmbeddingRequestInput,
        TextToSpeechRequestInput,
        SynthesisRequestInput,
    ),
    "ChatterboxEmbedding": (SpeechEmbeddingRequestInput, TextToSpeechRequestInput),
    "ChatterboxVC": (VoiceConversionRequestInput,),
}


def _assert_request_types_accepted(model_config, requests_to_batch):
    """Raise like ``task_handler.inputs.prepare_inputs`` on a wrong request type."""
    accepted = _ACCEPTED_REQUEST_TYPES[model_config.model_type]
    for entry in requests_to_batch:
        if not isinstance(entry.request_data, accepted):
            expected = " or ".join(cls.__name__ for cls in accepted)
            raise ValueError(
                f"request ID {entry.request_id} is not of type {expected} "
                f"(got {type(entry.request_data).__name__})"
            )


@dataclasses.dataclass
class _FakeWhisperSegment:
    """``faster_whisper.Segment`` stand-in (``asdict``-able, like the real one)."""

    start: float = 0.0
    end: float = 1.0
    text: str = "hello"
    words: list = dataclasses.field(default_factory=list)
    id: int = 0
    seek: int = 0
    tokens: list = dataclasses.field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0


@dataclasses.dataclass
class _FakeWhisperInfo:
    """``faster_whisper.TranscriptionInfo`` stand-in."""

    language: str = "en"
    language_probability: float = 1.0
    duration: float = 1.0
    duration_after_vad: float = 1.0
    all_language_probs: list | None = None
    transcription_options: object | None = None
    vad_options: object | None = None


class _FakeWhisperPipeline:
    """``BatchedInferencePipeline`` stand-in for the OpenVoice VAD step.

    The VAD step is part of the real OpenVoice clone flow (``vad → embed``), so
    the chunked regression must run it instead of pre-supplying
    ``input_vad_data``: it is what routes the one-time embedding sub-request.
    """

    def __init__(self, model=None):
        self.model = model
        self.last_speech_timestamp = 0.0
        self.calls = 0

    def transcribe(self, audio, batch_size=8, task="transcribe", language=None, initial_prompt=None):  # noqa: ARG002
        self.calls += 1
        return [_FakeWhisperSegment()], _FakeWhisperInfo()


def _prepare_inputs(model_config, requests_to_batch):
    """Per-family stand-in for ``prepare_inputs`` (no weights, no text frontend).

    Type-strict: a request type the real pipeline would reject is rejected here
    too, with the same message shape (see ``_ACCEPTED_REQUEST_TYPES``).
    """
    _assert_request_types_accepted(model_config, requests_to_batch)
    model_type = model_config.model_type
    inputs = []
    for entry in requests_to_batch:
        request = entry.request_data
        text = getattr(request, "input_text", "") or ""
        options = getattr(request, "generation_options", None)
        speed = float(getattr(options, "speed", None) or 1.0)
        if model_type in ("ChatterboxTTS", "ChatterboxTurboTTS", "ChatterboxMultilingualTTS"):
            if isinstance(request, SpeechEmbeddingRequestInput):
                inputs.append(entry.artifacts["input_audio"])
                continue
            speaker = entry.artifacts.get("input_embedding")
            if model_type == "ChatterboxTurboTTS":
                inputs.append((text, speaker, 0.8, 1.2, 0.95, 1000, True))
            elif model_type == "ChatterboxMultilingualTTS":
                inputs.append((text, "en", speaker, 0.5, 0.5, 0.8, 1.2, 1.0, 0.05))
            else:
                inputs.append((text, speaker, 0.5, 0.5, 0.8, 1.2, 1.0, 0.05))
        elif model_type == "ChatterboxEmbedding":
            inputs.append(entry.artifacts["input_audio"])
        elif model_type == "ChatterboxVC":
            inputs.append(
                (
                    entry.artifacts["source_audio"],
                    None,
                    entry.artifacts.get("target_audio"),
                )
            )
        elif model_type == "FasterWhisper":
            inputs.append(entry.artifacts["input_audio"])
        elif model_type == "KittenTTSSynthesizer":
            inputs.append((text, "Jasper", speed))
        elif model_type == "MeloTTSSynthesizer":
            phones = torch.arange(1, max(2, len(text)) + 1, dtype=torch.long)
            item = (
                torch.zeros(8, phones.numel()),
                torch.zeros(8, phones.numel()),
                phones,
                torch.zeros(phones.numel(), dtype=torch.long),
                torch.zeros(phones.numel(), dtype=torch.long),
            )
            inputs.append(([item], 0, speed, {"noise_scale": 0.0, "noise_scale_w": 0.0, "sdp_ratio": 0.2}))
        elif model_type == "OpenVoiceV1Synthesizer":
            phones = torch.arange(1, max(2, len(text)) + 1, dtype=torch.long)
            inputs.append(([phones], 0, speed, {"noise_scale": 0.0, "noise_scale_w": 0.0, "sdp_ratio": 0.2}))
        elif model_type == "HarmonySpeechSynthesizer":
            chars = list(range(1, max(2, len(text)) + 1))
            inputs.append((chars, np.zeros(4, dtype=np.float32), speed, lambda x: x, lambda x: x))
        elif model_type == "HarmonySpeechVocoder":
            mel = entry.artifacts["input_audio"]
            inputs.append((mel, mel.breaks))
        elif model_type == "HarmonySpeechEncoder":
            inputs.append(entry.artifacts["input_audio"])
        elif model_type in ("OpenVoiceV1ToneConverterEncoder", "OpenVoiceV2ToneConverterEncoder"):
            # The fake VAD yields exactly one segment, so the faithful segment
            # set is one row (the real prepare slices the VAD segments).
            segment = AudioArtifact.from_numpy(
                np.full((SAMPLE_RATE,), 0.1, dtype=np.float32), SAMPLE_RATE
            )
            inputs.append([segment])
        elif model_type in ("OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"):
            request = entry.request_data
            if isinstance(request, VoiceConversionRequestInput):
                # VC reads source_audio + target_embedding and uses the target
                # embedding as the source too (no native source voice).
                audio = entry.artifacts["source_audio"]
                target = entry.artifacts.get("target_embedding")
                source = target
            else:
                audio = entry.artifacts["input_audio"]
                target = entry.artifacts.get("input_embedding") or SpeakerArtifact(
                    embedding=torch.ones(1, 4, 1), backend="openvoice"
                )
                source = SpeakerArtifact(embedding=torch.ones(1, 4, 1), backend="openvoice")
            inputs.append((audio, target, source))
        else:  # pragma: no cover - harness guard
            raise AssertionError(f"no stub prepare for {model_type}")
    return inputs


class EngineHarness:
    def __init__(
        self,
        configs,
        models,
        prepare=_prepare_inputs,
        *,
        executor_delay_s=0.0,
        engine_kwargs=None,
    ):
        self.configs = configs
        self.models = models
        self.prepare = prepare
        self.executor_delay_s = float(executor_delay_s)
        self.engine_kwargs = dict(engine_kwargs or {})
        self.engine = None
        self.serving = None
        self._patchers = []

    async def __aenter__(self):
        self._patchers = [
            patch("harmonyspeech.executor.cpu_executor.CPUExecutor._init_worker"),
            patch(
                "harmonyspeech.task_handler.model_runner_base.prepare_inputs",
                side_effect=self.prepare,
            ),
            patch(
                "harmonyspeech.task_handler.model_runner_base.get_model_config",
                side_effect=lambda *args, **kwargs: _fake_hf_config(args[1] if len(args) > 1 else "?"),
            ),
            patch(
                "harmonyspeech.task_handler.model_runner_base.get_model_flavour",
                return_value="default",
            ),
            # The OpenVoice VAD step is part of the real clone flow; the stub
            # pipeline yields one deterministic segment so the step forwards a
            # typed VadArtifact to the encoder exactly like the real one.
            patch(
                "harmonyspeech.task_handler.model_runner_base.BatchedInferencePipeline",
                _FakeWhisperPipeline,
            ),
            # The HarmonySpeech clone path runs its encoder on the pack WAV;
            # the real numpy frontend is irrelevant to the routing/cache gates,
            # so it is replaced by one deterministic partial-utterance row.
            patch(
                "harmonyspeech.task_handler.model_runner_base.preprocess_wav",
                side_effect=lambda wav, source_sr=None: wav,
            ),
            patch(
                "harmonyspeech.task_handler.model_runner_base.get_input_frames",
                side_effect=lambda wav: np.zeros((1, 3, 40), dtype=np.float32),
            ),
        ]
        for patcher in self._patchers:
            patcher.start()
        self.engine = AsyncHarmonySpeech(
            model_configs=self.configs,
            log_stats=False,
            start_engine_loop=True,
            **self.engine_kwargs,
        )
        cache = getattr(self.engine.engine, "embedding_cache", None)
        executors = {}
        for config in self.configs:
            runner = _runner(config, self.models[config.name])
            if cache is not None:
                # Mirror ``ModelRunnerBase._refresh_model_identity`` so the
                # Phase 7-2 cache keys are the production ones.
                runner.embedding_cache = cache
                version = cache.bump_load_version(config.name)
                runner.model_identity = "|".join(
                    (
                        config.name,
                        str(config.model_type),
                        str(config.dtype),
                        f"{config.model}@{config.revision or 'main'}",
                        f"v{version}",
                    )
                )
            executors[config.name] = StubExecutor(runner, delay_s=self.executor_delay_s)
        self.engine.engine.model_executors = executors
        self.serving = OpenAIServingTextToSpeech(
            self.engine, OpenAIServingTextToSpeech.models_from_config(self.configs)
        )
        return self

    async def __aexit__(self, *exc):
        warmup = getattr(self.engine, "_voice_pack_warmup_task", None)
        if warmup is not None and not warmup.done():
            warmup.cancel()
            with pytest.raises(asyncio.CancelledError):
                await warmup
        loop = getattr(self.engine, "_background_loop_unshielded", None)
        if loop is not None and not loop.done():
            loop.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loop
        for patcher in reversed(self._patchers):
            patcher.stop()
        return False


def _chatterbox_harness(model_type="ChatterboxTTS", *, fail_on=None):
    config = _model_config("chatterbox", model_type)
    model = ChatterboxStubModel(model_type=model_type, fail_on=fail_on)
    return EngineHarness([config], {"chatterbox": model}), model


def _kitten_harness():
    config = _model_config("kitten-mini", "KittenTTSSynthesizer", voices=["Jasper"])
    model = KittenStubModel()
    return EngineHarness([config], {"kitten-mini": model}), model


def _request(text, model="chatterbox", **options):
    is_kitten = model.startswith("kitten")
    return TextToSpeechRequest(
        model=model,
        input=text,
        mode="single_speaker_tts",
        voice="Jasper" if is_kitten else None,
        language="default" if is_kitten else None,
        generation_options=GenerationOptions(**options),
    )


class DisconnectedRequest:
    async def is_disconnected(self):
        return False


RAW = DisconnectedRequest()


async def _speak(harness, request):
    return await harness.serving.create_text_to_speech(request, RAW)


def _decode(response):
    from harmonyspeech.common.audio_codec import decode_audio

    assert isinstance(response, TextToSpeechResponse), response
    waveform, sr = decode_audio(base64.b64decode(response.data))
    return waveform, sr


TEXT_3 = "First sentence here. Second sentence here. Third sentence here."
TEXT_5 = (
    "One sentence here. Two sentence here. Three sentence here. "
    "Four sentence here. Five sentence here."
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_native_speed_model_types_are_pinned():
    assert frozenset(
        {
            "HarmonySpeechSynthesizer",
            "OpenVoiceV1Synthesizer",
            "MeloTTSSynthesizer",
            "KittenTTSSynthesizer",
        }
    ) == NATIVE_SPEED_MODEL_TYPES


@pytest.mark.integration
def test_cjk_language_hint_reaches_the_splitter():
    from harmonyspeech.common.inputs import TextToSpeechRequestInput

    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    request_data = TextToSpeechRequestInput(
        request_id="cjk",
        requested_model="chatterbox_multilingual",
        model="chatterbox_multilingual",
        input_text="第一句。第二句。第三句。",
        mode="single_speaker_tts",
        language_id="zh",
        generation_options=GenerationOptions(split_text=True, chunk_size=6),
    )
    chunks = serving._resolve_chunks(request_data)
    assert chunks == ["第一句。", "第二句。", "第三句。"]


@pytest.mark.integration
def test_default_chunk_budget_is_300_characters():
    from harmonyspeech.common.inputs import TextToSpeechRequestInput

    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    text = " ".join(f"Sentence number {index} is here." for index in range(60))
    request_data = TextToSpeechRequestInput(
        request_id="budget",
        requested_model="chatterbox",
        model="chatterbox",
        input_text=text,
        mode="single_speaker_tts",
        generation_options=GenerationOptions(split_text=True),
    )
    chunks = serving._resolve_chunks(request_data)
    assert chunks is not None and len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_turbo_default_chunk_budget_is_the_unified_300_characters():
    """Turbo's unset ``chunk_size`` resolves to the unified 300-char default.

    Phase 12-7/12-8: the 150-char workaround was a symptom of AR KV-slot
    truncation; the 1536-token cap fixes that at the source.
    """
    from harmonyspeech.common.audio_utils import split_text_into_chunks

    config = _model_config("chatterbox_turbo", "ChatterboxTurboTTS")
    model = ChatterboxStubModel(model_type="ChatterboxTurboTTS")
    harness = EngineHarness([config], {"chatterbox_turbo": model})
    text = " ".join(f"Sentence number {index} is here." for index in range(60))
    async with harness:
        response = await _speak(
            harness, _request(text, model="chatterbox_turbo", split_text=True)
        )

    assert isinstance(response, TextToSpeechResponse), response
    expected = split_text_into_chunks(text, chunk_size=300)
    assert len(expected) > 1
    assert len(model.generate_calls) == len(expected)
    assert all(len(chunk) <= 300 for chunk in model.generate_calls)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multilingual_default_chunk_budget_is_the_unified_300_characters():
    """Multilingual's unset ``chunk_size`` resolves to the unified 300-char default.

    Phase 12-7: the 75-char workaround masked premature analyzer EOS, which the
    `repetition_penalty=2.0` + opt-in-analyzer fixes address at the source.
    """
    from harmonyspeech.common.audio_utils import split_text_into_chunks

    config = _model_config("chatterbox_multilingual", "ChatterboxMultilingualTTS")
    model = ChatterboxStubModel(model_type="ChatterboxMultilingualTTS")
    harness = EngineHarness([config], {"chatterbox_multilingual": model})
    text = " ".join(f"Sentence number {index} is here." for index in range(60))
    request = TextToSpeechRequest(
        model="chatterbox_multilingual",
        input=text,
        mode="single_speaker_tts",
        language="en",
        generation_options=GenerationOptions(split_text=True),
    )
    async with harness:
        response = await _speak(harness, request)

    assert isinstance(response, TextToSpeechResponse), response
    expected = split_text_into_chunks(text, chunk_size=300)
    assert len(expected) > 1
    assert len(model.generate_calls) == len(expected)
    assert all(len(chunk) <= 300 for chunk in model.generate_calls)


@pytest.mark.integration
def test_chunk_size_alone_enables_chunking():
    from harmonyspeech.common.inputs import TextToSpeechRequestInput

    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    request_data = TextToSpeechRequestInput(
        request_id="size-only",
        requested_model="chatterbox",
        model="chatterbox",
        input_text="AAAA. BBBB. CCCC.",
        mode="single_speaker_tts",
        generation_options=GenerationOptions(chunk_size=8),
    )
    assert serving._resolve_chunks(request_data) == ["AAAA.", "BBBB.", "CCCC."]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crossfade_shortens_the_stitched_buffer():
    harness, _model = _chatterbox_harness()
    async with harness:
        response = await _speak(
            harness,
            _request(
                "AAAA. BBBB. CCCC.",
                split_text=True,
                chunk_size=8,
                crossfade_ms=2,
                sentence_pause_ms=0,
            ),
        )
    waveform, sr = _decode(response)
    chunk_len = len("AAAA.") * SAMPLES_PER_CHAR
    fade = int(sr * 2 / 1000)
    assert len(waveform) == chunk_len * 3 - fade * 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unset_join_fields_use_the_tuned_40_120_defaults():
    """Phase 12-5: chunking on + join fields unset => 40 ms fade / 120 ms pause."""
    harness, _model = _chatterbox_harness()
    text = (
        "First sentence has quite a few words in it. "
        "Second sentence has quite a few words in it."
    )
    async with harness:
        response = await _speak(harness, _request(text, split_text=True, chunk_size=45))
    waveform, sr = _decode(response)

    assert sr == SAMPLE_RATE
    chunks = [
        "First sentence has quite a few words in it.",
        "Second sentence has quite a few words in it.",
    ]
    total = sum(len(chunk) * SAMPLES_PER_CHAR for chunk in chunks)
    fade = int(sr * 40 / 1000)
    pause = int(sr * 120 / 1000)
    assert len(waveform) == total + pause - fade
    # The pre-12-5 unset defaults were 0/0 (bare sum); this pins that they moved.
    assert len(waveform) > total


@pytest.mark.integration
@pytest.mark.asyncio
async def test_short_text_is_byte_identical_to_unchunked():
    harness, _model = _chatterbox_harness()
    async with harness:
        plain = await _speak(harness, _request("One short sentence."))
        chunked = await _speak(
            harness, _request("One short sentence.", split_text=True, chunk_size=300)
        )
    assert plain.data == chunked.data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ordered_stitch_uses_chunk_index_order():
    harness, model = _chatterbox_harness()
    async with harness:
        response = await _speak(
            harness,
            _request(
                "AAAA. BBBB. CCCC.",
                split_text=True,
                chunk_size=8,
                crossfade_ms=0,
                sentence_pause_ms=10,
            ),
        )
    waveform, sr = _decode(response)
    assert sr == SAMPLE_RATE
    # Each chunk is a constant 0.1 of its own length; no crossfade => the
    # stitched buffer is chunk0 + pause + chunk1 + pause + chunk2.
    chunk_len = len("AAAA.") * SAMPLES_PER_CHAR
    pause = int(SAMPLE_RATE * 10 / 1000)
    assert model.generate_calls == ["AAAA.", "BBBB.", "CCCC."]
    assert len(waveform) == chunk_len * 3 + pause * 2
    # Ordering: each chunk carries a distinct constant, so the stitched blocks
    # prove chunk-index order (never completion order).
    values = [("A", 0.16), ("B", 0.17), ("C", 0.18)]
    for index, (letter, expected) in enumerate(values):
        start = index * (chunk_len + pause)
        assert np.allclose(waveform[start : start + chunk_len], expected, atol=2e-3), letter
    assert np.allclose(waveform[chunk_len : chunk_len + pause], 0.0, atol=1e-6)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clone_embeds_exactly_once_for_n_chunks():
    harness, model = _chatterbox_harness()
    request = TextToSpeechRequest(
        model="chatterbox",
        input=TEXT_3,
        mode="voice_cloning",
        input_audio=_reference_audio_b64(),
        generation_options=GenerationOptions(split_text=True, chunk_size=40),
    )
    async with harness:
        response = await _speak(harness, request)
    assert isinstance(response, TextToSpeechResponse), response
    assert model.prepare_calls == 1
    assert len(model.generate_calls) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_watermark_runs_exactly_once_per_request():
    harness, model = _chatterbox_harness()
    async with harness:
        await _speak(harness, _request(TEXT_3, split_text=True, chunk_size=40))
    assert model.watermarker.calls == 1

    # Baseline: the un-chunked path still watermarks exactly once (inside the
    # model's own generate), so the chunked and single-shot contracts agree.
    model.watermarker.calls = 0
    async with _chatterbox_harness()[0] as baseline_harness:
        baseline_model = baseline_harness.models["chatterbox"]
        await _speak(baseline_harness, _request("One short sentence."))
    assert baseline_model.watermarker.calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_failure_fails_the_whole_request():
    harness, _model = _chatterbox_harness(fail_on="FAIL")
    async with harness:
        response = await _speak(
            harness,
            _request("Good sentence. FAIL sentence. Last sentence.", split_text=True, chunk_size=16),
        )
    from harmonyspeech.endpoints.openai.protocol import ErrorResponse

    assert isinstance(response, ErrorResponse)
    assert "failed" in response.message.lower() or "failure" in response.message.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_seeds_are_derived_and_stable():
    harness, _model = _kitten_harness()
    seen: list[str] = []

    def prepare(model_config, requests_to_batch):
        out = _prepare_inputs(model_config, requests_to_batch)
        for _model_input, entry in zip(out, requests_to_batch, strict=True):
            options = getattr(entry.request_data, "generation_options", None)
            seed = getattr(options, "seed", None)
            seen.append(seed)
        return out

    harness.prepare = prepare
    request = _request(
        TEXT_3, model="kitten-mini", split_text=True, chunk_size=40, seed=100
    )
    async with harness:
        first = await _speak(harness, request)
        first_seen = list(seen)
        seen.clear()
        second = await _speak(harness, request)
        second_seen = list(seen)
    assert first_seen == [100, 101, 102]
    assert second_seen == [100, 101, 102]
    assert first.data == second.data


# ---------------------------------------------------------------------------
# Phase 12-4 finding 3 (2026-11-09): per-chunk determinism through the lanes
#
# The host's ``determinism:run1-vs-run2`` gate failed for KittenTTS with the
# signature of a chunk crossing (same duration, corr ~0.992). The model itself
# turned out to sample on every inference (its ONNX graph carries unseeded
# ``RandomNormalLike`` / ``RandomUniformLike`` nodes), so that axis can only be
# distributional. What must still hold — and what this test pins — is the
# *engine* invariant underneath it: KittenTTS executes one request per lane
# (Phase 8-5), and every lane's result must land on its own request, in chunk
# index order, identically on every run.
# ---------------------------------------------------------------------------

#: Four one-word sentences: 4 chunks at ``chunk_size=6``, each with a distinct
#: first character so its block in the stitched buffer is identifiable.
LANE_CHUNK_TEXT = "AAAA. BBBB. CCCC. DDDD."


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kitten_chunked_lanes_pair_by_index_and_repeat_byte_identically():
    config = _model_config("kitten-mini", "KittenTTSSynthesizer", voices=["Jasper"])
    config.max_batch_size = 4  # one lane per chunk
    model = KittenTextStubModel()
    harness = EngineHarness([config], {"kitten-mini": model})
    request = _request(
        LANE_CHUNK_TEXT,
        model="kitten-mini",
        split_text=True,
        chunk_size=6,
        crossfade_ms=0,
        sentence_pause_ms=0,
    )

    async with harness:
        first = await _speak(harness, request)
        first_calls = list(model.generate_calls)
        model.generate_calls.clear()
        second = await _speak(harness, request)
        second_calls = list(model.generate_calls)
        batch_sizes = harness.engine.engine.model_executors["kitten-mini"].batch_sizes

    assert isinstance(first, TextToSpeechResponse), first
    assert isinstance(second, TextToSpeechResponse), second
    # The same chunked request synthesises the same chunk texts in index order.
    assert first_calls == ["AAAA.", "BBBB.", "CCCC.", "DDDD."]
    assert second_calls == first_calls
    # The four chunks really were admitted together (one step, four lanes) and
    # executed on more than one lane thread: this is the lane path, not the
    # single-lane loop.
    assert max(batch_sizes) >= 4, batch_sizes
    assert len(model.thread_ids) >= 2, model.thread_ids
    # Byte-identical stitched output for a deterministic model.
    assert first.data == second.data

    # Each chunk's block is the audio of *its own* chunk text, in chunk order:
    # a chunk→result crossing would put another letter's value there.
    waveform, sr = _decode(first)
    assert sr == SAMPLE_RATE
    chunk_len = len("AAAA.") * SAMPLES_PER_CHAR
    assert waveform.size == chunk_len * 4
    for index, expected in enumerate((0.16, 0.17, 0.18, 0.19)):
        block = waveform[index * chunk_len : (index + 1) * chunk_len]
        assert np.allclose(block, expected, atol=2e-3), (index, block[:4])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_native_family_applies_post_stitch_time_stretch():
    harness, _model = _chatterbox_harness()
    text = "One sentence here. Two sentence here."
    async with harness:
        normal = await _speak(
            harness,
            _request(
                text, split_text=True, chunk_size=30, crossfade_ms=0, sentence_pause_ms=0
            ),
        )
        fast = await _speak(
            harness,
            _request(
                text,
                split_text=True,
                chunk_size=30,
                speed=2.0,
                crossfade_ms=0,
                sentence_pause_ms=0,
            ),
        )
    normal_wave, _ = _decode(normal)
    fast_wave, _ = _decode(fast)
    assert len(fast_wave) == pytest.approx(len(normal_wave) / 2, rel=0.08)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_native_speed_family_is_not_double_stretched():
    harness, _model = _kitten_harness()
    text = "One sentence here. Two sentence here."
    async with harness:
        normal = await _speak(
            harness,
            _request(
                text,
                model="kitten-mini",
                split_text=True,
                chunk_size=30,
                crossfade_ms=0,
                sentence_pause_ms=0,
            ),
        )
        fast = await _speak(
            harness,
            _request(
                text,
                model="kitten-mini",
                split_text=True,
                chunk_size=30,
                speed=2.0,
                crossfade_ms=0,
                sentence_pause_ms=0,
            ),
        )
    normal_wave, _ = _decode(normal)
    fast_wave, _ = _decode(fast)
    # Native synthesis already halved it; a second stretch would quarter it.
    assert len(fast_wave) == pytest.approx(len(normal_wave) / 2, rel=0.08)


def _reference_audio_b64():
    import struct

    samples = 16000
    pcm = struct.pack("<" + "h" * samples, *([1000] * samples))
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        SAMPLE_RATE,
        SAMPLE_RATE * 2,
        2,
        16,
        b"data",
        len(pcm),
    )
    return base64.b64encode(header + pcm).decode()


# ---------------------------------------------------------------------------
# Per-family coverage (March list): every family produces a valid WAV and
# duration grows with the chunk count when a sentence pause is requested.
# ---------------------------------------------------------------------------


class _UnusedStubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))


def _single_speaker_family(family: str):
    """Return ``(harness, model_name, request_kwargs)`` for one family."""
    if family == "chatterbox":
        harness, _model = _chatterbox_harness("ChatterboxTTS")
        return harness, "chatterbox", {}
    if family == "turbo":
        config = _model_config("chatterbox_turbo", "ChatterboxTurboTTS")
        return (
            EngineHarness(
                [config],
                {"chatterbox_turbo": ChatterboxStubModel(model_type="ChatterboxTurboTTS")},
            ),
            "chatterbox_turbo",
            {},
        )
    if family == "multilingual":
        config = _model_config("chatterbox_multilingual", "ChatterboxMultilingualTTS")
        return (
            EngineHarness(
                [config],
                {"chatterbox_multilingual": ChatterboxStubModel(model_type="ChatterboxMultilingualTTS")},
            ),
            "chatterbox_multilingual",
            {"language": "en"},
        )
    if family.startswith("kitten"):
        config = _model_config(family, "KittenTTSSynthesizer", voices=["Jasper"])
        return (
            EngineHarness([config], {family: KittenStubModel()}),
            family,
            {"language": "default", "voice": "Jasper"},
        )
    if family == "melo-en":
        config = _model_config(
            "melo-en", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"]
        )
        return (
            EngineHarness([config], {"melo-en": MeloStubModel()}),
            "melo-en",
            {"language": "EN", "voice": "EN-Newest"},
        )
    if family == "ov1-synth-en":
        config = _model_config(
            "ov1-synth-en", "OpenVoiceV1Synthesizer", language="EN", voices=["EN-Newest"]
        )
        return (
            EngineHarness([config], {"ov1-synth-en": OpenVoiceStubModel()}),
            "ov1-synth-en",
            {"language": "EN", "voice": "EN-Newest"},
        )
    if family == "harmonyspeech":
        configs = [
            _model_config("hs1-encoder", "HarmonySpeechEncoder"),
            _model_config("hs1-synthesizer", "HarmonySpeechSynthesizer"),
            _model_config("hs1-vocoder", "HarmonySpeechVocoder"),
        ]
        models = {
            "hs1-encoder": _UnusedStubModel(),
            "hs1-synthesizer": HarmonySpeechSynthStubModel(),
            "hs1-vocoder": HarmonySpeechVocoderStubModel(),
        }
        return (
            EngineHarness(configs, models),
            "harmonyspeech",
            {"input_embedding": base64.b64encode(b"\x00" * 16).decode()},
        )
    raise AssertionError(f"unknown family {family}")


SINGLE_SPEAKER_FAMILIES = [
    "chatterbox",
    "turbo",
    "multilingual",
    "kitten-mini",
    "kitten-micro",
    "kitten-nano",
    "kitten-nano-int8",
    "melo-en",
    "ov1-synth-en",
    "harmonyspeech",
]


@pytest.mark.integration
@pytest.mark.parametrize("family", SINGLE_SPEAKER_FAMILIES)
@pytest.mark.asyncio
async def test_per_family_chunked_wav_and_duration_growth(family):
    harness, model_name, extra = _single_speaker_family(family)
    async with harness:
        plain = await _speak(
            harness,
            TextToSpeechRequest(
                model=model_name,
                input=TEXT_3,
                mode="single_speaker_tts",
                generation_options=GenerationOptions(split_text=True, chunk_size=1000),
                **extra,
            ),
        )
        chunked = await _speak(
            harness,
            TextToSpeechRequest(
                model=model_name,
                input=TEXT_3,
                mode="single_speaker_tts",
                generation_options=GenerationOptions(
                    split_text=True,
                    chunk_size=40,
                    crossfade_ms=0,
                    sentence_pause_ms=50,
                ),
                **extra,
            ),
        )

    plain_wave, plain_sr = _decode(plain)
    chunked_wave, chunked_sr = _decode(chunked)
    assert plain_sr == chunked_sr and plain_sr > 0
    assert plain_wave.size > 0 and chunked_wave.size > 0
    pause = int(chunked_sr * 50 / 1000)
    if family == "harmonyspeech":
        # This pipeline appends a 150 ms silence break after *every* chunk. The
        # stitcher's edge conditioning (Phase 12-5) normalises each to the 10 ms
        # guard, so chunking removes more native silence than the 50 ms pause
        # adds: the joins are uniform now, not accumulating. Pin that direction
        # rather than the pre-12-5 growth.
        assert chunked_wave.size < plain_wave.size
    else:
        # Two extra sentence pauses (50 ms at the model rate) must be audible;
        # these families emit no native per-chunk edge silence to normalise.
        assert chunked_wave.size > plain_wave.size + pause


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_sub_requests_co_batch_in_one_engine_step():
    config = _model_config("chatterbox", "ChatterboxTTS")
    config.max_batch_size = 5
    model = ChatterboxStubModel()
    harness = EngineHarness([config], {"chatterbox": model})
    async with harness:
        response = await _speak(
            harness,
            _request(TEXT_5, split_text=True, chunk_size=18),
        )
    assert isinstance(response, TextToSpeechResponse), response
    executor = harness.engine.engine.model_executors["chatterbox"]
    assert max(executor.batch_sizes) >= 5, executor.batch_sizes


# ---------------------------------------------------------------------------
# OpenVoice V1 voice cloning through the full stub toolchain
# ---------------------------------------------------------------------------


def _openvoice_clone_harness():
    configs = [
        _model_config("faster-whisper", "FasterWhisper"),
        _model_config(
            "ov1-encoder", "OpenVoiceV1ToneConverterEncoder", language="EN", voices=["EN-Newest"]
        ),
        _model_config(
            "ov1-synth-en", "OpenVoiceV1Synthesizer", language="EN", voices=["EN-Newest"]
        ),
        _model_config("ov1-tone-converter", "OpenVoiceV1ToneConverter"),
    ]
    models = {
        "faster-whisper": _UnusedStubModel(),
        "ov1-encoder": OpenVoiceEncoderStubModel(),
        "ov1-synth-en": OpenVoiceStubModel(),
        "ov1-tone-converter": ToneConverterStubModel(),
    }
    return EngineHarness(configs, models), models


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openvoice_clone_chunking_embeds_once():
    harness, models = _openvoice_clone_harness()
    vad = (
        '{"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], '
        '"text": "hello", "source": "whisper"}'
    )
    request = TextToSpeechRequest(
        model="openvoice_v1",
        input=TEXT_3,
        mode="voice_cloning",
        language="EN",
        voice="EN-Newest",
        input_audio=_reference_audio_b64(),
        input_vad_data=vad,
        generation_options=GenerationOptions(split_text=True, chunk_size=40),
    )
    async with harness:
        response = await _speak(harness, request)
    assert isinstance(response, TextToSpeechResponse), response
    wave, sr = _decode(response)
    assert wave.size > 0 and sr > 0
    # The encoder step ran once (embedding-once); the synthesizer ran per chunk.
    encoder_batches = harness.engine.engine.model_executors["ov1-encoder"].batch_sizes
    assert sum(encoder_batches) == 1, encoder_batches


# ---------------------------------------------------------------------------
# OpenVoice V2 voice cloning (the 2026-11-09 chunked-clone regression)
#
# The Phase 12-4 host benchmark sent a chunked ``voice_cloning`` request to the
# OpenVoice V2 pipeline and got an instant 400:
#
#   request ID tts-...-embed is not of type TextToSpeechRequestInput or
#   SynthesisRequestInput
#
# Root cause: the serving dispatcher *always* built a standalone
# ``SpeechEmbeddingRequestInput`` for a clone request. When the request names a
# concrete synthesizer (``ov2-synthesizer-en`` = MeloTTS), no pipeline routing
# happens, so that embedding sub-request went straight to the synthesizer
# executor, whose ``prepare_inputs`` accepts only TTS/synthesis requests. The
# OpenVoice clone conditioning lives in the tone-converter encoder step, which
# only runs for the ``openvoice_v2`` toolchain id (``vad → embed → synthesize →
# tone_convert``).
#
# The 12-2 stubs accepted any request type for any model, which is why this
# escaped. ``_prepare_inputs`` is now type-strict (see
# ``_ACCEPTED_REQUEST_TYPES``), so both shapes below fail loudly if the
# dispatcher regresses.
# ---------------------------------------------------------------------------


def _openvoice_v2_clone_harness(*, engine_kwargs=None):
    configs = [
        _model_config("faster-whisper", "FasterWhisper"),
        _model_config("ov2-encoder", "OpenVoiceV2ToneConverterEncoder"),
        _model_config(
            "ov2-synth-en", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"]
        ),
        _model_config("ov2-tone-converter", "OpenVoiceV2ToneConverter"),
    ]
    models = {
        "faster-whisper": _UnusedStubModel(),
        "ov2-encoder": OpenVoiceEncoderStubModel(),
        "ov2-synth-en": MeloStubModel(),
        "ov2-tone-converter": ToneConverterStubModel(),
    }
    return EngineHarness(configs, models, engine_kwargs=engine_kwargs), models


def _ov2_clone_request(*, model="openvoice_v2", split_text=True, vad_data=None):
    return TextToSpeechRequest(
        model=model,
        input=TEXT_3,
        mode="voice_cloning",
        language="EN",
        voice="EN-Newest",
        input_audio=_reference_audio_b64(),
        input_vad_data=vad_data,
        generation_options=GenerationOptions(
            split_text=split_text, chunk_size=40 if split_text else None
        ),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openvoice_v2_clone_chunking_embeds_once():
    """Full real flow: vad → embed → synthesize (MeloTTS) → tone_convert.

    ``input_vad_data`` is deliberately absent, so the one-time embedding
    sub-request enters the pipeline at the VAD step and forwards to the
    tone-converter encoder — exactly what a non-chunked OV2 clone request does.
    """
    harness, _models = _openvoice_v2_clone_harness()
    async with harness:
        response = await _speak(harness, _ov2_clone_request())
    assert isinstance(response, TextToSpeechResponse), response
    wave, sr = _decode(response)
    assert wave.size > 0 and sr > 0

    executors = harness.engine.engine.model_executors
    assert sum(executors["faster-whisper"].batch_sizes) == 1
    assert sum(executors["ov2-encoder"].batch_sizes) == 1
    # One synthesize + one tone-convert row per chunk (they may co-batch into a
    # single step — that is the Phase 12-2 concurrency rule).
    assert sum(executors["ov2-synth-en"].batch_sizes) == 3
    assert sum(executors["ov2-tone-converter"].batch_sizes) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openvoice_v2_chunked_clone_uses_the_same_embedding_cache_key():
    """A later single-shot clone with the same reference hits the 7-2 cache.

    The chunked path's one-time embed must key exactly like the non-chunked
    pipeline's encoder step (same encoder identity, reference content hash and
    VAD source), so a following single-shot request recomputes nothing.
    """
    harness, _models = _openvoice_v2_clone_harness()
    async with harness:
        chunked = await _speak(harness, _ov2_clone_request())
        cache = harness.engine.engine.embedding_cache
        after_chunked = cache.stats()
        encoder = harness.models["ov2-encoder"].ref_enc
        forwards_after_chunked = encoder.forward_calls

        single_shot = await _speak(harness, _ov2_clone_request(split_text=False))
        after_single = cache.stats()

    assert isinstance(chunked, TextToSpeechResponse), chunked
    assert isinstance(single_shot, TextToSpeechResponse), single_shot
    assert after_chunked["stores"] == 1 and after_chunked["misses"] == 1
    assert after_single["hits"] == 1
    assert encoder.forward_calls == forwards_after_chunked


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunked_clone_of_a_concrete_synthesizer_skips_the_embed_step():
    """Regression: ``ov2-synthesizer-en`` + ``voice_cloning`` + ``split_text``.

    The non-chunked flow for a directly named MeloTTS executor never reads
    ``input_audio``, so the chunked flow must not invent an embedding step: it
    forwards the request fields to every chunk (N single-shot-equivalent
    requests) and never sends a ``SpeechEmbeddingRequestInput`` to MeloTTS.
    """
    config = _model_config(
        "ov2-synth-en", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"]
    )
    harness = EngineHarness([config], {"ov2-synth-en": MeloStubModel()})
    recorded: list[tuple[str, str]] = []
    async with harness:
        original_add = harness.engine.add_request

        async def recording_add(request_id, request_data, **kwargs):
            recorded.append((request_id, type(request_data).__name__))
            return await original_add(request_id, request_data, **kwargs)

        harness.engine.add_request = recording_add
        response = await _speak(harness, _ov2_clone_request(model="ov2-synth-en"))

    assert isinstance(response, TextToSpeechResponse), response
    wave, _sr = _decode(response)
    assert wave.size > 0
    # No one-time embedding sub-request was ever built…
    assert not any(request_id.endswith("-embed") for request_id, _ in recorded), recorded
    # …and every chunk is an ordinary single-shot-shaped TTS request.
    assert [name for _rid, name in recorded] == ["TextToSpeechRequestInput"] * 3, recorded


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunked_clone_of_a_directly_named_chatterbox_model_embeds_once():
    """A Chatterbox executor named directly still pre-embeds exactly once.

    ``ChatterboxTTS.prepare_inputs`` accepts a standalone
    ``SpeechEmbeddingRequestInput`` and the handler runs the same
    ``prepare_conditionals`` the single-shot clone path runs inline, so the
    embed-once hoist is faithful for this shape (and must stay).
    """
    config = _model_config("cb-tts", "ChatterboxTTS")
    model = ChatterboxStubModel(model_type="ChatterboxTTS")
    harness = EngineHarness([config], {"cb-tts": model})
    async with harness:
        response = await _speak(harness, _ov2_clone_request(model="cb-tts"))
    assert isinstance(response, TextToSpeechResponse), response
    assert model.prepare_calls == 1
    assert len(model.generate_calls) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_type_strict_stub_would_have_caught_the_embed_regression(monkeypatch):
    """The test gap itself, pinned: the old unconditional embed must fail.

    Phase 12-2's stubs accepted any request type, so the dispatcher could send a
    ``SpeechEmbeddingRequestInput`` to a MeloTTS executor without a test
    noticing. With the stubs type-strict, restoring the pre-fix behaviour (always
    pre-embed) makes the request fail exactly like the benchmark did.
    """
    monkeypatch.setattr(
        OpenAIServingTextToSpeech,
        "_reference_conditioning_mode",
        lambda self, request_data: "embed",  # noqa: ARG005 - the pre-fix behaviour
    )
    config = _model_config(
        "ov2-synth-en", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"]
    )
    harness = EngineHarness([config], {"ov2-synth-en": MeloStubModel()})
    async with harness:
        response = await _speak(harness, _ov2_clone_request(model="ov2-synth-en"))
    assert isinstance(response, ErrorResponse), response
    assert "-embed is not of type" in response.message, response.message


@pytest.mark.integration
def test_reference_conditioning_mode_resolves_per_pipeline():
    """The dispatcher's per-request discriminator, pinned for every shape."""

    class _Engine:
        def __init__(self, configs):
            self.model_configs = configs

    configs = [
        _model_config("cb-tts", "ChatterboxTTS"),
        _model_config("melo-en", "MeloTTSSynthesizer", language="EN"),
        _model_config("ov1-synth-en", "OpenVoiceV1Synthesizer", language="EN"),
        _model_config("kitten-mini", "KittenTTSSynthesizer"),
        _model_config("hs1-synth", "HarmonySpeechSynthesizer"),
    ]
    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    serving.engine = types.SimpleNamespace(engine=_Engine(configs))

    def mode(requested_model, model=None):
        return serving._reference_conditioning_mode(
            TextToSpeechRequestInput(
                request_id="mode",
                requested_model=requested_model,
                model=model or requested_model,
                input_text="text",
                mode="voice_cloning",
            )
        )

    # Pipelines whose steps produce an embedding artifact.
    for pipeline in ("openvoice_v1", "openvoice_v2", "chatterbox", "harmonyspeech"):
        assert mode(pipeline) == "embed", pipeline
    # Directly named Chatterbox executors accept a standalone embed request.
    assert mode("cb-tts") == "embed"
    # Concrete synthesizers reject it and never condition on the reference.
    for name in ("melo-en", "ov1-synth-en", "kitten-mini", "hs1-synth"):
        assert mode(name) == "none", name
    # A single-step pipeline id (no embedding producer) is not embed-capable.
    assert mode("kitten_tts", model="kitten-mini") == "none"
    # Pin the Chatterbox-only membership of the standalone-embed set.
    assert frozenset(
        {
            "ChatterboxEmbedding",
            "ChatterboxTTS",
            "ChatterboxTurboTTS",
            "ChatterboxMultilingualTTS",
        }
    ) == STANDALONE_EMBED_MODEL_TYPES


# ---------------------------------------------------------------------------
# Per-chunk deadline override under a slow loop executor (Phase 12-2
# follow-up, 2026-11-09)
#
# The Phase 12-4 host benchmark reproduced a real failure: 12 concurrent
# ``split_text`` requests to a ``max_batch_size=1`` family queued their chunk
# sub-requests, and tail chunks waited longer than the engine's global 120 s
# deadline, so they were expired before they ever ran. Queueing delay is not
# synthesis failure; the serving layer now stamps ``base_timeout *
# (chunk_count + 1)`` on every chunk sub-request. These tests pin both sides:
# the fix makes the burst succeed, and without it the same burst times out.
# ---------------------------------------------------------------------------

#: Four sentence-sized chunks (``chunk_size=24``) per request.
TEXT_4 = "One sentence here. Two sentence here. Three sentence here. Four sentence here."
SLOW_CHUNK_SIZE = 24
SLOW_CHUNK_COUNT = 4
SLOW_SYNTHESIS_S = 0.4
SLOW_BASE_TIMEOUT_S = 1.0


def _slow_chunk_harness():
    """A ``max_batch_size=1`` multilingual loop executor with 0.4 s synthesis."""
    config = _model_config("chatterbox_multilingual", "ChatterboxMultilingualTTS")
    config.max_batch_size = 1
    model = ChatterboxStubModel(model_type="ChatterboxMultilingualTTS")
    harness = EngineHarness(
        [config],
        {"chatterbox_multilingual": model},
        executor_delay_s=SLOW_SYNTHESIS_S,
        engine_kwargs={"request_timeout_s": SLOW_BASE_TIMEOUT_S},
    )
    return harness, model


def _slow_chunk_request():
    return TextToSpeechRequest(
        model="chatterbox_multilingual",
        input=TEXT_4,
        mode="single_speaker_tts",
        language="en",
        generation_options=GenerationOptions(split_text=True, chunk_size=SLOW_CHUNK_SIZE),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slow_loop_chunked_burst_survives_queue_delay():
    """Two concurrent 4-chunk requests (8 queued chunks) all succeed.

    The tail chunk waits ~7 * 0.4 s = 2.8 s in the queue, far beyond the 1 s
    engine-wide timeout, but well inside the chunk override
    ``1.0 * (4 + 1) = 5.0 s``.
    """
    harness, _model = _slow_chunk_harness()
    recorded: list[tuple[str, float | None]] = []
    async with harness:
        original_add = harness.engine.add_request

        async def recording_add(request_id, request_data, **kwargs):
            recorded.append((request_id, getattr(request_data, "request_timeout_s", None)))
            return await original_add(request_id, request_data, **kwargs)

        harness.engine.add_request = recording_add
        responses = await asyncio.gather(
            *[_speak(harness, _slow_chunk_request()) for _ in range(2)]
        )

    assert all(isinstance(response, TextToSpeechResponse) for response in responses), responses
    # Every chunk sub-request carries the extended override; nothing else does.
    chunk_timeouts = {
        timeout for request_id, timeout in recorded if "-chunk-" in request_id
    }
    assert chunk_timeouts == {SLOW_BASE_TIMEOUT_S * (SLOW_CHUNK_COUNT + 1)}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slow_loop_chunked_burst_expires_without_override():
    """Companion: pinning the override off reproduces the benchmark failure.

    With the chunk sub-requests inheriting the 1 s engine-wide timeout, the
    same burst expires in the queue and the request fails honestly (no partial
    audio with a 200).
    """
    harness, _model = _slow_chunk_harness()
    async with harness:
        # Disable the serving-layer override: chunks inherit the global 1 s.
        harness.serving._chunk_request_timeout_s = lambda chunk_count: None
        responses = await asyncio.gather(
            *[_speak(harness, _slow_chunk_request()) for _ in range(2)]
        )

    assert any(isinstance(response, ErrorResponse) for response in responses), responses
    errors = [response for response in responses if isinstance(response, ErrorResponse)]
    assert all("exceeded timeout" in response.message for response in errors), errors



# ---------------------------------------------------------------------------
# Phase 13-1: standard voice pack routing
#
# The pack ships reference WAVs; a pack name is resolved ONCE per HTTP request
# into the request's `input_audio` (+ cloning mode), before the boundary decode
# and the 12-2 chunk dispatcher. These tests drive the real serving layer and
# engine loop with the production-shaped ChatterboxEmbedding executor, so the
# 7-2 embedding cache assertions are real.
# ---------------------------------------------------------------------------


class ChatterboxVCStubModel(torch.nn.Module):
    """ChatterboxVC stand-in: target-voice prep + deterministic vocode."""

    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.sr = SAMPLE_RATE
        self.watermarker = CountingWatermarker()
        self.ref_dict = {"target": 0}
        self.target_calls = 0

    def set_target_voice(self, wav):  # noqa: ARG002 - upstream signature
        self.target_calls += 1
        self.ref_dict = {"target": self.target_calls}

    def generate(self, audio=None, **kwargs):  # noqa: ARG002 - upstream signature
        samples = max(1, int(audio.shape[-1])) if hasattr(audio, "shape") else SAMPLE_RATE
        return torch.full((1, samples), 0.1)


def _pack_chatterbox_harness(*, with_turbo=False):
    """Production-shaped chatterbox deployment: dedicated embed + TTS executors."""
    configs = [
        _model_config("chatterbox_embedding", "ChatterboxEmbedding"),
        _model_config("chatterbox", "ChatterboxTTS"),
    ]
    models = {
        "chatterbox_embedding": ChatterboxStubModel(model_type="ChatterboxEmbedding"),
        "chatterbox": ChatterboxStubModel(model_type="ChatterboxTTS"),
    }
    if with_turbo:
        configs.append(_model_config("chatterbox_turbo", "ChatterboxTurboTTS"))
        models["chatterbox_turbo"] = ChatterboxStubModel(model_type="ChatterboxTurboTTS")
    return EngineHarness(configs, models), models


def _pack_audio_hash(name):
    from harmonyspeech.common.audio_codec import decode_audio
    from harmonyspeech.modeling.voice_catalog import standard_voice_path

    path = standard_voice_path(name)
    assert path is not None and path.is_file()
    waveform, sample_rate = decode_audio(path.read_bytes())
    return AudioArtifact.from_numpy(waveform, sample_rate).content_hash


def _pack_request(model, voice="Abigail", text=None, language=None, **options):
    return TextToSpeechRequest(
        model=model,
        input=text or "Hello from the standard voice pack.",
        mode="voice_cloning",
        voice=voice,
        language=language,
        generation_options=GenerationOptions(**options),
    )


def _vc_serving(harness):
    from harmonyspeech.endpoints.openai.serving_voice_conversion import (
        OpenAIServingVoiceConversion,
    )

    return OpenAIServingVoiceConversion(
        harness.engine,
        OpenAIServingVoiceConversion.models_from_config(harness.configs),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_voice_resolves_to_the_shipped_clip_and_caches():
    """`voice: Abigail` becomes the pack WAV; the repeat request is a cache hit."""
    harness, models = _pack_chatterbox_harness()
    embed = models["chatterbox_embedding"]
    tts = models["chatterbox"]
    recorded = []

    async with harness:
        original_add = harness.engine.add_request

        async def recording_add(request_id, request_data, **kwargs):
            recorded.append((request_data, dict(kwargs.get("artifacts") or {})))
            return await original_add(request_id, request_data, **kwargs)

        harness.engine.add_request = recording_add
        first = await _speak(harness, _pack_request("chatterbox"))
        cache = harness.engine.engine.embedding_cache
        after_first = cache.stats()
        second = await _speak(harness, _pack_request("chatterbox"))
        after_second = cache.stats()

    assert isinstance(first, TextToSpeechResponse), first
    assert isinstance(second, TextToSpeechResponse), second
    # The dispatched request carries the pack WAV as its cloning reference and
    # the derived cloning mode; no native voice id is left behind.
    dispatched, artifacts = recorded[0]
    assert dispatched.mode == "voice_cloning"
    assert dispatched.voice_id is None
    assert artifacts["input_audio"].content_hash == _pack_audio_hash("Abigail")
    # One real embedding computation, reused by the repeat request (7-2).
    assert embed.prepare_calls == 1
    assert tts.prepare_calls == 0
    assert after_first["misses"] == 1 and after_first["stores"] == 1
    assert after_second["hits"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_voice_with_split_text_embeds_once_across_all_chunks():
    """Extends the 12-2 embed-once gate: a pack reference is embedded once."""
    harness, models = _pack_chatterbox_harness()
    embed = models["chatterbox_embedding"]
    tts = models["chatterbox"]

    async with harness:
        response = await _speak(
            harness,
            _pack_request(
                "chatterbox", text=TEXT_3, split_text=True, chunk_size=40
            ),
        )
        chunks_generated = len(tts.generate_calls)
        cache = harness.engine.engine.embedding_cache
        after_chunked = cache.stats()
        again = await _speak(
            harness,
            _pack_request(
                "chatterbox", text=TEXT_3, split_text=True, chunk_size=40
            ),
        )
        after_repeat = cache.stats()

    assert isinstance(response, TextToSpeechResponse), response
    assert isinstance(again, TextToSpeechResponse), again
    wave, sr = _decode(response)
    assert wave.size > 0 and sr > 0
    assert embed.prepare_calls == 1
    assert chunks_generated == 3
    assert len(tts.generate_calls) == 6
    assert after_chunked["stores"] == 1 and after_chunked["misses"] == 1
    assert after_repeat["hits"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_same_pack_voice_works_on_two_chatterbox_families():
    """The same name resolves on chatterbox and chatterbox_turbo (shared VE)."""
    harness, models = _pack_chatterbox_harness(with_turbo=True)
    async with harness:
        standard = await _speak(harness, _pack_request("chatterbox"))
        turbo = await _speak(harness, _pack_request("chatterbox_turbo"))

    assert isinstance(standard, TextToSpeechResponse), standard
    assert isinstance(turbo, TextToSpeechResponse), turbo
    for response in (standard, turbo):
        wave, sr = _decode(response)
        assert wave.size > 0 and sr > 0
    # One embedding identity (the shared VoiceEncoder) computed once, reused.
    assert models["chatterbox_embedding"].prepare_calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_voice_on_melo_is_an_honest_400():
    """A single-speaker model lists native voices only; a pack name never maps."""
    config = _model_config(
        "melo-en", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"]
    )
    harness = EngineHarness([config], {"melo-en": MeloStubModel()})
    request = TextToSpeechRequest(
        model="melo-en",
        input="Hello.",
        mode="single_speaker_tts",
        language="EN",
        voice="Abigail",
    )
    async with harness:
        response = await _speak(harness, request)
        dispatched = harness.engine.engine.model_executors["melo-en"].batch_sizes

    assert isinstance(response, ErrorResponse), response
    assert "standard voice pack name" in response.message
    assert "does not support voice cloning" in response.message
    assert "EN-Newest" in response.message
    assert dispatched == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_target_voice_on_chatterbox_vc_resolves_and_caches():
    """`target_voice: Abigail` becomes the VC target clip (and rides 7-2)."""
    config = _model_config("chatterbox_vc", "ChatterboxVC")
    model = ChatterboxVCStubModel()
    harness = EngineHarness([config], {"chatterbox_vc": model})
    request = VoiceConversionRequest(
        model="chatterbox_vc",
        source_audio=_reference_audio_b64(),
        target_voice="Abigail",
    )
    async with harness:
        serving_vc = _vc_serving(harness)
        first = await serving_vc.convert_voice(request, RAW)
        cache = harness.engine.engine.embedding_cache
        after_first = cache.stats()
        second = await serving_vc.convert_voice(request, RAW)
        after_second = cache.stats()

    assert isinstance(first, VoiceConversionResponse), first
    assert isinstance(second, VoiceConversionResponse), second
    assert model.target_calls == 1
    assert after_first["misses"] == 1 and after_first["stores"] == 1
    assert after_second["hits"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_voice_is_rejected_when_the_pack_is_disabled():
    """`standard_voice_pack: false` turns a pack name into an honest 400."""
    harness, _models = _pack_chatterbox_harness()
    harness.engine_kwargs["standard_voice_pack"] = False
    async with harness:
        response = await _speak(harness, _pack_request("chatterbox"))
    assert isinstance(response, ErrorResponse), response
    assert "standard voice pack is disabled" in response.message


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_voice_conflicts_with_an_explicit_reference():
    harness, _models = _pack_chatterbox_harness()
    request = TextToSpeechRequest(
        model="chatterbox",
        input="Hello.",
        mode="voice_cloning",
        voice="Abigail",
        input_audio=_reference_audio_b64(),
    )
    async with harness:
        response = await _speak(harness, request)
    assert isinstance(response, ErrorResponse), response
    assert "not both" in response.message


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_voice_on_openvoice_v2_clone_resolves_and_caches():
    """`voice: Abigail` on the openvoice_v2 TTS clone path embeds once.

    The pack name resolves to the shipped clip, the native source-voice default
    (``EN-Newest``) is applied, and the repeat request is a 7-2 cache hit.
    """
    harness, models = _openvoice_v2_clone_harness()
    encoder = models["ov2-encoder"]
    recorded = []

    async with harness:
        original_add = harness.engine.add_request

        async def recording_add(request_id, request_data, **kwargs):
            recorded.append(request_data)
            return await original_add(request_id, request_data, **kwargs)

        harness.engine.add_request = recording_add
        first = await _speak(
            harness, _pack_request("openvoice_v2", language="EN")
        )
        cache = harness.engine.engine.embedding_cache
        after_first = cache.stats()
        forwards_after_first = encoder.ref_enc.forward_calls
        second = await _speak(
            harness, _pack_request("openvoice_v2", language="EN")
        )
        after_second = cache.stats()

    assert isinstance(first, TextToSpeechResponse), first
    assert isinstance(second, TextToSpeechResponse), second
    # The dispatched clone request carries the pack WAV and the native default
    # source voice (OpenVoice still needs a valid style for its synthesizer).
    tts_requests = [
        request for request in recorded if isinstance(request, TextToSpeechRequestInput)
    ]
    assert tts_requests
    assert tts_requests[0].voice_id == "EN-Newest"
    assert tts_requests[0].mode == "voice_cloning"
    # One real encoder forward for the first request, none for the repeat.
    assert forwards_after_first == 1
    assert encoder.ref_enc.forward_calls == forwards_after_first
    assert after_first["misses"] == 1 and after_first["stores"] == 1
    assert after_second["hits"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_voice_on_harmonyspeech_clone_mode():
    """`voice: Abigail` drives the harmonyspeech encoder → synth → vocoder path."""
    configs = [
        _model_config("hs1-encoder", "HarmonySpeechEncoder"),
        _model_config("hs1-synthesizer", "HarmonySpeechSynthesizer"),
        _model_config("hs1-vocoder", "HarmonySpeechVocoder"),
    ]
    encoder = HarmonySpeechEncoderStubModel()
    models = {
        "hs1-encoder": encoder,
        "hs1-synthesizer": HarmonySpeechSynthStubModel(),
        "hs1-vocoder": HarmonySpeechVocoderStubModel(),
    }
    harness = EngineHarness(configs, models)
    recorded = []

    async with harness:
        original_add = harness.engine.add_request

        async def recording_add(request_id, request_data, **kwargs):
            recorded.append(request_data)
            return await original_add(request_id, request_data, **kwargs)

        harness.engine.add_request = recording_add
        first = await _speak(harness, _pack_request("harmonyspeech"))
        cache = harness.engine.engine.embedding_cache
        after_first = cache.stats()
        forwards_after_first = encoder.forward_calls
        second = await _speak(harness, _pack_request("harmonyspeech"))
        after_second = cache.stats()

    assert isinstance(first, TextToSpeechResponse), first
    assert isinstance(second, TextToSpeechResponse), second
    tts_requests = [
        request for request in recorded if isinstance(request, TextToSpeechRequestInput)
    ]
    assert tts_requests
    # Harmony Speech has no native voices, so the pack voice clears voice_id.
    assert tts_requests[0].voice_id is None
    assert tts_requests[0].mode == "voice_cloning"
    assert forwards_after_first == 1
    assert encoder.forward_calls == forwards_after_first
    assert after_first["misses"] == 1 and after_first["stores"] == 1
    assert after_second["hits"] == 1


def _ov2_vc_request(model="openvoice_v2"):
    return VoiceConversionRequest(
        model=model,
        source_audio=_reference_audio_b64(),
        target_voice="Abigail",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_target_voice_on_openvoice_v2_embeds_cold_then_caches():
    """OV VC `target_voice` runs the pipeline embed step once, then rides 7-2.

    The pack clip is embedded through ``vad → embed`` and set as
    ``target_embedding`` (never ``target_audio``, which the tone converter
    ignores). Cold = one encoder forward; warm = zero.
    """
    harness, models = _openvoice_v2_clone_harness()
    encoder = models["ov2-encoder"]
    recorded = []

    async with harness:
        original_add = harness.engine.add_request

        async def recording_add(request_id, request_data, **kwargs):
            recorded.append((request_data, dict(kwargs.get("artifacts") or {})))
            return await original_add(request_id, request_data, **kwargs)

        harness.engine.add_request = recording_add
        serving_vc = _vc_serving(harness)
        first = await serving_vc.convert_voice(_ov2_vc_request(), RAW)
        cache = harness.engine.engine.embedding_cache
        after_first = cache.stats()
        forwards_after_first = encoder.ref_enc.forward_calls
        second = await serving_vc.convert_voice(_ov2_vc_request(), RAW)
        after_second = cache.stats()

    assert isinstance(first, VoiceConversionResponse), first
    assert isinstance(second, VoiceConversionResponse), second
    # One embedding sub-request per HTTP request, routed at the pipeline id
    # (VAD first), then the VC request itself.
    embed_requests = [
        request
        for request, _artifacts in recorded
        if isinstance(request, SpeechEmbeddingRequestInput)
    ]
    assert len(embed_requests) == 2
    assert all(request.requested_model == "openvoice_v2" for request in embed_requests)
    vc_artifacts = [
        artifacts
        for request, artifacts in recorded
        if isinstance(request, VoiceConversionRequestInput)
    ]
    assert len(vc_artifacts) == 2
    for artifacts in vc_artifacts:
        assert isinstance(artifacts["target_embedding"], SpeakerArtifact)
        assert "target_audio" not in artifacts
    # Cold: exactly one encoder forward; warm: none.
    assert forwards_after_first == 1
    assert encoder.ref_enc.forward_calls == forwards_after_first
    assert after_first["misses"] == 1 and after_first["stores"] == 1
    assert after_second["hits"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_target_voice_on_openvoice_v2_is_warm_after_pack_warmup(tmp_path):
    """The VC target embedding shares the 13-2 warm-up cache: zero forwards.

    Warm-up computes the OV2 encoder artifact for every pack voice (through the
    same ``vad → embed`` route the VC request uses); the first VC request for a
    warmed voice must therefore perform no encoder forward.
    """
    harness, models = _openvoice_v2_clone_harness(
        engine_kwargs={
            "voice_pack_warmup": True,
            "voice_pack_cache_dir": str(tmp_path),
        }
    )
    encoder = models["ov2-encoder"]
    async with harness:
        stats = await harness.engine.start_voice_pack_warmup()
        warmed_forwards = encoder.ref_enc.forward_calls
        encoder.ref_enc.forward_calls = 0
        serving_vc = _vc_serving(harness)
        response = await serving_vc.convert_voice(_ov2_vc_request(), RAW)
        cache = harness.engine.engine.embedding_cache

    assert stats["failed"] == 0
    assert warmed_forwards > 0
    assert isinstance(response, VoiceConversionResponse), response
    # The VC path reuses the warm artifact: no encoder forward at all.
    assert encoder.ref_enc.forward_calls == 0
    assert cache.stats()["hits"] >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_target_voice_on_concrete_ov2_tone_converter():
    """A directly named tone converter resolves its sibling encoder pipeline."""
    harness, models = _openvoice_v2_clone_harness()
    encoder = models["ov2-encoder"]
    recorded = []

    async with harness:
        original_add = harness.engine.add_request

        async def recording_add(request_id, request_data, **kwargs):
            recorded.append(request_data)
            return await original_add(request_id, request_data, **kwargs)

        harness.engine.add_request = recording_add
        serving_vc = _vc_serving(harness)
        response = await serving_vc.convert_voice(
            _ov2_vc_request(model="ov2-tone-converter"), RAW
        )

    assert isinstance(response, VoiceConversionResponse), response
    embed_requests = [
        request
        for request in recorded
        if isinstance(request, SpeechEmbeddingRequestInput)
    ]
    assert len(embed_requests) == 1
    # The concrete converter routes its embedding through the pipeline id.
    assert embed_requests[0].requested_model == "openvoice_v2"
    assert encoder.ref_enc.forward_calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pack_target_voice_on_a_tone_converter_without_encoder_is_a_400():
    """No configured encoder/VAD => honest 400, never a half-run request."""
    config = _model_config("ov2-tone-converter", "OpenVoiceV2ToneConverter")
    harness = EngineHarness([config], {"ov2-tone-converter": ToneConverterStubModel()})
    async with harness:
        serving_vc = _vc_serving(harness)
        response = await serving_vc.convert_voice(
            _ov2_vc_request(model="ov2-tone-converter"), RAW
        )
    assert isinstance(response, ErrorResponse), response
    assert "does not accept a standard target voice" in response.message


# ---------------------------------------------------------------------------
# Phase 13-2: persistent pack-embedding cache + warm-up
#
# The disk layer persists the *same* artifact the 7-2 handler caches, so these
# tests use a Chatterbox stub whose conditionals are a real upstream
# ``Conditionals`` (the shared chunking stub's SimpleNamespace is deliberately
# not serializable). Warm-up goes through the real engine loop/scheduler; the
# second-boot gate proves hydration serves a request with zero embed forwards.
# ---------------------------------------------------------------------------


class _SerializableChatterboxStubModel(ChatterboxStubModel):
    """Chatterbox stub whose conditionals round-trip through the disk store."""

    def prepare_conditionals(self, audio_buf, **kwargs):  # noqa: ARG002 - upstream signature
        from chatterbox.tts import Conditionals, T3Cond
        from harmonyspeech.common.audio_codec import decode_audio

        waveform, _sr = decode_audio(audio_buf.read())
        self.prepare_calls += 1
        self.conds = Conditionals(
            T3Cond(
                speaker_emb=torch.tensor([float(waveform[0]), float(waveform.std())]),
                emotion_adv=torch.zeros(1, 1, 1),
            ),
            {"ge_bes": torch.tensor([float(waveform.std())])},
        )


class _FailingEmbedStubModel(ChatterboxStubModel):
    """Every conditioning pass fails: warm-up must log, skip and keep serving."""

    def prepare_conditionals(self, audio_buf, **kwargs):  # noqa: ARG002 - upstream signature
        self.prepare_calls += 1
        raise RuntimeError("synthetic warm-up embedding failure")


def _warmup_harness(cache_dir, *, embed_model=None):
    """Production-shaped chatterbox deployment with the 13-2 knobs enabled."""
    configs = [
        _model_config("chatterbox_embedding", "ChatterboxEmbedding"),
        _model_config("chatterbox", "ChatterboxTTS"),
    ]
    models = {
        "chatterbox_embedding": embed_model
        or _SerializableChatterboxStubModel(model_type="ChatterboxEmbedding"),
        "chatterbox": ChatterboxStubModel(model_type="ChatterboxTTS"),
    }
    harness = EngineHarness(
        configs,
        models,
        engine_kwargs={
            "voice_pack_warmup": True,
            "voice_pack_cache_dir": str(cache_dir),
        },
    )
    return harness, models


def _warmup_conditionals_artifact():
    from chatterbox.tts import Conditionals, T3Cond

    return SpeakerArtifact(
        conditionals=Conditionals(
            T3Cond(speaker_emb=torch.zeros(1, 2), emotion_adv=torch.zeros(1, 1, 1)),
            {"ge_bes": torch.tensor([0.0])},
        ),
        backend="chatterbox",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warmup_populates_disk_and_memory(tmp_path):
    """Warm-up computes every pack voice once and persists each artifact."""
    from harmonyspeech.modeling.voice_catalog import standard_voice_names

    harness, models = _warmup_harness(tmp_path)
    embed = models["chatterbox_embedding"]
    voices = standard_voice_names()

    async with harness:
        task = harness.engine.start_voice_pack_warmup()
        assert task is not None
        stats = await task
        cache = harness.engine.engine.embedding_cache
        entries = cache.stats()["entries"]

    assert stats["targets"] == len(voices)
    assert stats["computed"] == len(voices)
    assert stats["persisted"] == len(voices)
    assert stats["failed"] == 0
    assert embed.prepare_calls == len(voices)
    assert entries == len(voices)
    assert len(list(tmp_path.glob("*.pt"))) == len(voices)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_second_boot_with_warm_disk_does_zero_embed_forwards(tmp_path):
    """A warm disk hydrates the 7-2 cache: the first request is a pure hit."""
    from harmonyspeech.modeling.voice_catalog import standard_voice_names
    from harmonyspeech.modeling.voice_pack_cache import hydrate_embedding_cache

    first_boot, _models = _warmup_harness(tmp_path)
    async with first_boot:
        await first_boot.engine.start_voice_pack_warmup()

    second_boot, models = _warmup_harness(tmp_path)
    embed = models["chatterbox_embedding"]
    async with second_boot:
        restored = hydrate_embedding_cache(second_boot.engine.engine)
        response = await _speak(second_boot, _pack_request("chatterbox"))

    assert restored == len(standard_voice_names())
    assert isinstance(response, TextToSpeechResponse), response
    # The whole point: no conditioning pass on the first request after restart.
    assert embed.prepare_calls == 0
    assert models["chatterbox"].generate_calls


@pytest.mark.integration
@pytest.mark.asyncio
async def test_corrupt_disk_entry_is_ignored_and_recomputed(tmp_path):
    """A corrupt file never fails boot; the on-demand/warm-up path recomputes it."""
    from harmonyspeech.modeling.voice_catalog import standard_voice_names
    from harmonyspeech.modeling.voice_pack_cache import hydrate_embedding_cache

    first_boot, _models = _warmup_harness(tmp_path)
    async with first_boot:
        await first_boot.engine.start_voice_pack_warmup()

    files = sorted(tmp_path.glob("*.pt"))
    assert len(files) == len(standard_voice_names())
    files[0].write_bytes(b"this is not a torch checkpoint")

    second_boot, models = _warmup_harness(tmp_path)
    embed = models["chatterbox_embedding"]
    async with second_boot:
        restored = hydrate_embedding_cache(second_boot.engine.engine)
        stats = await second_boot.engine.start_voice_pack_warmup()

    assert restored == len(standard_voice_names()) - 1
    assert stats["computed"] == 1
    assert stats["cache_hits"] == len(standard_voice_names()) - 1
    assert embed.prepare_calls == 1
    assert len(list(tmp_path.glob("*.pt"))) == len(standard_voice_names())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warmup_failure_does_not_block_boot_or_serving(tmp_path):
    """Every voice may fail; the task returns stats and the engine still serves."""
    from harmonyspeech.modeling.voice_catalog import standard_voice_names

    harness, models = _warmup_harness(
        tmp_path, embed_model=_FailingEmbedStubModel(model_type="ChatterboxEmbedding")
    )
    embed = models["chatterbox_embedding"]

    async with harness:
        stats = await harness.engine.start_voice_pack_warmup()
        response = await _speak(harness, _pack_request("chatterbox"))

    assert stats["failed"] == len(standard_voice_names())
    assert stats["computed"] == 0
    # Warm-up attempted every voice; the follow-up request attempted one more.
    assert embed.prepare_calls >= len(standard_voice_names())
    assert isinstance(response, ErrorResponse), response
    assert len(list(tmp_path.glob("*.pt"))) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_on_demand_compute_dedupes_warmup(tmp_path):
    """A voice an on-demand request already cached is not recomputed by warm-up."""
    from harmonyspeech.modeling.voice_catalog import standard_voice_names
    from harmonyspeech.modeling.voice_pack_cache import build_warmup_targets

    harness, models = _warmup_harness(tmp_path)
    embed = models["chatterbox_embedding"]
    voices = standard_voice_names()

    async with harness:
        inner = harness.engine.engine
        cache = inner.embedding_cache
        target = next(t for t in build_warmup_targets(inner) if t.voice == "Abigail")
        # Simulate an on-demand request that won the race and cached the voice.
        cache.put(
            target.cache_key,
            _warmup_conditionals_artifact(),
            model_identity=target.model_identity,
        )
        stats = await harness.engine.start_voice_pack_warmup()

    assert stats["cache_hits"] == 1
    assert stats["computed"] == len(voices) - 1
    assert embed.prepare_calls == len(voices) - 1
    # The already-computed voice still lands on disk for the next boot.
    assert len(list(tmp_path.glob("*.pt"))) == len(voices)

