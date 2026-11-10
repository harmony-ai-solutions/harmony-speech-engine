"""Shared helpers for wire-contract tests (Phase 1).

The engine is replaced by :class:`FakeEngine`, which returns canned outputs but
optionally exercises the *real* boundary decoder on the audio fields it sees.
That makes it possible to test transports, output formats/sample rates and
input codecs end-to-end through FastAPI without downloading any model weights.
"""

from __future__ import annotations

import base64
import json
from contextlib import ExitStack
from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient

from harmonyspeech.common import audio_codec
from harmonyspeech.common.inputs import (
    AudioConversionRequestInput,
    DetectVoiceActivityRequestInput,
    SpeechEmbeddingRequestInput,
    SpeechTranscribeRequestInput,
    TextToSpeechRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.outputs import (
    AudioConversionRequestOutput,
    DetectVoiceActivityRequestOutput,
    SpeechEmbeddingRequestOutput,
    SpeechTranscriptionRequestOutput,
    TextToSpeechRequestOutput,
    VoiceConversionRequestOutput,
)
from harmonyspeech.endpoints.openai.protocol import ModelCard

TESTS_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = TESTS_ROOT / "test-data" / "samples"

#: Executor payload used by the fake engine: a 1 s 440 Hz tone at 44.1 kHz.
DEFAULT_OUTPUT_WAV = SAMPLES_DIR / "codec_tone.wav"

TRANSCRIPTION_PAYLOAD = {
    "text": "hello world",
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
    "info": {"language": "en"},
}


def sample_bytes(name: str) -> bytes:
    return (SAMPLES_DIR / name).read_bytes()


def sample_b64(name: str) -> str:
    return base64.b64encode(sample_bytes(name)).decode("ascii")


def wav_b64(path: Path | None = None) -> str:
    return base64.b64encode((path or DEFAULT_OUTPUT_WAV).read_bytes()).decode("ascii")


def synth_wav_b64(sample_rate: int = 24000, seconds: float = 1.0, amplitude: float = 0.3) -> str:
    """Synthesize a small WAV payload (used where fixtures are inconvenient)."""
    import io

    import numpy as np

    t = np.arange(int(sample_rate * seconds)) / sample_rate
    waveform = (amplitude * np.sin(2 * np.pi * 440 * t)).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _SingleOutputStream:
    """One-item async stream, shaped like ``engine.AsyncStream``."""

    def __init__(self, output):
        self._output = output
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return self._output


class FakeEngine:
    """Stand-in for :class:`AsyncHarmonySpeech` with optional boundary decoding."""

    def __init__(
        self,
        *,
        decode_inputs: bool = False,
        output_wav_b64: str | None = None,
        model_configs: list | None = None,
        standard_voice_pack: bool = True,
    ):
        self.decode_inputs = decode_inputs
        self.output_wav_b64 = output_wav_b64 or wav_b64()
        self.requests = []
        self.abort_calls = []
        # Catalog state read by ``GET /v1/audio/voices`` (Phase 13-3); the
        # serving layer reads the same attributes off the real engine.
        self.model_configs = list(model_configs or [])
        self.standard_voice_pack = bool(standard_voice_pack)

    async def get_model_configs(self):
        return []

    async def check_health(self):
        return {"healthy": True}

    async def abort(self, request_id: str):
        self.abort_calls.append(request_id)

    def _decode(self, payload: str | None, target_sr: int | None = None):
        if not self.decode_inputs or not payload:
            return None
        return audio_codec.decode_audio(base64.b64decode(payload), target_sr=target_sr)

    async def generate(self, request_id, request_data, artifacts=None):
        self.requests.append(request_data)
        yield self._make_output(request_id, request_data)

    async def add_request(self, request_id, request_data, artifacts=None, arrival_time=None):
        """Streaming-path hook (Phase 14-3): one canned output, like ``generate``.

        The binary/SSE envelopes submit chunk sub-requests sequentially and need
        the per-request id to abort one in flight, which is exactly
        ``AsyncHarmonySpeech.add_request``'s contract.
        """
        self.requests.append(request_data)
        return _SingleOutputStream(self._make_output(request_id, request_data))

    def _make_output(self, request_id, request_data):
        try:
            if isinstance(request_data, TextToSpeechRequestInput):
                self._decode(request_data.input_audio)
                return TextToSpeechRequestOutput(
                    request_id=request_id,
                    text=request_data.input_text,
                    output=self.output_wav_b64,
                    finish_reason="stop",
                )
            if isinstance(request_data, SpeechTranscribeRequestInput):
                self._decode(request_data.input_audio, target_sr=16000)
                return SpeechTranscriptionRequestOutput(
                    request_id=request_id,
                    output=json.dumps(TRANSCRIPTION_PAYLOAD),
                    finish_reason="stop",
                )
            if isinstance(request_data, DetectVoiceActivityRequestInput):
                self._decode(request_data.input_audio, target_sr=16000)
                return DetectVoiceActivityRequestOutput(
                    request_id=request_id,
                    output=json.dumps({"speech_activity": True, "segments": []}),
                    finish_reason="stop",
                )
            if isinstance(request_data, SpeechEmbeddingRequestInput):
                self._decode(request_data.input_audio, target_sr=16000)
                return SpeechEmbeddingRequestOutput(
                    request_id=request_id, output="ZGF0YQ==", finish_reason="stop"
                )
            if isinstance(request_data, VoiceConversionRequestInput):
                self._decode(request_data.source_audio)
                self._decode(request_data.target_audio)
                return VoiceConversionRequestOutput(
                    request_id=request_id, output=self.output_wav_b64, finish_reason="stop"
                )
            if isinstance(request_data, AudioConversionRequestInput):
                self._decode(request_data.source_audio)
                return AudioConversionRequestOutput(
                    request_id=request_id, output=self.output_wav_b64, finish_reason="stop"
                )
        except audio_codec.AudioDecodeError as exc:
            return self._error_output(request_id, request_data, str(exc))
        raise AssertionError(f"FakeEngine received unsupported request data: {type(request_data)!r}")

    def _error_output(self, request_id, request_data, message):
        if isinstance(request_data, TextToSpeechRequestInput):
            output = TextToSpeechRequestOutput(
                request_id=request_id, text=request_data.input_text, output="", finish_reason="error"
            )
        elif isinstance(request_data, SpeechTranscribeRequestInput):
            output = SpeechTranscriptionRequestOutput(request_id=request_id, output="", finish_reason="error")
        elif isinstance(request_data, DetectVoiceActivityRequestInput):
            output = DetectVoiceActivityRequestOutput(request_id=request_id, output="", finish_reason="error")
        elif isinstance(request_data, SpeechEmbeddingRequestInput):
            output = SpeechEmbeddingRequestOutput(request_id=request_id, output="", finish_reason="error")
        elif isinstance(request_data, VoiceConversionRequestInput):
            output = VoiceConversionRequestOutput(request_id=request_id, output="", finish_reason="error")
        else:
            output = AudioConversionRequestOutput(request_id=request_id, output="", finish_reason="error")
        output.error = message
        return output


class WireApp:
    """FastAPI app + TestClient with every serving module wired to a FakeEngine."""

    MODEL_IDS = {
        "tts": "test-tts",
        "stt": "test-stt",
        "vc": "test-vc",
        "embed": "test-embed",
        "vad": "test-vad",
        "ac": "test-ac",
    }

    def __init__(
        self,
        *,
        decode_inputs: bool = False,
        max_upload_bytes: int | None = None,
        model_configs: list | None = None,
        standard_voice_pack: bool = True,
    ):
        self.engine = FakeEngine(
            decode_inputs=decode_inputs,
            model_configs=model_configs,
            standard_voice_pack=standard_voice_pack,
        )
        self.max_upload_bytes = max_upload_bytes
        self._stack: ExitStack | None = None
        self.app = None
        self.client: TestClient | None = None

    def __enter__(self):
        from harmonyspeech.endpoints.openai import api_server
        from harmonyspeech.endpoints.openai.args import make_arg_parser
        from harmonyspeech.endpoints.openai.serving_audio_conversion import OpenAIServingAudioConversion
        from harmonyspeech.endpoints.openai.serving_speech_to_text import OpenAIServingSpeechToText
        from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
        from harmonyspeech.endpoints.openai.serving_voice_activity_detection import (
            OpenAIServingVoiceActivityDetection,
        )
        from harmonyspeech.endpoints.openai.serving_voice_conversion import OpenAIServingVoiceConversion
        from harmonyspeech.endpoints.openai.serving_voice_embed import OpenAIServingVoiceEmbedding
        from types import SimpleNamespace

        serving = {
            "openai_serving_tts": OpenAIServingTextToSpeech(
                self.engine, [ModelCard(id=self.MODEL_IDS["tts"])]
            ),
            "openai_serving_stt": OpenAIServingSpeechToText(self.engine, [ModelCard(id=self.MODEL_IDS["stt"])]),
            "openai_serving_vc": OpenAIServingVoiceConversion(self.engine, [ModelCard(id=self.MODEL_IDS["vc"])]),
            "openai_serving_embedding": OpenAIServingVoiceEmbedding(
                self.engine, [ModelCard(id=self.MODEL_IDS["embed"])]
            ),
            "openai_serving_vad": OpenAIServingVoiceActivityDetection(
                self.engine, [ModelCard(id=self.MODEL_IDS["vad"])]
            ),
            "openai_serving_ac": OpenAIServingAudioConversion(self.engine, [ModelCard(id=self.MODEL_IDS["ac"])]),
        }

        self._stack = ExitStack()
        for name, instance in serving.items():
            self._stack.enter_context(_patch_attr(api_server, name, instance))
        self._stack.enter_context(_patch_attr(api_server, "engine", self.engine))
        self._stack.enter_context(
            _patch_attr(api_server, "engine_args", SimpleNamespace(disable_log_stats=True))
        )

        args = make_arg_parser().parse_args([])
        self.app = api_server.build_app(args)
        if self.max_upload_bytes is not None:
            self.app.state.max_upload_bytes = self.max_upload_bytes
        self.client = TestClient(self.app)
        return self

    def __exit__(self, *exc_info):
        if self.client is not None:
            self.client.close()
        if self._stack is not None:
            self._stack.close()
        return False


def _patch_attr(obj, name, value):
    from unittest.mock import patch

    return patch.object(obj, name, value)
