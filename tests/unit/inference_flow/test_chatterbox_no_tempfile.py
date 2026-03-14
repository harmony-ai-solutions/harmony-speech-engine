"""Unit test: verify Chatterbox execution path does not create temporary files.

REQ-PERF-01: No filesystem I/O (tempfiles, open()) during Chatterbox inference.
All audio processing uses in-memory BytesIO objects exclusively.
"""
import io
import base64
from unittest.mock import patch, MagicMock
import pytest

import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chatterbox_model():
    """Mock Chatterbox TTS model that returns a valid audio tensor."""
    model = MagicMock()
    fake_audio = torch.zeros(1, 24000, dtype=torch.float32)
    model.generate.return_value = fake_audio
    model.sr = 24000
    return model


@pytest.fixture
def mock_chatterbox_embedding_model():
    """Mock Chatterbox model that supports prepare_conditionals for embedding."""
    model = MagicMock()
    fake_conds = MagicMock()
    model.conds = fake_conds
    model.prepare_conditionals.return_value = None
    return model


@pytest.fixture
def model_runner_with_tts_model(mock_chatterbox_model):
    """ModelRunnerBase instance wired with a mock ChatterboxTTS model."""
    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
    from harmonyspeech.common.config import ModelConfig, DeviceConfig

    device_config = MagicMock(spec=DeviceConfig)
    device_config.device = torch.device("cpu")

    model_config = MagicMock(spec=ModelConfig)
    model_config.model_type = "ChatterboxTTS"
    model_config.name = "chatterbox"

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = mock_chatterbox_model
    runner.model_config = model_config
    runner.device_config = device_config
    return runner


@pytest.fixture
def model_runner_with_embedding_model(mock_chatterbox_embedding_model):
    """ModelRunnerBase instance wired with a mock ChatterboxEmbedding model."""
    from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase
    from harmonyspeech.common.config import ModelConfig, DeviceConfig

    device_config = MagicMock(spec=DeviceConfig)
    device_config.device = torch.device("cpu")

    model_config = MagicMock(spec=ModelConfig)
    model_config.model_type = "ChatterboxEmbedding"
    model_config.name = "chatterbox_embedding"

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = mock_chatterbox_embedding_model
    runner.model_config = model_config
    runner.device_config = device_config
    return runner


def _make_tts_request(text="Hello world"):
    """Create a minimal fake TTS batch entry."""
    from harmonyspeech.common.inputs import TextToSpeechRequestInput
    req = MagicMock()
    req.request_data = MagicMock(spec=TextToSpeechRequestInput)
    req.request_id = "test-tts-001"
    return req


def _make_embed_request():
    """Create a minimal fake embedding batch entry."""
    from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput
    req = MagicMock()
    req.request_data = MagicMock(spec=SpeechEmbeddingRequestInput)
    req.request_id = "test-emb-001"
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatterboxNoTempfile:

    def test_execute_chatterbox_tts_no_open_call(self, model_runner_with_tts_model):
        """_execute_chatterbox_tts() must not call builtins.open() for file I/O."""
        runner = model_runner_with_tts_model
        inputs = [("Hello world", None, 0.5, 0.5, 0.8, 1.0, 0)]
        requests = [_make_tts_request()]

        with patch("builtins.open") as mock_open, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("tempfile.mkstemp") as mock_mkstemp:
            try:
                runner._execute_chatterbox_tts(inputs, requests)
            except Exception:
                pass  # model mock may not produce full output; we only care about file I/O
            mock_open.assert_not_called()
            mock_tmp.assert_not_called()
            mock_mkstemp.assert_not_called()

    def test_execute_chatterbox_embedding_no_open_call(self, model_runner_with_embedding_model):
        """_execute_chatterbox_embedding() must not call builtins.open() or tempfile."""
        runner = model_runner_with_embedding_model
        fake_audio = b"\x00" * 1000
        inputs = [fake_audio]
        requests = [_make_embed_request()]

        with patch("builtins.open") as mock_open, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("tempfile.mkstemp") as mock_mkstemp:
            try:
                runner._execute_chatterbox_embedding(inputs, requests)
            except Exception:
                pass  # model mock may not produce full output; we only care about file I/O
            mock_open.assert_not_called()
            mock_tmp.assert_not_called()
            mock_mkstemp.assert_not_called()

    def test_bytesio_used_not_tempfile(self, model_runner_with_embedding_model):
        """_execute_chatterbox_embedding() uses io.BytesIO for in-memory processing."""
        runner = model_runner_with_embedding_model
        fake_audio = b"\x00" * 1000
        inputs = [fake_audio]
        requests = [_make_embed_request()]

        original_bytesio = io.BytesIO
        bytesio_calls = []

        def tracking_bytesio(*args, **kwargs):
            obj = original_bytesio(*args, **kwargs)
            bytesio_calls.append(obj)
            return obj

        with patch("io.BytesIO", side_effect=tracking_bytesio):
            try:
                runner._execute_chatterbox_embedding(inputs, requests)
            except Exception:
                pass
        # At least one BytesIO should have been created for in-memory audio processing
        assert len(bytesio_calls) >= 1, "Expected io.BytesIO to be used for in-memory audio processing"