"""Unit tests for HarmonySpeechEngine initialization logic."""
import pytest
from unittest.mock import MagicMock, patch
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine


@pytest.fixture
def cpu_model_config():
    """A minimal CPU-targeted ModelConfig."""
    return ModelConfig(
        name="tts-cpu",
        model="path/to/model",
        model_type="MeloTTSSynthesizer",
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


@pytest.fixture
def second_cpu_model_config():
    """A second CPU-targeted ModelConfig with a different name."""
    return ModelConfig(
        name="stt-cpu",
        model="path/to/stt",
        model_type="FasterWhisper",
        max_batch_size=2,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


@pytest.fixture
def gpu_model_config():
    """A GPU-targeted ModelConfig."""
    return ModelConfig(
        name="tts-gpu",
        model="path/to/model",
        model_type="MeloTTSSynthesizer",
        max_batch_size=8,
        device_config=DeviceConfig("cuda"),
        dtype="float16",
    )


@patch("harmonyspeech.processing.scheduler.Scheduler")
@patch("harmonyspeech.executor.cpu_executor.CPUExecutor")
def test_engine_creates_one_executor_per_model(mock_cpu_exec, mock_scheduler, cpu_model_config):
    """Engine with one CPU model creates one executor keyed by model name."""
    mock_cpu_exec.return_value = MagicMock()
    mock_scheduler.return_value = MagicMock()

    engine = HarmonySpeechEngine(
        model_configs=[cpu_model_config],
        log_stats=False,
    )

    assert len(engine.model_executors) == 1
    assert "tts-cpu" in engine.model_executors


@patch("harmonyspeech.processing.scheduler.Scheduler")
@patch("harmonyspeech.executor.cpu_executor.CPUExecutor")
def test_engine_creates_executors_for_multiple_models(
    mock_cpu_exec, mock_scheduler, cpu_model_config, second_cpu_model_config
):
    """Engine with two models creates two executors keyed by their names."""
    mock_cpu_exec.return_value = MagicMock()
    mock_scheduler.return_value = MagicMock()

    engine = HarmonySpeechEngine(
        model_configs=[cpu_model_config, second_cpu_model_config],
        log_stats=False,
    )

    assert len(engine.model_executors) == 2
    assert "tts-cpu" in engine.model_executors
    assert "stt-cpu" in engine.model_executors


@patch("harmonyspeech.processing.scheduler.Scheduler")
@patch("harmonyspeech.executor.cpu_executor.CPUExecutor")
def test_engine_uses_cpu_executor_for_cpu_device(mock_cpu_exec, mock_scheduler, cpu_model_config):
    """CPUExecutor is instantiated for CPU-device model configs."""
    mock_instance = MagicMock()
    mock_cpu_exec.return_value = mock_instance
    mock_scheduler.return_value = MagicMock()

    engine = HarmonySpeechEngine(
        model_configs=[cpu_model_config],
        log_stats=False,
    )

    mock_cpu_exec.assert_called_once_with(model_config=cpu_model_config)
    assert engine.model_executors["tts-cpu"] is mock_instance


@patch("harmonyspeech.processing.scheduler.Scheduler")
@patch("harmonyspeech.executor.gpu_executor.GPUExecutorAsync")
def test_engine_uses_gpu_executor_for_gpu_device(mock_gpu_exec, mock_scheduler, gpu_model_config):
    """GPUExecutorAsync is instantiated for GPU-device model configs."""
    mock_instance = MagicMock()
    mock_gpu_exec.return_value = mock_instance
    mock_scheduler.return_value = MagicMock()

    engine = HarmonySpeechEngine(
        model_configs=[gpu_model_config],
        log_stats=False,
    )

    mock_gpu_exec.assert_called_once_with(model_config=gpu_model_config)
    assert engine.model_executors["tts-gpu"] is mock_instance


@patch("harmonyspeech.processing.scheduler.Scheduler")
@patch("harmonyspeech.executor.cpu_executor.CPUExecutor")
def test_engine_no_stat_logger_when_log_stats_false(mock_cpu_exec, mock_scheduler, cpu_model_config):
    """When log_stats=False, no stat_logger is attached to engine."""
    mock_cpu_exec.return_value = MagicMock()
    mock_scheduler.return_value = MagicMock()

    engine = HarmonySpeechEngine(
        model_configs=[cpu_model_config],
        log_stats=False,
    )

    assert not hasattr(engine, "stat_logger")


@patch("harmonyspeech.processing.scheduler.Scheduler")
@patch("harmonyspeech.executor.cpu_executor.CPUExecutor")
def test_engine_creates_stat_logger_when_log_stats_true(mock_cpu_exec, mock_scheduler, cpu_model_config):
    """When log_stats=True, stat_logger is attached to engine."""
    mock_cpu_exec.return_value = MagicMock()
    mock_scheduler.return_value = MagicMock()

    engine = HarmonySpeechEngine(
        model_configs=[cpu_model_config],
        log_stats=True,
    )

    assert hasattr(engine, "stat_logger")


@patch("harmonyspeech.processing.scheduler.Scheduler")
@patch("harmonyspeech.executor.cpu_executor.CPUExecutor")
def test_engine_stores_model_configs(mock_cpu_exec, mock_scheduler, cpu_model_config):
    """Engine stores the model_configs list."""
    mock_cpu_exec.return_value = MagicMock()
    mock_scheduler.return_value = MagicMock()

    engine = HarmonySpeechEngine(
        model_configs=[cpu_model_config],
        log_stats=False,
    )

    assert engine.model_configs == [cpu_model_config]


@patch("harmonyspeech.processing.scheduler.Scheduler")
@patch("harmonyspeech.executor.cpu_executor.CPUExecutor")
def test_engine_executor_keyed_by_model_name(mock_cpu_exec, mock_scheduler, cpu_model_config):
    """Engine model_executors dict is keyed by model_cfg.name."""
    mock_cpu_exec.return_value = MagicMock()
    mock_scheduler.return_value = MagicMock()

    engine = HarmonySpeechEngine(
        model_configs=[cpu_model_config],
        log_stats=False,
    )

    # Verify the key is the model name from ModelConfig
    assert "tts-cpu" in engine.model_executors
    # Verify we can access the executor by its model name
    executor = engine.model_executors[cpu_model_config.name]
    assert executor is not None
