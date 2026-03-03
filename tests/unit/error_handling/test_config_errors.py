"""Unit tests for config error paths and _get_and_verify_dtype edge cases."""
import pytest
import torch
from harmonyspeech.common.config import _get_and_verify_dtype, EngineConfig


# ── _get_and_verify_dtype ─────────────────────────────────────────────────────

def test_dtype_auto_resolves_to_float16():
    """'auto' should resolve to float16 because config_dtype is float32."""
    result = _get_and_verify_dtype("auto")
    assert result == torch.float16


def test_dtype_float32_string():
    assert _get_and_verify_dtype("float32") == torch.float32


def test_dtype_float_alias():
    assert _get_and_verify_dtype("float") == torch.float32


def test_dtype_float16_string():
    assert _get_and_verify_dtype("float16") == torch.float16


def test_dtype_half_alias():
    assert _get_and_verify_dtype("half") == torch.float16


def test_dtype_bfloat16_string():
    assert _get_and_verify_dtype("bfloat16") == torch.bfloat16


def test_dtype_torch_float32_passthrough():
    """torch.dtype values are returned as-is."""
    result = _get_and_verify_dtype(torch.float32)
    assert result == torch.float32


def test_dtype_torch_float16_passthrough():
    result = _get_and_verify_dtype(torch.float16)
    assert result == torch.float16


def test_dtype_unknown_string_raises():
    with pytest.raises(ValueError, match="Unknown dtype"):
        _get_and_verify_dtype("INVALID")


def test_dtype_integer_raises():
    with pytest.raises(ValueError, match="Unknown dtype"):
        _get_and_verify_dtype(42)


def test_dtype_none_raises():
    with pytest.raises((ValueError, AttributeError)):
        _get_and_verify_dtype(None)


# ── EngineConfig file-loading errors ─────────────────────────────────────────

def test_engine_config_missing_yaml_raises():
    """Loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        EngineConfig.load_config_from_yaml("/tmp/does_not_exist_xyz_abc.yml")
