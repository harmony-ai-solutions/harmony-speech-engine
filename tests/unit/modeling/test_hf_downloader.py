"""Unit tests for harmonyspeech/modeling/hf_downloader.py — caching and resolution."""

from unittest.mock import patch, MagicMock, call

import pytest

from harmonyspeech.modeling.hf_downloader import (
    load_or_download_config,
    load_or_download_file,
    load_or_download_model,
    _resolve_file_path,
    _config_cache,
    _file_cache,
    HParams,
)
from huggingface_hub.errors import LocalEntryNotFoundError


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear caches before and after each test for isolation."""
    _config_cache.clear()
    _file_cache.clear()
    yield
    _config_cache.clear()
    _file_cache.clear()


@pytest.fixture
def tmp_config_file(tmp_path):
    """Create a temporary config JSON file on disk."""
    import json

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"data": {"sampling_rate": 22050}}))
    return str(tmp_path), "config.json"


@pytest.fixture
def tmp_binary_file(tmp_path):
    """Create a temporary binary file on disk."""
    binary_file = tmp_path / "speaker.pth"
    binary_file.write_bytes(b"\x80\x00\x01\x02fake_pth_data")
    return str(tmp_path), "speaker.pth"


# ===========================================================================
# load_or_download_config caching
# ===========================================================================


@pytest.mark.unit
def test_config_cached_after_first_load(tmp_config_file):
    """Second call returns the cached HParams object without re-reading disk."""
    base_path, filename = tmp_config_file

    result1 = load_or_download_config(base_path, filename)
    result2 = load_or_download_config(base_path, filename)

    # Same object reference — proves it came from cache, not a re-read
    assert result1 is result2
    assert isinstance(result1, HParams)
    assert result1.data.sampling_rate == 22050


@pytest.mark.unit
def test_config_cache_hits_do_not_call_hf_hub_download(tmp_config_file):
    """After initial load, hf_hub_download must NOT be called again."""
    base_path, filename = tmp_config_file

    with patch("harmonyspeech.modeling.hf_downloader.hf_hub_download") as mock_dl:
        load_or_download_config(base_path, filename)  # populates cache
        mock_dl.assert_not_called()  # local file exists, so no download at all

        # Now call again — should still not invoke download
        load_or_download_config(base_path, filename)
        mock_dl.assert_not_called()


@pytest.mark.unit
def test_config_cache_different_keys_do_not_collide(tmp_path):
    """Different model paths produce different cache entries."""
    import json

    path_a = tmp_path / "model_a"
    path_b = tmp_path / "model_b"
    path_a.mkdir()
    path_b.mkdir()
    (path_a / "config.json").write_text(json.dumps({"data": {"sampling_rate": 16000}}))
    (path_b / "config.json").write_text(json.dumps({"data": {"sampling_rate": 22050}}))

    result_a = load_or_download_config(str(path_a), "config.json")
    result_b = load_or_download_config(str(path_b), "config.json")

    assert result_a.data.sampling_rate == 16000
    assert result_b.data.sampling_rate == 22050
    assert len(_config_cache) == 2


# ===========================================================================
# load_or_download_file caching
# ===========================================================================


@pytest.mark.unit
def test_file_cached_after_first_load(tmp_binary_file):
    """Second call returns the cached bytes without re-reading disk."""
    base_path, filename = tmp_binary_file

    result1 = load_or_download_file(base_path, filename)
    result2 = load_or_download_file(base_path, filename)

    assert result1 is result2  # same bytes object from cache
    assert result1 == b"\x80\x00\x01\x02fake_pth_data"


@pytest.mark.unit
def test_file_cache_does_not_call_hf_hub_download_on_hit(tmp_binary_file):
    """After initial load, hf_hub_download must NOT be called again."""
    base_path, filename = tmp_binary_file

    with patch("harmonyspeech.modeling.hf_downloader.hf_hub_download") as mock_dl:
        load_or_download_file(base_path, filename)  # populates cache
        load_or_download_file(base_path, filename)  # should hit cache
        mock_dl.assert_not_called()


# ===========================================================================
# _resolve_file_path — three-tier resolution
# ===========================================================================


@pytest.mark.unit
def test_resolve_tier1_local_filesystem(tmp_path):
    """Tier 1: local filesystem path is used when the file exists on disk."""
    f = tmp_path / "config.json"
    f.write_text('{"ok": true}')

    result = _resolve_file_path(str(tmp_path), "config.json")

    assert result == str(f)


@pytest.mark.unit
def test_resolve_tier1_does_not_hit_hf_at_all(tmp_path):
    """When a local file exists, hf_hub_download is never called."""
    (tmp_path / "config.json").write_text("{}")

    with patch("harmonyspeech.modeling.hf_downloader.hf_hub_download") as mock_dl:
        _resolve_file_path(str(tmp_path), "config.json")
        mock_dl.assert_not_called()


@pytest.mark.unit
@patch("harmonyspeech.modeling.hf_downloader.hf_hub_download")
def test_resolve_tier2_hf_disk_cache_no_network(mock_dl):
    """Tier 2: when no local file, try HF cache with local_files_only=True first."""
    cached_path = "/fake/hf_cache/config.json"
    mock_dl.return_value = cached_path

    result = _resolve_file_path("myshell-ai/OpenVoice", "config.json")

    assert result == cached_path
    # Must have been called with local_files_only=True
    mock_dl.assert_called_once()
    assert mock_dl.call_args.kwargs["local_files_only"] is True


@pytest.mark.unit
@patch("harmonyspeech.modeling.hf_downloader.hf_hub_download")
def test_resolve_tier3_network_fallback(mock_dl):
    """Tier 3: when HF cache misses, falls back to full network download."""
    network_path = "/fake/downloaded/config.json"

    def _side_effect(**kwargs):
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not in cache")
        return network_path

    mock_dl.side_effect = _side_effect

    result = _resolve_file_path("myshell-ai/OpenVoice", "config.json")

    assert result == network_path
    assert mock_dl.call_count == 2
    # First call: local_files_only=True (cache check)
    assert mock_dl.call_args_list[0].kwargs["local_files_only"] is True
    # Second call: full download (no local_files_only)
    assert mock_dl.call_args_list[1].kwargs.get("local_files_only") is not True


@pytest.mark.unit
@patch("harmonyspeech.modeling.hf_downloader.hf_hub_download")
def test_resolve_tier3_logs_download_message(mock_dl):
    """A one-time INFO log is emitted when a network download is triggered."""
    mock_dl.side_effect = lambda **kw: (
        (_ for _ in ()).throw(LocalEntryNotFoundError("miss"))
        if kw.get("local_files_only")
        else "/fake/path"
    )

    with patch("harmonyspeech.modeling.hf_downloader.logger") as mock_logger:
        _resolve_file_path("myshell-ai/OpenVoice", "config.json")
        mock_logger.info.assert_called_once()
        assert "Downloading" in mock_logger.info.call_args[0][0]
