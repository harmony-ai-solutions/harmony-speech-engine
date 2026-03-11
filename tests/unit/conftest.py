"""Unit test conftest.py — mock fixtures for isolated unit tests."""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_model_loader():
    """Returns a MagicMock that stands in for harmonyspeech.modeling.loader functions."""
    with patch("harmonyspeech.modeling.loader.get_model_class") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_hf_downloader():
    """Patches HuggingFace Hub download to prevent network calls in unit tests."""
    with patch("harmonyspeech.modeling.hf_downloader.download_model") as mock:
        mock.return_value = "/tmp/mock_model_path"
        yield mock
