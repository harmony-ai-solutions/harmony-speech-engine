"""Root conftest.py — shared fixtures and pytest CLI hooks for all tests."""
import pytest
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI option registration
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Target device for tests: cpu (default) or cuda. "
             "Set to 'cuda' only if CUDA is available; otherwise the suite fails.",
    )
    parser.addoption(
        "--dtype",
        action="store",
        default="float32",
        help="Default tensor dtype for tests: float32 (default) or float16.",
    )

def pytest_configure(config):
    """Validate --device early; fail fast if CUDA is requested but not available."""
    device = config.getoption("--device", default="cpu")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.exit(
            "pytest --device=cuda requested but CUDA is not available on this machine. "
            "Aborting test run.",
            returncode=1,
        )

# ---------------------------------------------------------------------------
# Session-scoped shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device(request):
    """The target device string ('cpu' or 'cuda')."""
    return request.config.getoption("--device")

@pytest.fixture(scope="session")
def dtype(request):
    """The default tensor dtype string ('float32' or 'float16')."""
    return request.config.getoption("--dtype")

@pytest.fixture(scope="session")
def tests_root() -> Path:
    """Absolute path to the tests/ directory."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def test_data_dir(tests_root) -> Path:
    """Absolute path to tests/test-data/ for binary test assets."""
    return tests_root / "test-data"

@pytest.fixture
def sample_config() -> dict:
    """Minimal valid model configuration dict for unit testing config parsing."""
    return {
        "model_type": "KittenTTSSynthesizer",
        "hf_model_id": "harmony-ai/test-model",
        "device": "cpu",
        "dtype": "float32",
        "max_num_seqs": 1,
    }
