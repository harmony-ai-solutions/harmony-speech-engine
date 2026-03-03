"""E2E test conftest.py — marks all e2e tests and provides model download fixtures."""
import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in tests/e2e/ with @pytest.mark.e2e."""
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def models_cache_dir(tmp_path_factory):
    """
    Session-scoped temp directory for model weight caching during E2E tests.
    Models downloaded here are shared across all e2e tests in one session.
    """
    return tmp_path_factory.mktemp("models_cache")
