"""Integration test conftest.py — app and async client fixtures."""
import pytest


@pytest.fixture(scope="module")
def test_app():
    """
    Provides a FastAPI TestClient for integration tests.
    
    Note: Full app startup (model loading) is NOT performed here —
    integration tests in Phase 3 will configure this fixture with
    mock models. This fixture is a placeholder that will be expanded.
    """
    # Import deferred to avoid triggering model loading at collection time
    from fastapi.testclient import TestClient
    from harmonyspeech.endpoints.openai.api_server import app
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
