"""Fixtures for wire-contract (Phase 1) unit tests."""

import pytest

from tests.wire_contract_support import WireApp


@pytest.fixture
def wire_client():
    """App/TestClient with a boundary-decoding FakeEngine and 50 MiB upload limit."""
    with WireApp(decode_inputs=True) as wire:
        yield wire


@pytest.fixture
def wire_client_small_limit():
    """App/TestClient with a 4 KiB upload limit (for 413 tests)."""
    with WireApp(decode_inputs=True, max_upload_bytes=4096) as wire:
        yield wire
