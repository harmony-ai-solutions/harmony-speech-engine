"""Multi-device placement, end to end (Phase 11-0).

Two layers:

* **Hermetic** (always runs): resolution, per-device pool accounting, and the
  runner registering its arena on the model's device even when the pool's
  default differs — the exact Phase 9 regression (``cuda`` vs ``cuda:0``) that
  disabled AR batching on every CUDA model.
* **Real** (opt-in): set ``HSE_CUDA_PLACEMENT=1`` plus the matching checkpoint
  env vars to load a real model on ``cuda:1`` and prove model + arena + runner
  grants all land there, and to run a two-device smoke (one tiny CT2 model on
  each card). These are opt-in because they load real weights; the default
  suite stays hermetic.
"""

import os

import pytest
import torch

from harmonyspeech.common.device import devices_match, resolve_device
from harmonyspeech.engine.ar_cache import ARCacheDescriptor, ARCachePool
from harmonyspeech.task_handler.ar_runner import ARBatchRunner

PLACEMENT = os.environ.get("HSE_CUDA_PLACEMENT") == "1"
CHATTERBOX_MODEL = os.environ.get("HSE_CHATTERBOX_PARITY_MODEL")
WHISPER_MODEL = os.environ.get("HSE_WHISPER_PARITY_MODEL")


def _two_cards() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:  # pragma: no cover - broken driver
        return False


def _expose_cudnn8_shim() -> None:
    """Test-side cuDNN 8 shim for CTranslate2 (never an engine change).

    The benchmark host needed a side-installed cuDNN 8 for CUDA 12; the engine's
    pre-check cannot see it because ``ctypes.CDLL`` by name does not search
    ``PATH`` on Python 3.8+. Expose it here for the CT2 smoke only.
    """
    base = os.environ.get("HSE_DLL_SHIM_BASE") or r"C:\Users\sge20\AppData\Local\Temp\opencode\cudnn8\nvidia"
    for sub in ("cudnn\\bin", "cublas\\bin", "cuda_nvrtc\\bin"):
        path = os.path.join(base, sub)
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except OSError:  # pragma: no cover - platform-specific
                pass


# ---------------------------------------------------------------------------
# Hermetic: resolution + per-device accounting + registration
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_unindexed_cuda_and_its_index_are_the_same_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert resolve_device("cuda") == torch.device("cuda:0")
    assert devices_match("cuda", "cuda:0")
    assert resolve_device("cuda:1") == torch.device("cuda:1")


@pytest.mark.integration
def test_pool_keeps_one_budget_per_device():
    descriptor = ARCacheDescriptor(
        model_key="a", layers=2, kv_heads=2, head_dim=4, dtype=torch.float32, cap_tokens=8, slots=1
    )
    slab = descriptor.slab_bytes(8)
    pool = ARCachePool(budget_bytes=2 * slab, device="cpu")
    pool.register(descriptor, device="cpu")
    pool.register(
        ARCacheDescriptor(
            model_key="b", layers=2, kv_heads=2, head_dim=4, dtype=torch.float32, cap_tokens=8, slots=1
        ),
        device="meta",
    )
    assert pool.remaining_bytes_for("cpu") == slab
    assert pool.remaining_bytes_for("meta") == slab


@pytest.mark.integration
def test_runner_arena_follows_the_model_device_not_the_pool_default():
    class _FakeT3:
        is_gpt = False
        device = torch.device("meta")
        cfg = type(
            "Cfg",
            (),
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "head_dim": 8,
                "hidden_size": 32,
                "num_attention_heads": 4,
            },
        )()

        def parameters(self):
            return iter([torch.zeros(1, dtype=torch.float16, device="meta")])

    slab = ARCacheDescriptor(
        model_key="m", layers=2, kv_heads=4, head_dim=8, dtype=torch.float16, cap_tokens=64
    ).slab_bytes(64)
    pool = ARCachePool(budget_bytes=4 * slab, device="cpu", default_cap_tokens=64)
    runner = ARBatchRunner(
        t3=_FakeT3(),
        pool=pool,
        model_key="m",
        model_config=type("Cfg", (), {"model_type": "ChatterboxTTS", "seed": 0, "max_batch_size": 2})(),
    )
    assert runner.register() is True
    assert runner.arena.device == torch.device("meta")


# ---------------------------------------------------------------------------
# Real placement (opt-in): model + arena + runner grants on cuda:1
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not PLACEMENT or not CHATTERBOX_MODEL or not os.path.isdir(CHATTERBOX_MODEL or ""),
    reason="set HSE_CUDA_PLACEMENT=1 and HSE_CHATTERBOX_PARITY_MODEL to a local checkpoint",
)
@pytest.mark.skipif(not _two_cards(), reason="needs two CUDA devices")
def test_real_chatterbox_on_cuda1_with_arena_and_grants():
    from harmonyspeech.common.config import DeviceConfig, ModelConfig
    from harmonyspeech.modeling.loader import get_model
    from harmonyspeech.task_handler.ar_runner import english_text_tokens

    config = ModelConfig(
        name="chatterbox-cuda1",
        model=CHATTERBOX_MODEL,
        model_type="ChatterboxTTS",
        max_batch_size=2,
        device_config=DeviceConfig("cuda:1"),
        dtype="float32",
    )
    model = get_model(config, config.device_config)

    # 1) The model really sits on the secondary card.
    assert torch.device(model.t3.device).index == 1, f"model on {model.t3.device}"

    # 2) The pool default is deliberately a *different* device (cuda:0). The
    #    runner must still register on cuda:1 instead of disabling itself.
    pool = ARCachePool(budget_bytes=4 * 1024**3, device="cuda:0", default_cap_tokens=768)
    runner = ARBatchRunner(
        t3=model.t3,
        pool=pool,
        model_key=config.name,
        model_config=config,
        model_type=config.model_type,
    )
    assert runner.register() is True, runner._registration_error
    assert torch.device(runner.arena.device).index == 1, f"arena on {runner.arena.device}"

    # 3) A real decode step grants slots on cuda:1 and advances rows.
    row = runner.begin(
        "cuda1-row",
        conds=model.conds,
        text_tokens=english_text_tokens(model, "Hello there."),
        seed=0,
        max_new_tokens=8,
    )
    assert row is not None
    for _ in range(12):
        outcome = runner.step(["cuda1-row"])[0]
        if outcome.finished:
            break
    assert pool.grants_total >= 2, "English needs two CFG slots"
    assert pool.stats()["devices"]["cuda:1"]["used_bytes"] > 0
    assert pool.stats()["devices"].get("cuda:0") is None, "nothing allocated on the default device"
    runner.pop("cuda1-row")


# ---------------------------------------------------------------------------
# Two-device smoke (opt-in): one tiny CT2 model per card
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not PLACEMENT or not WHISPER_MODEL or not os.path.isdir(WHISPER_MODEL or ""),
    reason="set HSE_CUDA_PLACEMENT=1 and HSE_WHISPER_PARITY_MODEL to a local CTranslate2 directory",
)
@pytest.mark.skipif(not _two_cards(), reason="needs two CUDA devices")
def test_two_device_smoke_one_tiny_whisper_per_card():
    import numpy as np
    from faster_whisper import WhisperModel

    _expose_cudnn8_shim()

    audio = np.zeros(16000, dtype=np.float32)
    for index in (0, 1):
        model = WhisperModel(WHISPER_MODEL, device="cuda", device_index=index, compute_type="float16")
        segments, info = model.transcribe(audio, vad_filter=False)
        list(segments)
        native_index = getattr(model.model, "device_index", None)
        if isinstance(native_index, (list, tuple)) and native_index:
            native_index = int(native_index[0])
        assert native_index == index, f"model asked for cuda:{index}, CTranslate2 reports {native_index}"
        assert info.duration == pytest.approx(1.0, abs=0.01)
