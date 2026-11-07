"""Real-model seeded parity gate for the Chatterbox AR runner (Phase 9-6).

The phase's ``batching:1-vs-N`` gate for the AR family. It needs a local
Chatterbox checkpoint (no weights are downloaded here) and is skipped unless
``HSE_CHATTERBOX_PARITY_MODEL`` (English) / ``HSE_CHATTERBOX_TURBO_PARITY_MODEL``
(Turbo) point at one, e.g.::

    set HSE_CHATTERBOX_PARITY_MODEL=C:\\models\\chatterbox
    pytest tests/e2e/tts/test_chatterbox_ar_parity.py -m e2e -q

What is asserted (9-3's seeded contract):

* a row's token stream is identical whether it decodes alone or alongside
  another request (per-row ``torch.Generator``, never one batched draw);
* the two rows in a batch do not collapse into one stream (the batch really
  carries two independent requests);
* the decode never mutates ``model.conds`` (Phase 0-2's identity invariant is
  superseded on this path, not broken).

The vocode-noise parity (a row's audio identical alone or batched) is the
Phase 9-5 unit gate; this harness stops at the token stream, which is what the
AR batching decision changes.
"""

import os

import pytest
import torch

MODEL_PATH = os.environ.get("HSE_CHATTERBOX_PARITY_MODEL")
TURBO_MODEL_PATH = os.environ.get("HSE_CHATTERBOX_TURBO_PARITY_MODEL")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        not MODEL_PATH or not os.path.isdir(MODEL_PATH),
        reason="set HSE_CHATTERBOX_PARITY_MODEL to a local Chatterbox checkpoint",
    ),
]


def _model_config(model_path: str, model_type: str):
    from harmonyspeech.common.config import DeviceConfig, ModelConfig

    return ModelConfig(
        name="chatterbox-parity",
        model=model_path,
        model_type=model_type,
        max_batch_size=4,
        device_config=DeviceConfig("auto"),
        dtype="float32",
    )


def _load(model_path: str, model_type: str):
    from harmonyspeech.modeling.loader import get_model

    config = _model_config(model_path, model_type)
    return config, get_model(config, config.device_config)


def _runner(config, model):
    from harmonyspeech.engine.ar_cache import ARCachePool
    from harmonyspeech.task_handler.ar_runner import ARBatchRunner

    pool = ARCachePool(budget_bytes=4 * 1024**3, device=model.t3.device, default_cap_tokens=768)
    runner = ARBatchRunner(
        t3=model.t3,
        pool=pool,
        model_key=config.name,
        model_config=config,
        model_type=config.model_type,
    )
    assert runner.register(), "the parity pool must fit the gate's 4-request arena"
    return runner


def _decode(runner, model, text: str, request_id: str, *, turbo: bool, seed: int = 0, max_new_tokens: int = 64):
    from harmonyspeech.task_handler.ar_runner import english_text_tokens, turbo_text_tokens

    tokens = (turbo_text_tokens if turbo else english_text_tokens)(model, text)
    row = runner.begin(
        request_id,
        conds=model.conds,
        text_tokens=tokens,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    assert row is not None, f"runner refused the row for {request_id}"
    for _ in range(max_new_tokens + 8):
        outcome = runner.step([request_id])[0]
        if outcome.finished:
            break
    else:  # pragma: no cover - a real model that never stops
        raise AssertionError(f"{request_id} never finished within the token cap")
    stream = runner.final_tokens(request_id)
    runner.pop(request_id)
    return stream


def _assert_row_is_batch_invariant(model_path: str, model_type: str, *, turbo: bool):
    from harmonyspeech.task_handler.ar_runner import english_text_tokens, turbo_text_tokens

    text_tokens_for = turbo_text_tokens if turbo else english_text_tokens
    config, model = _load(model_path, model_type)
    baseline_conds = model.conds
    runner = _runner(config, model)

    alone = _decode(runner, model, "Hello there.", "alone", turbo=turbo)
    # The decode must not leave request state on the model (Phase 0-2 identity).
    assert model.conds is baseline_conds

    runner.begin(
        "row-a",
        conds=model.conds,
        text_tokens=text_tokens_for(model, "Hello there."),
        seed=0,
        max_new_tokens=64,
    )
    runner.begin(
        "row-b",
        conds=model.conds,
        text_tokens=text_tokens_for(
            model, "This is a much longer sentence that should decode for more tokens."
        ),
        seed=0,
        max_new_tokens=64,
    )
    for _ in range(72):
        outcomes = runner.step(["row-a", "row-b"])
        if all(outcome.finished for outcome in outcomes):
            break
    batched_a = runner.final_tokens("row-a")
    batched_b = runner.final_tokens("row-b")
    runner.pop("row-a")
    runner.pop("row-b")

    assert torch.equal(alone, batched_a), "a row's tokens changed when it shared a batch"
    assert not torch.equal(batched_a, batched_b), "two requests collapsed into one token stream"
    assert model.conds is baseline_conds


def test_english_row_tokens_are_batch_invariant():
    _assert_row_is_batch_invariant(MODEL_PATH, "ChatterboxTTS", turbo=False)


@pytest.mark.skipif(
    not TURBO_MODEL_PATH or not os.path.isdir(TURBO_MODEL_PATH),
    reason="set HSE_CHATTERBOX_TURBO_PARITY_MODEL to a local Chatterbox Turbo checkpoint",
)
def test_turbo_row_tokens_are_batch_invariant():
    _assert_row_is_batch_invariant(TURBO_MODEL_PATH, "ChatterboxTurboTTS", turbo=True)
