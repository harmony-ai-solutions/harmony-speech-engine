"""S3Gen autocast seam gates (Phase 11-2).

The experiment is CUDA-only and env-gated. These CPU tests pin the *branch*
behaviour (default off, value parsing, device policy, cast-back) without running
bf16 inference of the tiny S3Gen graph — the task's explicit constraint. The
GPU quality/throughput evidence lives in `docs/benchmarks/throughput.md` §Phase
11-2 and the `dtype:fp32-vs-autocast` harness axis.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from harmonyspeech.modeling.models.chatterbox import s3gen_precision as precision
from harmonyspeech.modeling.models.chatterbox.s3gen_precision import (
    S3GEN_AUTOCAST_ENV,
    S3GEN_PRECISION_FP16,
    S3GEN_PRECISION_FP32,
    autocast_enabled,
    effective_s3gen_precision,
    install_inference_autocast,
    normalize_s3gen_precision,
    s3gen_autocast,
    s3gen_autocast_dtype,
)


def test_normalize_aligns_aliases_to_the_dtype_vocabulary():
    """Phase 11-3 alignment: canonical spellings are float32/float16 (like
    ``dtype``); the pre-alignment fp32/fp16 remain runtime aliases, and
    anything unknown normalises to float32 (typo-safe)."""
    assert normalize_s3gen_precision("float16") == "float16"
    assert normalize_s3gen_precision("float32") == "float32"
    assert normalize_s3gen_precision("fp16") == "float16"
    assert normalize_s3gen_precision("FP16") == "float16"
    assert normalize_s3gen_precision("fp32") == "float32"
    assert normalize_s3gen_precision("bogus") == "float32"
    assert normalize_s3gen_precision(None) == "float32"


def test_install_none_preserves_the_shared_seam_value():
    """Phase 11-3 flip guard (embedding-role build): ``precision=None`` wraps
    the seam idempotently but must NOT write a value onto a SHARED S3Gen —
    the embedding role (no vocode) builds last in a backbone group and would
    otherwise clobber the vocode roles' precision back to float32."""
    s3gen = _StubS3Gen(dtype=torch.float32)
    install_inference_autocast(s3gen, precision="float16")
    install_inference_autocast(s3gen, precision=None)
    assert s3gen._hse_s3gen_precision == "float16"


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    """Every test starts with the gate unset and no memoised warnings."""
    monkeypatch.delenv(S3GEN_AUTOCAST_ENV, raising=False)
    monkeypatch.setattr(precision, "_WARNED", set())


# ---------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------
def test_gate_defaults_off():
    assert s3gen_autocast_dtype() is None
    assert s3gen_autocast_dtype({S3GEN_AUTOCAST_ENV: "off"}) is None
    assert s3gen_autocast_dtype({S3GEN_AUTOCAST_ENV: "  "}) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("BF16", torch.bfloat16),
        ("float16", torch.float16),
        ("fp16", torch.float16),
        (" half ", torch.float16),
    ],
)
def test_gate_parses_supported_values(value, expected):
    assert s3gen_autocast_dtype({S3GEN_AUTOCAST_ENV: value}) is expected


def test_gate_reads_the_process_environment(monkeypatch):
    monkeypatch.setenv(S3GEN_AUTOCAST_ENV, "bfloat16")
    assert s3gen_autocast_dtype() is torch.bfloat16


def test_unknown_value_is_off_and_warns_once():
    source = {S3GEN_AUTOCAST_ENV: "float8"}
    assert s3gen_autocast_dtype(source) is None
    assert s3gen_autocast_dtype(source) is None  # memoised: no second warning
    assert "float8" in precision._WARNED


# ---------------------------------------------------------------------------
# Device policy + context manager
# ---------------------------------------------------------------------------
def test_autocast_is_off_on_non_cuda_devices_even_when_requested(monkeypatch):
    monkeypatch.setenv(S3GEN_AUTOCAST_ENV, "bfloat16")
    assert autocast_enabled("cpu") is False
    assert autocast_enabled(torch.device("cpu")) is False


def test_cpu_context_is_a_no_op(monkeypatch):
    """Gate on + CPU: the context yields None and numerics stay fp32."""
    monkeypatch.setenv(S3GEN_AUTOCAST_ENV, "bfloat16")
    linear = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    with s3gen_autocast("cpu") as active:
        assert active is None
        out = linear(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, linear(x))


def test_context_engages_the_branch_when_the_device_check_allows_it(monkeypatch):
    """Drive the real autocast branch on CPU by stubbing only the device gate."""
    monkeypatch.setattr(precision, "_autocast_device_type", lambda device: "cpu")
    linear = nn.Linear(4, 4)
    x = torch.randn(2, 4)

    monkeypatch.setenv(S3GEN_AUTOCAST_ENV, "bfloat16")
    with s3gen_autocast("cpu") as active:
        assert active is torch.bfloat16
        assert torch.is_autocast_enabled("cpu")
        assert linear(x).dtype == torch.bfloat16
    assert not torch.is_autocast_enabled("cpu")

    monkeypatch.delenv(S3GEN_AUTOCAST_ENV)
    with s3gen_autocast("cpu") as active:
        assert active is None
        assert linear(x).dtype == torch.float32


def test_explicit_dtype_argument_wins_over_the_env(monkeypatch):
    monkeypatch.setattr(precision, "_autocast_device_type", lambda device: "cpu")
    monkeypatch.setenv(S3GEN_AUTOCAST_ENV, "bfloat16")
    with s3gen_autocast("cpu", dtype=torch.float16) as active:
        assert active is torch.float16


# ---------------------------------------------------------------------------
# The instance seam
# ---------------------------------------------------------------------------
class _StubS3Gen:
    """Minimal stand-in exposing what the seam touches."""

    def __init__(self, dtype=torch.float32):
        self.device = torch.device("cpu")
        self.output_dtype = dtype
        self.calls = 0
        self.autocast_seen: list[bool] = []

    def inference(self, speech_tokens):
        self.calls += 1
        self.autocast_seen.append(torch.is_autocast_enabled("cpu"))
        wav = torch.zeros(1, 8, dtype=self.output_dtype)
        sources = [torch.zeros(1, 8, dtype=self.output_dtype)]
        return wav, sources


def test_seam_is_a_pass_through_when_off():
    s3gen = _StubS3Gen()
    install_inference_autocast(s3gen)
    wav, sources = s3gen.inference(torch.zeros(1, 2, dtype=torch.long))
    assert s3gen._hse_autocast_seam is True
    assert s3gen.autocast_seen == [False]
    assert wav.dtype == torch.float32


def test_seam_casts_half_results_back_to_float32(monkeypatch):
    monkeypatch.setattr(precision, "_autocast_device_type", lambda device: "cpu")
    monkeypatch.setenv(S3GEN_AUTOCAST_ENV, "bfloat16")
    s3gen = _StubS3Gen(dtype=torch.bfloat16)
    install_inference_autocast(s3gen)

    wav, sources = s3gen.inference(torch.zeros(1, 2, dtype=torch.long))

    assert s3gen.autocast_seen == [True]
    assert wav.dtype == torch.float32
    assert sources[0].dtype == torch.float32


def test_seam_install_is_idempotent():
    s3gen = _StubS3Gen()
    original = s3gen.inference
    install_inference_autocast(s3gen)
    installed = s3gen.inference
    install_inference_autocast(s3gen)
    assert s3gen.inference is installed
    assert s3gen.inference is not original


def test_seam_tolerates_objects_without_inference():
    marker = nn.Linear(2, 2)
    assert install_inference_autocast(marker) is marker
    assert install_inference_autocast(None) is None
    assert not getattr(marker, "_hse_autocast_seam", False)


def test_loader_installs_the_seam():
    from harmonyspeech.modeling.models.chatterbox.chatterbox import apply_model_dtype

    model = SimpleNamespace(
        t3=nn.Linear(2, 2),
        s3gen=_StubS3Gen(),
        ve=nn.Linear(2, 2),
        device="cpu",
    )
    apply_model_dtype(model, torch.float32)
    assert model.s3gen._hse_autocast_seam is True


# ---------------------------------------------------------------------------
# Phase 11-3: the config knob (config vs env precedence)
# ---------------------------------------------------------------------------
def test_config_fp32_is_the_default_and_resolves_to_off():
    assert s3gen_autocast_dtype(configured=S3GEN_PRECISION_FP32) is None
    assert s3gen_autocast_dtype(configured=None) is None
    assert effective_s3gen_precision(S3GEN_PRECISION_FP32) == "float32"


def test_config_fp16_resolves_when_the_env_is_absent():
    assert s3gen_autocast_dtype(configured=S3GEN_PRECISION_FP16) is torch.float16
    assert effective_s3gen_precision(S3GEN_PRECISION_FP16) == "float16"


def test_env_overrides_config():
    """The env escape hatch wins in both directions, including an explicit off."""
    assert (
        s3gen_autocast_dtype({S3GEN_AUTOCAST_ENV: "bfloat16"}, configured="float16")
        is torch.bfloat16
    )
    assert s3gen_autocast_dtype({S3GEN_AUTOCAST_ENV: "off"}, configured="float16") is None
    assert s3gen_autocast_dtype({S3GEN_AUTOCAST_ENV: "float16"}, configured="float32") is torch.float16
    assert effective_s3gen_precision("float16", environ={S3GEN_AUTOCAST_ENV: "off"}) == "float32"
    assert effective_s3gen_precision("float32", environ={S3GEN_AUTOCAST_ENV: "bfloat16"}) == "bfloat16"


def test_install_stores_the_configured_precision_and_drives_the_seam(monkeypatch):
    """The fp16 config path engages autocast without any env variable."""
    monkeypatch.setattr(precision, "_autocast_device_type", lambda device: "cpu")
    s3gen = _StubS3Gen(dtype=torch.float16)
    install_inference_autocast(s3gen, precision="float16")

    assert s3gen._hse_s3gen_precision == "float16"
    wav, sources = s3gen.inference(torch.zeros(1, 2, dtype=torch.long))
    assert s3gen.autocast_seen == [True]
    assert wav.dtype == torch.float32  # cast back for the fp32 artifact path
    assert sources[0].dtype == torch.float32


def test_install_defaults_to_fp32_and_is_bit_identical(monkeypatch):
    """Default install (no config value) leaves the seam off."""
    s3gen = _StubS3Gen(dtype=torch.float32)
    install_inference_autocast(s3gen)
    assert getattr(s3gen, "_hse_s3gen_precision", S3GEN_PRECISION_FP32) == "float32"

    wav, sources = s3gen.inference(torch.zeros(1, 2, dtype=torch.long))
    assert s3gen.autocast_seen == [False]
    assert wav.dtype == torch.float32


def test_install_can_update_precision_after_the_seam_exists():
    s3gen = _StubS3Gen()
    install_inference_autocast(s3gen, precision="float32")
    install_inference_autocast(s3gen, precision="float16")
    assert s3gen._hse_s3gen_precision == "float16"


def test_context_precision_argument_resolves_the_config(monkeypatch):
    monkeypatch.setattr(precision, "_autocast_device_type", lambda device: "cpu")
    with s3gen_autocast("cpu", precision="float16") as active:
        assert active is torch.float16
    with s3gen_autocast("cpu", precision="float32") as active:
        assert active is None


def test_autocast_enabled_honors_precision(monkeypatch):
    monkeypatch.setattr(precision, "_autocast_device_type", lambda device: "cpu")
    assert autocast_enabled("cpu", precision="float16") is True
    assert autocast_enabled("cpu", precision="float32") is False


def test_batched_path_threads_the_instance_precision(monkeypatch):
    """``inference_batched`` passes the per-model value to ``_flow_batched``."""
    from harmonyspeech.modeling.models.chatterbox import s3gen_batch

    seen = {}

    def fake_flow(flow, rows, generators, *, n_timesteps, precision=None):
        seen["precision"] = precision
        return [torch.zeros(1, 80, 4)]

    monkeypatch.setattr(s3gen_batch, "supports_batched_s3gen", lambda s3gen: True)
    monkeypatch.setattr(s3gen_batch, "_flow_batched", fake_flow)
    monkeypatch.setattr(s3gen_batch, "_hift_batched", lambda mel2wav, mels, gens: [torch.zeros(1, 8)])

    s3gen = _StubS3Gen()
    s3gen.flow = nn.Linear(2, 2)
    s3gen.mel2wav = nn.Linear(2, 2)
    s3gen.meanflow = False
    install_inference_autocast(s3gen, precision="float16")

    rows = [
        s3gen_batch.S3GenRow(tokens=torch.zeros(4, dtype=torch.long), ref_dict={}, seed=1),
        s3gen_batch.S3GenRow(tokens=torch.zeros(4, dtype=torch.long), ref_dict={}, seed=2),
    ]
    s3gen_batch.inference_batched(s3gen, rows)
    assert seen["precision"] == "float16"
