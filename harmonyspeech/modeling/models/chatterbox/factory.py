"""Chatterbox backbone factory and role assembly (Phase 4-4).

Why this module exists
----------------------
The shipped catalogue configures five Chatterbox executors (TTS, Turbo,
Multilingual, VC, Embedding). Historically each one called upstream
``from_pretrained`` and therefore constructed — and held — its own copy of the
shared S3Gen (≈1.0 GiB fp32) and VoiceEncoder (≈6 MiB). This module builds the
shared components **once per ``(repo, revision, component, dtype)``** through the
engine's :class:`~harmonyspeech.modeling.weight_registry.WeightRegistry` and
grafts the same module objects into every role that opts in with
``backbone_group``.

Verified upstream topology (``chatterbox-tts==0.1.6``)
------------------------------------------------------
``chatterbox/tts.py`` (``ChatterboxTTS.from_local``)::

    ve = VoiceEncoder(); ve.load_state_dict(load_file("ve.safetensors"))
    t3 = T3(); t3.load_state_dict(load_file("t3_cfg.safetensors"))
    s3gen = S3Gen(); s3gen.load_state_dict(load_file("s3gen.safetensors"), strict=False)
    tokenizer = EnTokenizer("tokenizer.json"); conds = Conditionals.load("conds.pt")

``chatterbox/mtl_tts.py`` (``ChatterboxMultilingualTTS.from_local``)::

    ve = VoiceEncoder(); ve.load_state_dict(torch.load("ve.pt", weights_only=True))
    t3 = T3(T3Config.multilingual()); t3.load_state_dict(load_file("t3_mtl23ls_v2.safetensors"))
    s3gen = S3Gen(); s3gen.load_state_dict(torch.load("s3gen.pt", weights_only=True))
    tokenizer = MTLTokenizer("grapheme_mtl_merged_expanded_v1.json")

``chatterbox/vc.py`` (``ChatterboxVC.from_local``)::

    s3gen = S3Gen(); s3gen.load_state_dict(load_file("s3gen.safetensors"), strict=False)
    ref_dict = torch.load("conds.pt")["gen"]

``chatterbox/tts_turbo.py`` (``ChatterboxTurboTTS.from_local``)::

    s3gen = S3Gen(meanflow=True); s3gen.load_state_dict(load_file("s3gen_meanflow.safetensors"), strict=True)
    t3 = T3(<turbo hp>); del t3.tfmr.wte

Role → component matrix (task 4-4)::

    Role                       T3               S3Gen   VE
    ChatterboxTTS              own (t3_cfg)     shared  shared
    ChatterboxEmbedding        shared with TTS  shared  shared
    ChatterboxVC               -                shared  shared
    ChatterboxMultilingualTTS  own (t3_mtl*)    shared* shared*
    ChatterboxTurboTTS         own (unshared)   own     own

``*`` Multilingual joins the base group only because the cached checkpoints are
bit-identical: ``s3gen.pt`` vs ``s3gen.safetensors`` share all 2489 tensors
bit-for-bit (the ``.pt`` file only adds the non-persistent ``tokenizer.window``
buffer), and ``ve.pt`` == ``ve.safetensors`` exactly. Evidence recorded in
``.current_work/engine-modernization/4-4-ChatterboxBackboneSharing.md``. Turbo
is deliberately ungrouped: it synthesises from ``s3gen_meanflow.safetensors``
(a different model), even though its dormant ``s3gen.safetensors`` copy is
identical to the base one.

Per-request state (``conds`` / ``ref_dict``) is created per role and stays on
the wrapper instance — never on the shared modules (Phase 0-2 invariant).

AR decode hygiene (Phase 9-1)
-----------------------------
The TTS role's T3 is instantiated as
:class:`~harmonyspeech.modeling.models.chatterbox.t3_ar.ChatterboxT3AR`, an
upstream ``T3`` subclass that only overrides the inference path (no per-step
``output_attentions`` when no analyzer is attached, no per-token ``tqdm``). The
embedding role shares that same object; the multilingual role keeps the plain
class (its analyzer needs attention) and turbo keeps its own structurally
different loop (``T3.inference_turbo``) until 9-3.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import EnTokenizer, MTLTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.mtl_tts import Conditionals as MTL_Conditionals
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts import Conditionals as TTS_Conditionals
from chatterbox.vc import ChatterboxVC
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from loguru import logger
from safetensors.torch import load_file

from harmonyspeech.modeling.models.chatterbox.chatterbox import (
    _resolve_load_dtype,
    apply_model_dtype,
)
from harmonyspeech.modeling.models.chatterbox.t3_ar import ChatterboxT3AR
from harmonyspeech.modeling.weight_registry import WeightRegistry

# Default HF repo for the base + multilingual Chatterbox checkpoints (mirrors
# `chatterbox/tts.py`/`mtl_tts.py` REPO_ID).
BASE_REPO = "ResembleAI/chatterbox"

#: HF repo ids are case-sensitive on disk (``models--ResembleAI--chatterbox``),
#: and the shipped config spells them lowercase. Normalise before resolving
#: cache paths or registry keys so casing never splits a shared group.
_CANONICAL_REPOS = {
    "resembleai/chatterbox": BASE_REPO,
    "resemble-ai/chatterbox": BASE_REPO,
    "resembleai/chatterbox-turbo": "ResembleAI/chatterbox-turbo",
    "resemble-ai/chatterbox-turbo": "ResembleAI/chatterbox-turbo",
}


def canonical_repo(repo: str) -> str:
    """Return the on-disk (case-correct) repo id for a configured model name."""
    return _CANONICAL_REPOS.get(str(repo).strip().lower(), repo)

# Component names used in registry keys. Keep them stable — they show up in
# /metrics labels and benchmark tables.
COMPONENT_S3GEN = "s3gen"
COMPONENT_VE = "ve"
COMPONENT_T3 = "t3"

# Roles
ROLE_TTS = "tts"
ROLE_EMBEDDING = "embedding"
ROLE_VC = "vc"
ROLE_MULTILINGUAL = "multilingual"
ROLE_TURBO = "turbo"

#: Roles whose T3 decodes with HSE's cleaned AR loop (Phase 9-1). The embedding
#: role shares the TTS T3 object, so it is covered by loading that role as TTS.
_AR_T3_ROLES = frozenset({ROLE_TTS})


def _t3_class_for_role(role: str) -> type[T3]:
    """Return the T3 implementation to instantiate for ``role``.

    Only the English TTS decode path gets :class:`ChatterboxT3AR`; multilingual
    keeps the upstream class (per-request analyzer state on the shared Llama) and
    turbo keeps ``T3.inference_turbo`` (a different loop, deferred to 9-3).
    """
    return ChatterboxT3AR if role in _AR_T3_ROLES else T3


# File names per role family (verified against chatterbox-tts 0.1.6).
_FILES = {
    ROLE_TTS: {
        "ve": "ve.safetensors",
        "t3": "t3_cfg.safetensors",
        "s3gen": "s3gen.safetensors",
        "tokenizer": "tokenizer.json",
        "conds": "conds.pt",
    },
    ROLE_VC: {
        "ve": "ve.safetensors",
        "s3gen": "s3gen.safetensors",
        "conds": "conds.pt",
    },
    ROLE_MULTILINGUAL: {
        "ve": "ve.pt",
        "t3": "t3_mtl23ls_v2.safetensors",
        "s3gen": "s3gen.pt",
        "tokenizer": "grapheme_mtl_merged_expanded_v1.json",
        "conds": "conds.pt",
    },
    ROLE_TURBO: {
        "ve": "ve.safetensors",
        "t3": "t3_turbo_v1.safetensors",
        "s3gen": "s3gen_meanflow.safetensors",
        "conds": "conds.pt",
    },
}

# Turbo's own T3 hyper-parameters (copied from tts_turbo.py `from_local`).
_TURBO_HP = {
    "text_tokens_dict_size": 50276,
    "speech_tokens_dict_size": 6563,
    "input_pos_emb": None,
    "speech_cond_prompt_len": 375,
    "use_perceiver_resampler": False,
    "emotion_adv": False,
}
_TURBO_LLAMA_CONFIG = "GPT2_medium"


@dataclass
class ChatterboxBackbone:
    """Shared backbone components for one Chatterbox checkpoint family.

    Every field is a *shared, read-only* module (or ``None`` when the role does
    not need it). :meth:`factory` returns this from the registry; roles graft
    these references into their own wrapper instance.
    """

    repo: str
    revision: str | None
    device: str
    dtype: torch.dtype
    s3gen: nn.Module
    ve: nn.Module
    t3: nn.Module | None = None
    t3_dtype: torch.dtype = torch.float32


# ---------------------------------------------------------------------------
# File resolution (local cache only — never triggers a download)
# ---------------------------------------------------------------------------
def _ckpt_file(repo: str, filename: str, revision: str | None) -> str:
    """Resolve a Chatterbox checkpoint file from the local HF cache.

    Chatterbox checkpoints are loaded eagerly at boot, so a missing file is a
    configuration error: fail fast with an actionable message instead of
    silently downloading gigabytes mid-boot.
    """
    import os

    repo = canonical_repo(repo)
    local = os.path.join(repo, filename)
    if os.path.isfile(local):
        return local
    try:
        return hf_hub_download(repo_id=repo, revision=revision, filename=filename, local_files_only=True)
    except LocalEntryNotFoundError as exc:  # pragma: no cover - depends on local cache
        raise FileNotFoundError(
            f"Chatterbox checkpoint file '{filename}' for repo '{repo}' is not in the local "
            f"HuggingFace cache. Pre-download the model (e.g. `huggingface-cli download {repo}`) "
            f"before booting with backbone_group enabled."
        ) from exc


def _ckpt_dir(repo: str, revision: str | None) -> str:
    """Return the local directory that holds a complete Chatterbox checkpoint."""
    import os

    return os.path.dirname(_ckpt_file(repo, "conds.pt", revision))


# ---------------------------------------------------------------------------
# Component loaders (mirror the upstream from_local implementations)
# ---------------------------------------------------------------------------
def _map_location(device: str):
    return torch.device("cpu") if device in ("cpu", "mps") else None


def load_s3gen(repo: str, revision: str | None, device: str, *, role: str) -> S3Gen:
    """Load an S3Gen (as upstream does for ``role``), cast to float32."""
    ckpt = _ckpt_dir(repo, revision)
    meanflow = role == ROLE_TURBO
    s3gen = S3Gen(meanflow=meanflow)
    if meanflow:
        state = load_file(f"{ckpt}/{_FILES[ROLE_TURBO]['s3gen']}")
    elif role == ROLE_MULTILINGUAL:
        state = torch.load(
            f"{ckpt}/{_FILES[ROLE_MULTILINGUAL]['s3gen']}",
            map_location=_map_location(device),
            weights_only=True,
        )
    else:
        state = load_file(f"{ckpt}/{_FILES[ROLE_TTS]['s3gen']}")
    # strict=False mirrors upstream for the standard file; the multilingual
    # checkpoint additionally carries the non-persistent `tokenizer.window`
    # buffer, which the module already initialises in __init__.
    s3gen.load_state_dict(state, strict=False)
    s3gen = s3gen.to(device=device, dtype=torch.float32).eval()
    s3gen.requires_grad_(False)
    return s3gen


def load_ve(repo: str, revision: str | None, device: str, *, role: str) -> VoiceEncoder:
    """Load the VoiceEncoder (float32 — its LSTM path requires it)."""
    ckpt = _ckpt_dir(repo, revision)
    ve = VoiceEncoder()
    if role == ROLE_MULTILINGUAL:
        state = torch.load(
            f"{ckpt}/{_FILES[ROLE_MULTILINGUAL]['ve']}",
            map_location=_map_location(device),
            weights_only=True,
        )
    else:
        state = load_file(f"{ckpt}/{_FILES[ROLE_TTS]['ve']}")
    ve.load_state_dict(state)
    ve = ve.to(device=device, dtype=torch.float32).eval()
    ve.requires_grad_(False)
    return ve


def load_t3(repo: str, revision: str | None, device: str, dtype: torch.dtype, *, role: str) -> T3:
    """Load the T3 autoregressive backbone, cast to ``dtype``.

    The class comes from :func:`_t3_class_for_role`, so the English TTS role
    (and the embedding role that shares it) gets HSE's cleaned AR loop
    (Phase 9-1) while multilingual/turbo stay on the upstream class.
    """
    ckpt = _ckpt_dir(repo, revision)
    t3_cls = _t3_class_for_role(role)
    if role == ROLE_MULTILINGUAL:
        t3 = t3_cls(T3Config.multilingual())
        state = load_file(f"{ckpt}/{_FILES[ROLE_MULTILINGUAL]['t3']}")
    elif role == ROLE_TURBO:
        # Upstream `T3Config.__init__` only accepts `text_tokens_dict_size`; the
        # other turbo hyper-parameters are attributes set after construction
        # (mirrors `chatterbox/tts_turbo.py:149-155`). The old
        # `T3Config(**_TURBO_HP)` call raised TypeError if this branch was ever
        # reached (Phase 9-1 Deviation 6; fixed here because 9-3 owns turbo).
        hp = T3Config(text_tokens_dict_size=_TURBO_HP["text_tokens_dict_size"])
        hp.llama_config_name = _TURBO_LLAMA_CONFIG
        hp.speech_tokens_dict_size = _TURBO_HP["speech_tokens_dict_size"]
        hp.input_pos_emb = _TURBO_HP["input_pos_emb"]
        hp.speech_cond_prompt_len = _TURBO_HP["speech_cond_prompt_len"]
        hp.use_perceiver_resampler = _TURBO_HP["use_perceiver_resampler"]
        hp.emotion_adv = _TURBO_HP["emotion_adv"]
        t3 = t3_cls(hp)
        state = load_file(f"{ckpt}/{_FILES[ROLE_TURBO]['t3']}")
    else:
        t3 = t3_cls()
        state = load_file(f"{ckpt}/{_FILES[ROLE_TTS]['t3']}")
    if "model" in state:
        state = state["model"][0]
    t3.load_state_dict(state)
    if role == ROLE_TURBO:
        del t3.tfmr.wte
    t3 = t3.to(device=device, dtype=dtype).eval()
    t3.requires_grad_(False)
    return t3


def _load_tokenizer(role: str, ckpt: str):
    if role == ROLE_MULTILINGUAL:
        return MTLTokenizer(f"{ckpt}/{_FILES[ROLE_MULTILINGUAL]['tokenizer']}")
    if role == ROLE_TURBO:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    return EnTokenizer(f"{ckpt}/{_FILES[ROLE_TTS]['tokenizer']}")


def _detach_tensors(mapping: dict) -> dict:
    """Return a copy of a state dict with every tensor detached from autograd.

    Chatterbox's bundled ``conds.pt`` carries one tensor with
    ``requires_grad=True``; moving it to CUDA therefore produces a non-leaf
    tensor, which cannot be deep-copied by the Phase 0-2 baseline snapshot.
    Detaching at load keeps conditionals plain inference data.
    """
    out = {}
    for key, value in (mapping or {}).items():
        out[key] = value.detach() if hasattr(value, "detach") and hasattr(value, "is_leaf") else value
    return out


def _load_conditionals(repo: str, revision: str | None, device: str, *, role: str):
    """Load this role's own conditionals/ref_dict (per-executor state)."""
    conds_path = f"{_ckpt_dir(repo, revision)}/{_FILES[_FILES_ROLE[role]]['conds']}"
    loader = MTL_Conditionals if role == ROLE_MULTILINGUAL else TTS_Conditionals
    load = loader.load if role == ROLE_MULTILINGUAL else lambda p: loader.load(p, map_location=_map_location(device))
    conds = load(conds_path)
    conds = conds.to(device)
    # Detach autograd state so per-request conditionals stay plain leaves.
    if getattr(conds, "t3", None) is not None:
        conds.t3 = type(conds.t3)(**_detach_tensors(dict(vars(conds.t3))))
    if getattr(conds, "gen", None) is not None:
        conds.gen = _detach_tensors(dict(conds.gen))
    return conds


_FILES_ROLE = {
    ROLE_TTS: ROLE_TTS,
    ROLE_EMBEDDING: ROLE_TTS,
    ROLE_VC: ROLE_VC,
    ROLE_MULTILINGUAL: ROLE_MULTILINGUAL,
    ROLE_TURBO: ROLE_TURBO,
}


# ---------------------------------------------------------------------------
# Backbone construction / registry keys
# ---------------------------------------------------------------------------
def _key(repo: str, revision: str | None, component: str, dtype: torch.dtype) -> tuple:
    return (repo, revision, component, dtype)


def _has_standard_files(repo: str, revision: str | None) -> bool:
    try:
        _ckpt_file(repo, "s3gen.safetensors", revision)
        _ckpt_file(repo, "ve.safetensors", revision)
    except FileNotFoundError:
        return False
    return True


def _component_key_name(component: str, load_role: str) -> str:
    """Component key suffix: the file set differs per checkpoint family.

    ``s3gen``/``ve`` are loaded from different files for the multilingual
    (``.pt``) and turbo (meanflow) checkpoints, so they must not collide with
    the standard ``.safetensors`` keys when they are *not* sharing.
    """
    if load_role == ROLE_TURBO:
        return f"{component}:turbo"
    if load_role == ROLE_MULTILINGUAL:
        return f"{component}:mtl"
    return component


def build_backbone(
    repo: str,
    revision: str | None,
    device: str,
    dtype: torch.dtype,
    registry: WeightRegistry | None,
    *,
    role: str,
    share_t3: bool = False,
) -> ChatterboxBackbone:
    """Load (or reuse) the shared backbone set for ``(repo, revision, dtype)``.

    The registry key namespace is the *checkpoint* family, not the role: a
    multilingual role joining the base group resolves the base component keys
    (bit-identical checkpoints, see module docstring).
    """
    device = str(device)
    t3_dtype = _resolve_load_dtype(device, dtype)

    # Multilingual may transparently use the base component keys when the base
    # checkpoint files exist locally — that is what the hash evidence allows.
    base_role = role
    if role == ROLE_MULTILINGUAL and _has_standard_files(repo, revision):
        base_role = ROLE_TTS

    def _acquire(component: str, factory, *, comp_dtype: torch.dtype):
        key = _key(repo, revision, component, comp_dtype)
        if registry is None:
            return factory()
        return registry.acquire(key, factory)

    s3gen = _acquire(
        _component_key_name(COMPONENT_S3GEN, base_role),
        lambda: load_s3gen(repo, revision, device, role=base_role),
        comp_dtype=torch.float32,
    )
    ve = _acquire(
        _component_key_name(COMPONENT_VE, base_role),
        lambda: load_ve(repo, revision, device, role=base_role),
        comp_dtype=torch.float32,
    )

    t3 = None
    if role in (ROLE_TTS, ROLE_EMBEDDING, ROLE_MULTILINGUAL, ROLE_TURBO):
        # The embedding role never synthesizes, so it shares the TTS backbone's
        # T3 (stateless for inference) instead of loading a second ~1 GiB copy.
        t3_role = ROLE_TTS if role == ROLE_EMBEDDING else role
        key_suffix = t3_role if t3_role != ROLE_TTS else ""
        t3 = _acquire(
            f"{COMPONENT_T3}{':' + key_suffix if key_suffix else ''}",
            lambda: load_t3(repo, revision, device, t3_dtype, role=t3_role),
            comp_dtype=t3_dtype,
        )

    return ChatterboxBackbone(
        repo=repo,
        revision=revision,
        device=device,
        dtype=t3_dtype,
        s3gen=s3gen,
        ve=ve,
        t3=t3,
        t3_dtype=t3_dtype,
    )


# ---------------------------------------------------------------------------
# Role assembly
# ---------------------------------------------------------------------------
def assemble_role(
    role: str,
    backbone: ChatterboxBackbone,
    *,
    repo: str,
    revision: str | None,
    s3gen_precision: str = "fp32",
) -> object:
    """Construct a role wrapper around the shared backbone references.

    The returned object is a *distinct* upstream model instance per role, so
    ``conds`` / ``ref_dict`` remain per-executor (Phase 0-2). Only S3Gen and the
    VoiceEncoder (and, for the embedding role, T3) are shared objects.
    """
    device = backbone.device
    if role in (ROLE_TTS, ROLE_EMBEDDING):
        model = ChatterboxTTS(
            t3=backbone.t3,
            s3gen=backbone.s3gen,
            ve=backbone.ve,
            tokenizer=_load_tokenizer(ROLE_TTS, _ckpt_dir(repo, revision)),
            device=device,
            conds=_load_conditionals(repo, revision, device, role=ROLE_TTS),
        )
        apply_model_dtype(model, backbone.dtype, s3gen_precision)
        return model

    if role == ROLE_VC:
        conds_path = f"{_ckpt_dir(repo, revision)}/{_FILES[ROLE_VC]['conds']}"
        ref_dict = torch.load(conds_path, map_location=_map_location(device), weights_only=True)["gen"]
        model = ChatterboxVC(s3gen=backbone.s3gen, device=device, ref_dict=ref_dict)
        apply_model_dtype(model, backbone.dtype, s3gen_precision)
        return model

    if role == ROLE_MULTILINGUAL:
        model = ChatterboxMultilingualTTS(
            t3=backbone.t3,
            s3gen=backbone.s3gen,
            ve=backbone.ve,
            tokenizer=_load_tokenizer(ROLE_MULTILINGUAL, _ckpt_dir(repo, revision)),
            device=device,
            conds=_load_conditionals(repo, revision, device, role=ROLE_MULTILINGUAL),
        )
        if hasattr(model.t3, "tfmr") and hasattr(model.t3.tfmr, "config"):
            model.t3.tfmr.config._attn_implementation = "eager"
        apply_model_dtype(model, backbone.dtype, s3gen_precision)
        return model

    if role == ROLE_TURBO:
        # Turbo is kept on its own (unshared) path; see loader.py.
        raise ValueError("Turbo roles are not assembled by the sharing factory")

    raise ValueError(f"Unknown Chatterbox role '{role}'")


# ---------------------------------------------------------------------------
# Public entry point used by the loader
# ---------------------------------------------------------------------------
_ROLE_BY_MODEL_TYPE = {
    "ChatterboxTTS": ROLE_TTS,
    "ChatterboxEmbedding": ROLE_EMBEDDING,
    "ChatterboxVC": ROLE_VC,
    "ChatterboxMultilingualTTS": ROLE_MULTILINGUAL,
    "ChatterboxTurboTTS": ROLE_TURBO,
}


def role_for_model_type(model_type: str, model_name: str = "") -> str:
    """Map a config ``model_type`` (and optional model name) to a role."""
    if model_type == "ChatterboxEmbedding":
        lowered = (model_name or "").lower()
        if "turbo" in lowered:
            return ROLE_TURBO
        if "multilingual" in lowered:
            return ROLE_MULTILINGUAL
        return ROLE_EMBEDDING
    try:
        return _ROLE_BY_MODEL_TYPE[model_type]
    except KeyError as exc:
        raise ValueError(f"'{model_type}' is not a Chatterbox model type") from exc


def build_role_model(model_config, device, registry: WeightRegistry | None):
    """Build a Chatterbox role honoring ``model_config.backbone_group``.

    Returns ``None`` when the config does not opt in to sharing, so the loader
    can fall back to the historical per-config ``from_pretrained`` path.
    """
    if not getattr(model_config, "backbone_group", None):
        return None
    model_type = getattr(model_config, "model_type", "")
    role = role_for_model_type(model_type, getattr(model_config, "model", ""))
    if role == ROLE_TURBO:
        logger.info(
            "Chatterbox Turbo stays ungrouped (own meanflow S3Gen checkpoint) — "
            f"ignoring backbone_group='{model_config.backbone_group}'"
        )
        return None

    repo = canonical_repo(getattr(model_config, "model", None) or BASE_REPO)
    revision = getattr(model_config, "revision", None)
    dtype = getattr(model_config, "dtype", torch.float32)
    device = str(device)

    backbone = build_backbone(repo, revision, device, dtype, registry, role=role)
    model = assemble_role(
        role,
        backbone,
        repo=repo,
        revision=revision,
        s3gen_precision=getattr(model_config, "s3gen_precision", "fp32"),
    )
    logger.info(
        f"Assembled Chatterbox role '{role}' from backbone group "
        f"'{model_config.backbone_group}' (shared s3gen/ve"
        f"{', shared t3' if role == ROLE_EMBEDDING else ''}, t3 dtype={backbone.t3_dtype})"
    )
    return model


__all__ = [
    "BASE_REPO",
    "ChatterboxBackbone",
    "_t3_class_for_role",
    "assemble_role",
    "build_backbone",
    "build_role_model",
    "load_s3gen",
    "load_t3",
    "load_ve",
    "role_for_model_type",
    "ROLE_EMBEDDING",
    "ROLE_MULTILINGUAL",
    "ROLE_TTS",
    "ROLE_TURBO",
    "ROLE_VC",
]
