"""Standard voice pack catalog and name resolution (Phase 13-1).

The pack ships **reference WAVs**, not embeddings: every voice-cloning family
embeds a clip in its own embedding space, so the same name means "the same
reference clip", not "identical rendered audio". This module is the single
source of truth for:

* the vendored pack names and their file paths
  (:func:`standard_voice_names`, :func:`standard_voice_path`);
* each configured model's honest voice listing — ``native`` entries (from the
  model config's ``voices:`` keys) plus ``standard`` pack entries for models
  that can actually consume an external reference
  (:func:`voices_for_model`, :func:`voice_catalog`);
* request-level name resolution used by the serving layer
  (:func:`resolve_request_voice`, :func:`request_cloning_capable`,
  :func:`request_vc_target_audio_capable`, :func:`request_native_voices`).

It is consumed by routing (13-1), warm-up (13-2) and the voices endpoint
(13-3).

Cloning capability is a **pipeline** property, not a model-type blanket. A
pipeline is TTS-cloning capable when it offers ``tts`` and has a step that
produces ``embedding_artifact`` (OpenVoice V1/V2, HarmonySpeech, all three
Chatterbox TTS families). A concrete synthesizer (MeloTTS, the OpenVoice V1
synthesizer, the Harmony Speech synthesizer, KittenTTS) never reads a
reference clip on its own, so it is **not** capable even though it is a member
of a capable pipeline.

The VC endpoint has **two** target-voice capability kinds, deliberately kept
distinct instead of collapsed into one boolean:

* ``"target_audio"`` — the executor consumes a target *clip* directly
  (``ChatterboxVC``); the serving layer injects the pack WAV as ``target_audio``.
* ``"target_embedding"`` — the executor consumes ``target_embedding`` only
  (the OpenVoice V1/V2 tone converters). A pack ``target_voice`` is resolved by
  running the WAV through the pipeline's ``vad → embed`` steps first (the same
  encoder artifact the TTS clone path computes, so it rides the 7-2/disk cache).
  This needs the pipeline's encoder and VAD configured; without them the model is
  honestly *not* target-voice capable.

Pack clips are language-agnostic references; standard entries therefore carry
no language label. Native entries carry the config's language (``"default"``
for models such as KittenTTS that expose voices without a language).

Vendored from the MIT-licensed Chatterbox-TTS-Server project; see
``harmonyspeech/assets/voices/LICENSE`` and ``PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.processing.pipeline_registry import (
    PIPELINE_SPECS,
    TTS,
    VC,
    get_spec,
    model_types_advertising,
    validate_families,
)

__all__ = [
    "DIRECT_TTS_CLONING_MODEL_TYPES",
    "DIRECT_VC_TARGET_AUDIO_MODEL_TYPES",
    "DIRECT_VC_TARGET_EMBEDDING_MODEL_TYPES",
    "PACK_DIR",
    "VC_TARGET_AUDIO",
    "VC_TARGET_EMBEDDING",
    "ModelVoiceCatalog",
    "StandardVoiceLookup",
    "TargetEmbeddingRoute",
    "VoiceEntry",
    "VoiceResolution",
    "is_standard_voice",
    "lookup_standard_voice",
    "native_voices",
    "request_cloning_capable",
    "request_model_types",
    "request_native_voices",
    "request_vc_target_audio_capable",
    "request_vc_target_voice_kind",
    "resolve_request_voice",
    "resolve_target_embedding_route",
    "resolve_voice",
    "spec_is_tts_cloning_capable",
    "standard_voice_names",
    "standard_voice_path",
    "vc_target_voice_kind_for_model_type",
    "voice_catalog",
    "voice_catalog_listing",
    "voices_for_model",
]

#: Vendored pack directory (package data; see pyproject/setup.py).
PACK_DIR = Path(__file__).resolve().parent.parent / "assets" / "voices"

#: Model types that directly accept a reference clip on a **TTS** request.
#: These are the concrete TTS executors whose ``prepare_inputs`` reads
#: ``input_audio`` and computes the family's conditionals inline.
DIRECT_TTS_CLONING_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "ChatterboxTTS",
        "ChatterboxTurboTTS",
        "ChatterboxMultilingualTTS",
    }
)

#: Model types that directly accept a **target** clip on a VC request.
#: OpenVoice tone converters are deliberately absent: their VC path consumes
#: ``target_embedding`` and ignores ``target_audio``.
DIRECT_VC_TARGET_AUDIO_MODEL_TYPES: frozenset[str] = frozenset({"ChatterboxVC"})

#: Model types that accept a standard ``target_voice`` on a VC request by
#: embedding the pack clip first (target-**embedding** VC). The embedding is
#: computed by the model's sibling pipeline encoder (``vad → embed``), so
#: capability additionally requires that encoder and the VAD to be configured
#: (see :func:`resolve_target_embedding_route`).
DIRECT_VC_TARGET_EMBEDDING_MODEL_TYPES: frozenset[str] = frozenset(
    {"OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"}
)

#: VC target-voice capability kinds. Never collapse these into a boolean: a
#: target-audio model consumes the clip itself, while a target-embedding model
#: needs the serving layer to run the pipeline's embed step first.
VC_TARGET_AUDIO = "target_audio"
VC_TARGET_EMBEDDING = "target_embedding"

_NATIVE = "native"
_STANDARD = "standard"


@dataclass(frozen=True)
class VoiceEntry:
    """One selectable voice: a native preset or a standard pack identity."""

    name: str
    source: str  # "native" | "standard"
    languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelVoiceCatalog:
    """The honest voice listing for one configured model.

    ``cloning_capable`` means "a standard pack name is meaningful for this
    model" in either direction (TTS reference or VC target). ``vc_target_kind``
    keeps the two VC capability kinds apart for the voices endpoint (13-3):
    ``"target_audio"`` (the model consumes the clip itself), ``"target_embedding"``
    (the serving layer embeds the clip through the pipeline's encoder) or
    ``None`` when the model is not a VC target.

    ``kind`` is ``"model"`` for a configured executor and ``"toolchain"`` for a
    pipeline alias that is not also a configured model name; the 13-3 endpoint
    renders it as the entry's ``object``, mirroring ``/v1/audio/speech/models``.
    """

    model: str
    model_type: str | None
    cloning_capable: bool
    voices: tuple[VoiceEntry, ...]
    vc_target_kind: str | None = None
    kind: str = "model"


@dataclass(frozen=True)
class VoiceResolution:
    """A resolved ``voice``/``target_voice`` name.

    ``kind == "standard"`` carries the vendored ``wav_path``; ``kind ==
    "native"`` carries the model's own ``native_id`` (the wire value the family
    already understands). The serving layer only acts on standard resolutions;
    native names keep their pre-13 behaviour.
    """

    name: str
    kind: str
    wav_path: Path | None = None
    native_id: str | None = None


@dataclass(frozen=True)
class StandardVoiceLookup:
    """Outcome of resolving a request's ``voice``/``target_voice`` field.

    ``is_standard_name`` distinguishes "not a pack name, leave the request
    alone" from "a pack name that resolved or failed". ``native_default`` is the
    native voice id a cloning family needs as its *source* voice (OpenVoice
    V1/V2, MeloTTS); ``None`` for families that ignore ``voice_id``. ``error``
    is an honest admission message; ``None`` when nothing is wrong.
    ``vc_target_kind`` is set only for a ``target=True`` lookup: it tells the
    serving layer whether the pack clip becomes ``target_audio`` directly or must
    be embedded into ``target_embedding`` first.
    """

    is_standard_name: bool = False
    resolution: VoiceResolution | None = None
    native_default: str | None = None
    error: str | None = None
    vc_target_kind: str | None = None


@dataclass(frozen=True)
class TargetEmbeddingRoute:
    """How a VC ``target_voice`` pack clip is embedded for one request.

    ``pipeline_id`` is the routing target for the one-time
    ``SpeechEmbeddingRequestInput`` (the engine's ``vad → embed`` entry rules
    need the pipeline id, not the concrete encoder name), and
    ``encoder_model_name`` is the configured executor that will run the
    embedding. Returned only when the whole embed path is configured.
    """

    pipeline_id: str
    encoder_model_name: str


# ---------------------------------------------------------------------------
# Pack inventory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def standard_voice_names() -> tuple[str, ...]:
    """Return the vendored pack names, sorted (stable across boots).

    The set is part of the public API surface: renaming a clip changes the
    voice identifier. ``tests/unit/modeling/test_voice_catalog.py`` pins it.
    """
    if not PACK_DIR.is_dir():
        return ()
    return tuple(sorted(path.stem for path in PACK_DIR.glob("*.wav")))


def is_standard_voice(name: str | None) -> bool:
    """True when ``name`` is a vendored standard voice pack name."""
    return bool(name) and name in standard_voice_names()


def standard_voice_path(name: str) -> Path | None:
    """Return the WAV path for a pack name, or ``None`` for an unknown name.

    Membership is checked against :func:`standard_voice_names` first, so a
    caller-supplied string can never escape the pack directory.
    """
    if not is_standard_voice(name):
        return None
    return PACK_DIR / f"{name}.wav"


# ---------------------------------------------------------------------------
# Per-model catalog
# ---------------------------------------------------------------------------


def native_voices(model_config: ModelConfig) -> tuple[str, ...]:
    """Return the model's own voice ids from its ``voices:`` config keys."""
    return tuple(dict.fromkeys(model_config.voices or ()))


def _native_languages(model_config: ModelConfig) -> tuple[str, ...]:
    language = getattr(model_config, "language", None)
    return (language,) if language else ("default",)


def vc_target_voice_kind_for_model_type(model_type: str | None) -> str | None:
    """Return the model type's VC target-voice capability kind, if any.

    This is the per-model-type view (used by the voices listing). It does not
    know whether a target-embedding model's sibling encoder is configured; use
    :func:`request_vc_target_voice_kind` for an honest request-level answer.
    """
    if model_type in DIRECT_VC_TARGET_AUDIO_MODEL_TYPES:
        return VC_TARGET_AUDIO
    if model_type in DIRECT_VC_TARGET_EMBEDDING_MODEL_TYPES:
        return VC_TARGET_EMBEDDING
    return None


def _directly_standard_voice_capable(model_config: ModelConfig) -> bool:
    return model_config.model_type in (
        DIRECT_TTS_CLONING_MODEL_TYPES
        | DIRECT_VC_TARGET_AUDIO_MODEL_TYPES
        | DIRECT_VC_TARGET_EMBEDDING_MODEL_TYPES
    )


def voices_for_model(
    model_config: ModelConfig, *, include_standard: bool = True
) -> tuple[VoiceEntry, ...]:
    """Return native entries plus (for capable models) the standard pack names.

    ``include_standard=False`` keeps the native entries only. The 13-3 endpoint
    passes it when the pack is disabled, or when the model's capability is only
    partial (for example a tone converter whose sibling encoder is not
    configured), so the listing never advertises a name the request path would
    reject.
    """
    entries = [
        VoiceEntry(name=name, source=_NATIVE, languages=_native_languages(model_config))
        for name in native_voices(model_config)
    ]
    if include_standard and _directly_standard_voice_capable(model_config):
        entries.extend(
            VoiceEntry(name=name, source=_STANDARD) for name in standard_voice_names()
        )
    return tuple(entries)


def voice_catalog(model_config: ModelConfig) -> ModelVoiceCatalog:
    """Return the 13-3 endpoint's per-model catalog record."""
    capable = _directly_standard_voice_capable(model_config)
    return ModelVoiceCatalog(
        model=model_config.name,
        model_type=model_config.model_type,
        cloning_capable=capable,
        voices=voices_for_model(model_config),
        vc_target_kind=vc_target_voice_kind_for_model_type(model_config.model_type),
    )


# ---------------------------------------------------------------------------
# Endpoint listing (13-3)
# ---------------------------------------------------------------------------


def _standard_entries() -> list[VoiceEntry]:
    return [VoiceEntry(name=name, source=_STANDARD) for name in standard_voice_names()]


def _pipeline_native_entries(model_configs, spec) -> list[VoiceEntry]:
    """Native voices of a pipeline's configured members, grouped by name.

    A name that exists in several language variants (OpenVoice's ``default``)
    is listed once, with every language it is available in. First-seen order
    is stable: pipeline member order, then config order.
    """
    member_types = set(spec.member_types)
    languages_by_name: dict[str, list[str]] = {}
    for config in model_configs:
        if config.model_type not in member_types:
            continue
        for name in native_voices(config):
            languages = languages_by_name.setdefault(name, [])
            for language in _native_languages(config):
                if language not in languages:
                    languages.append(language)
    return [
        VoiceEntry(name=name, source=_NATIVE, languages=tuple(languages))
        for name, languages in languages_by_name.items()
    ]


def _individual_catalog(model_configs, config: ModelConfig, *, standard_pack_enabled: bool) -> ModelVoiceCatalog:
    """Catalog record for a configured executor, at the request level.

    Uses the request-level resolvers (the same ones the serving layer calls) so
    a partial configuration — e.g. a target-embedding tone converter without
    its encoder — is honestly not cloning-capable and lists no pack names.
    """
    tts_capable = request_cloning_capable(model_configs, config.name)
    vc_target_kind = request_vc_target_voice_kind(model_configs, config.name)
    capable = tts_capable or vc_target_kind is not None
    return ModelVoiceCatalog(
        model=config.name,
        model_type=config.model_type,
        cloning_capable=capable,
        voices=voices_for_model(config, include_standard=capable and standard_pack_enabled),
        vc_target_kind=vc_target_kind,
    )


def _pipeline_catalog(model_configs, spec, *, standard_pack_enabled: bool) -> ModelVoiceCatalog:
    """Catalog record for a complete toolchain alias (pipeline id)."""
    vc_target_kind = request_vc_target_voice_kind(model_configs, spec.id)
    capable = spec_is_tts_cloning_capable(spec) or vc_target_kind is not None
    entries = _pipeline_native_entries(model_configs, spec)
    if capable and standard_pack_enabled:
        entries.extend(_standard_entries())
    return ModelVoiceCatalog(
        model=spec.id,
        model_type=None,
        cloning_capable=capable,
        voices=tuple(entries),
        vc_target_kind=vc_target_kind,
        kind="toolchain",
    )


def _merge_voice_entries(base: tuple[VoiceEntry, ...], extra: tuple[VoiceEntry, ...]) -> tuple[VoiceEntry, ...]:
    """Union two voice lists by ``(name, source)``, merging language labels."""
    merged: list[VoiceEntry] = []
    positions: dict[tuple[str, str], int] = {}
    for entry in (*base, *extra):
        key = (entry.name, entry.source)
        position = positions.get(key)
        if position is None:
            positions[key] = len(merged)
            merged.append(entry)
            continue
        current = merged[position]
        languages = current.languages + tuple(
            language for language in entry.languages if language not in current.languages
        )
        if languages != current.languages:
            merged[position] = VoiceEntry(name=current.name, source=current.source, languages=languages)
    return tuple(merged)


def _merge_catalogs(base: ModelVoiceCatalog, extra: ModelVoiceCatalog) -> ModelVoiceCatalog:
    """Merge a toolchain record into the configured model of the same name.

    Mirrors ``/v1/audio/speech/models``, which reuses the individual card when a
    pipeline id is also a configured model name (the shipped ``chatterbox``,
    ``chatterbox_vc``): one id, one entry, with the toolchain metadata attached.
    """
    return ModelVoiceCatalog(
        model=base.model,
        model_type=base.model_type or extra.model_type,
        cloning_capable=base.cloning_capable or extra.cloning_capable,
        voices=_merge_voice_entries(base.voices, extra.voices),
        vc_target_kind=base.vc_target_kind or extra.vc_target_kind,
        kind=base.kind,
    )


def voice_catalog_listing(
    model_configs,
    *,
    standard_pack_enabled: bool = True,
) -> tuple[ModelVoiceCatalog, ...]:
    """Return the ``GET /v1/audio/voices`` listing.

    Mirrors the model endpoints' presentation: one entry per configured model
    whose ``model_type`` advertises ``tts`` or ``vc`` (individual cards), then
    one entry per complete, voice-capable toolchain id (``openvoice_v1``,
    ``harmonyspeech``, ...). A toolchain whose id is also a configured model
    name is merged into that model's entry, so no id is listed twice.

    ``standard_pack_enabled=False`` omits the standard pack entries from
    ``voices``. The capability flags keep describing the model: explicit
    reference clips still work when the pack is switched off, and the pack can
    be re-enabled without a restart.
    """
    configs = list(model_configs or ())
    entries: list[ModelVoiceCatalog] = []
    positions: dict[str, int] = {}

    advertised_types = model_types_advertising((TTS, VC))
    for config in configs:
        if config.model_type not in advertised_types or not config.name:
            continue
        if config.name in positions:
            continue
        positions[config.name] = len(entries)
        entries.append(
            _individual_catalog(configs, config, standard_pack_enabled=standard_pack_enabled)
        )

    complete_families = set(validate_families(configs))
    for spec in PIPELINE_SPECS.values():
        if not spec.advertise_toolchain or spec.id not in complete_families:
            continue
        pipeline = _pipeline_catalog(configs, spec, standard_pack_enabled=standard_pack_enabled)
        if not pipeline.cloning_capable:
            continue
        position = positions.get(spec.id)
        if position is None:
            positions[spec.id] = len(entries)
            entries.append(pipeline)
        else:
            entries[position] = _merge_catalogs(entries[position], pipeline)

    return tuple(entries)


def resolve_voice(model_config: ModelConfig, name: str) -> VoiceResolution | None:
    """Resolve ``name`` for one configured model (native wins on collision)."""
    native = native_voices(model_config)
    if name in native:
        return VoiceResolution(name=name, kind=_NATIVE, native_id=name)
    if _directly_standard_voice_capable(model_config):
        path = standard_voice_path(name)
        if path is not None:
            return VoiceResolution(name=name, kind=_STANDARD, wav_path=path)
    return None


# ---------------------------------------------------------------------------
# Request-level resolution (pipeline ids and concrete model names)
# ---------------------------------------------------------------------------


def spec_is_tts_cloning_capable(spec) -> bool:
    """True when a pipeline can clone from a reference clip.

    A pipeline qualifies when it offers ``tts`` and has a step that produces an
    ``embedding_artifact`` from ``input_audio`` (OpenVoice V1/V2 ``vad → embed``,
    HarmonySpeech ``encoder``, Chatterbox ``embed``). Single-step synthesizers
    have no such step and are therefore honestly not capable.
    """
    if spec is None or TTS not in getattr(spec, "capabilities", frozenset()):
        return False
    return any(
        artifact == "embedding_artifact"
        for step in getattr(spec, "steps", ()) or ()
        for artifact, _field in step.produces
    )


def request_model_types(
    model_configs,
    requested_model: str | None,
    model_name: str | None = None,
    *,
    only_configured: bool = False,
) -> set[str]:
    """Return the model types backing a request.

    Mirrors the serving layer's existing request-family lookup: a pipeline
    sentinel id resolves through its spec's steps, and a concrete configured
    name resolves through ``config.name``. ``only_configured`` restricts the
    spec types to types that are actually configured (the serving layer's
    native-speed/watermark lookup wants that; capability checks do not).
    """
    configs = list(model_configs or ())
    available = {config.model_type for config in configs}
    types: set[str] = set()
    spec = get_spec(requested_model)
    if spec is not None:
        for step in spec.steps:
            for model_type in step.requires:
                if only_configured and model_type not in available:
                    continue
                types.add(model_type)
    name = model_name or requested_model
    types.update(
        config.model_type
        for config in configs
        if config.name == name and config.model_type
    )
    return types


def request_cloning_capable(
    model_configs,
    requested_model: str | None,
    model_name: str | None = None,
) -> bool:
    """True when this request target can clone from a reference clip."""
    spec = get_spec(requested_model)
    if spec_is_tts_cloning_capable(spec):
        return True
    name = model_name or requested_model
    return any(
        config.name == name
        and config.model_type in DIRECT_TTS_CLONING_MODEL_TYPES
        for config in (model_configs or ())
    )


def request_vc_target_audio_capable(
    model_configs,
    requested_model: str | None,
    model_name: str | None = None,
) -> bool:
    """True when this VC target consumes a target *clip* (not just embedding)."""
    spec = get_spec(requested_model)
    if spec is not None and any(
        "ChatterboxVC" in step.requires for step in spec.steps
    ):
        return True
    name = model_name or requested_model
    return any(
        config.name == name
        and config.model_type in DIRECT_VC_TARGET_AUDIO_MODEL_TYPES
        for config in (model_configs or ())
    )


def _spec_embed_step(spec):
    """Return the pipeline step that produces an ``embedding_artifact``."""
    for step in getattr(spec, "steps", ()) or ():
        if any(artifact == "embedding_artifact" for artifact, _field in step.produces):
            return step
    return None


def _spec_advertises_vc(spec) -> bool:
    return any(VC in getattr(step, "advertises", ()) for step in getattr(spec, "steps", ()))


def _spec_containing_model_type(model_type: str | None):
    if not model_type:
        return None
    for spec in PIPELINE_SPECS.values():
        if any(model_type in step.requires for step in spec.steps):
            return spec
    return None


def _embed_path_configured(spec, by_type: dict[str, ModelConfig]) -> bool:
    """True when every mandatory step up to the embed step resolves.

    OpenVoice's target embedding is ``vad → embed``; without a configured VAD
    there is no segment set to embed, so the model is honestly not capable.
    """
    embed_step = _spec_embed_step(spec)
    if embed_step is None:
        return False
    for step in getattr(spec, "steps", ()) or ():
        if step.optional:
            continue
        if not any(model_type in by_type for model_type in step.requires):
            return False
        if step is embed_step:
            return True
    return False


def resolve_target_embedding_route(
    model_configs,
    requested_model: str | None,
    model_name: str | None = None,
) -> TargetEmbeddingRoute | None:
    """Resolve the pipeline + encoder that embeds a VC ``target_voice`` clip.

    Accepts either a pipeline id (``openvoice_v2``) or a concrete tone-converter
    name (``ov2-tone-converter``). Returns ``None`` when the target is not an
    OpenVoice-style target-embedding VC model, or when its mandatory embed path
    (VAD + encoder) is not fully configured — the caller turns that into an
    honest 400 instead of a half-run request.

    The returned ``pipeline_id`` is what the one-time
    ``SpeechEmbeddingRequestInput`` names as ``requested_model``: the engine's
    ``embed_from_audio`` entry rule needs the pipeline id so the VAD step runs
    before the encoder. A concrete encoder name would skip the VAD step.
    """
    configs = list(model_configs or ())
    by_type: dict[str, ModelConfig] = {}
    by_name: dict[str, ModelConfig] = {}
    for config in configs:
        if config.model_type and config.model_type not in by_type:
            by_type[config.model_type] = config
        if config.name:
            by_name[config.name] = config

    name = model_name or requested_model
    spec = get_spec(requested_model)
    if spec is None:
        config = by_name.get(name or "")
        if config is None or config.model_type not in DIRECT_VC_TARGET_EMBEDDING_MODEL_TYPES:
            return None
        spec = _spec_containing_model_type(config.model_type)
        if spec is None:
            return None

    if not _spec_advertises_vc(spec):
        return None
    if not _embed_path_configured(spec, by_type):
        return None
    embed_step = _spec_embed_step(spec)
    encoder = next(
        (by_type[model_type] for model_type in embed_step.requires if model_type in by_type),
        None,
    )
    if encoder is None:  # pragma: no cover - _embed_path_configured already checked
        return None
    return TargetEmbeddingRoute(pipeline_id=spec.id, encoder_model_name=encoder.name)


def request_vc_target_voice_kind(
    model_configs,
    requested_model: str | None,
    model_name: str | None = None,
) -> str | None:
    """Return how this VC target consumes a standard ``target_voice``.

    ``VC_TARGET_AUDIO`` for a target-clip model (``ChatterboxVC``),
    ``VC_TARGET_EMBEDDING`` for an OpenVoice tone converter whose pipeline
    encoder + VAD are configured, or ``None`` when the target cannot accept a
    standard voice name at all.
    """
    spec = get_spec(requested_model)
    if spec is not None and any(
        "ChatterboxVC" in step.requires for step in spec.steps
    ):
        return VC_TARGET_AUDIO
    name = model_name or requested_model
    for config in model_configs or ():
        if config.name == name and config.model_type in DIRECT_VC_TARGET_AUDIO_MODEL_TYPES:
            return VC_TARGET_AUDIO
    if resolve_target_embedding_route(model_configs, requested_model, model_name) is not None:
        return VC_TARGET_EMBEDDING
    return None


def request_native_voices(
    model_configs,
    requested_model: str | None,
    model_name: str | None = None,
    *,
    language_id: str | None = None,
) -> tuple[str, ...]:
    """Return the native voices of the configured models backing a request.

    When ``language_id`` is set, language-tagged configs for other languages
    are skipped (OpenVoice/Melo variants); untagged configs (KittenTTS) stay.
    """
    types = request_model_types(model_configs, requested_model, model_name)
    voices: list[str] = []
    for config in model_configs or ():
        if config.model_type not in types or not config.voices:
            continue
        config_language = getattr(config, "language", None)
        if language_id and config_language and config_language != language_id:
            continue
        voices.extend(config.voices)
    return tuple(dict.fromkeys(voices))


def resolve_request_voice(
    model_configs,
    requested_model: str | None,
    name: str,
    *,
    target: bool = False,
    language_id: str | None = None,
    model_name: str | None = None,
) -> VoiceResolution | None:
    """Resolve ``name`` for a request (native first, then a capable standard pack).

    ``target=True`` selects the VC target-voice capability set (either
    target-audio or target-embedding). Returns ``None`` for an unknown name, or
    for a pack name on a model that cannot consume an external reference — the
    caller turns that into an honest 400.
    """
    if target:
        if is_standard_voice(name) and request_vc_target_voice_kind(
            model_configs, requested_model, model_name
        ) is not None:
            return VoiceResolution(
                name=name, kind=_STANDARD, wav_path=standard_voice_path(name)
            )
        return None

    native = request_native_voices(
        model_configs,
        requested_model,
        model_name,
        language_id=language_id,
    )
    if name in native:
        return VoiceResolution(name=name, kind=_NATIVE, native_id=name)
    if is_standard_voice(name) and request_cloning_capable(
        model_configs, requested_model, model_name
    ):
        return VoiceResolution(
            name=name, kind=_STANDARD, wav_path=standard_voice_path(name)
        )
    return None


def lookup_standard_voice(
    model_configs,
    model: str | None,
    name: str | None,
    *,
    target: bool = False,
    language_id: str | None = None,
    enabled: bool = True,
    model_exists: bool = True,
    has_explicit_reference: bool = False,
) -> StandardVoiceLookup:
    """Resolve a request's ``voice`` / ``target_voice`` with honest errors.

    This is the serving layer's one-call entry point. Non-pack names return an
    empty lookup (``is_standard_name=False``) so the pre-13 behaviour is
    untouched. A pack name on a model that cannot consume an external reference
    returns a clear ``error`` — never a silent fallback to a native voice.

    ``model_exists=False`` returns an empty lookup: the serving layer's model
    check owns the 404, preserving the existing admission order.
    ``has_explicit_reference`` is true when the request also carries its own
    ``input_audio``/``input_embedding`` (or ``target_audio``/``target_embedding``
    for ``target=True``); mixing an explicit reference with a pack name is
    ambiguous and rejected.
    """
    if not name or not is_standard_voice(name):
        return StandardVoiceLookup()
    if not model_exists:
        return StandardVoiceLookup(is_standard_name=True)
    if not enabled:
        return StandardVoiceLookup(
            is_standard_name=True,
            error=(
                f"Voice '{name}' is a standard voice pack name, but the standard voice "
                "pack is disabled on this server (`standard_voice_pack: false`)."
            ),
        )

    requested = model or ""
    configs = list(model_configs or ())

    if has_explicit_reference:
        if target:
            return StandardVoiceLookup(
                is_standard_name=True,
                error=(
                    f"Provide either `target_voice: \"{name}\"` (a standard voice pack "
                    "name) or an explicit `target_audio`/`target_embedding`, not both."
                ),
            )
        return StandardVoiceLookup(
            is_standard_name=True,
            error=(
                f"Provide either `voice: \"{name}\"` (a standard voice pack name) or "
                "explicit reference fields (`input_audio`/`input_embedding`/"
                "`input_vad_data`), not both."
            ),
        )

    if target:
        target_kind = request_vc_target_voice_kind(configs, requested, requested)
        if target_kind is None:
            return StandardVoiceLookup(
                is_standard_name=True,
                error=(
                    f"Target voice '{name}' is a standard voice pack name, but model "
                    f"'{requested}' does not accept a standard target voice for voice "
                    "conversion. Provide `target_audio`/`target_embedding`, or use a "
                    "voice-conversion model that accepts a target voice."
                ),
            )
        resolution = resolve_request_voice(
            configs, requested, name, target=True, model_name=requested
        )
        if resolution is None:  # pragma: no cover - capability checked above
            return StandardVoiceLookup(
                is_standard_name=True,
                error=f"Target voice '{name}' could not be resolved for model '{requested}'.",
            )
        return StandardVoiceLookup(
            is_standard_name=True,
            resolution=resolution,
            vc_target_kind=target_kind,
        )

    if not request_cloning_capable(configs, requested, requested):
        native = request_native_voices(
            configs, requested, requested, language_id=language_id
        )
        detail = f" Native voices: {', '.join(native)}." if native else ""
        return StandardVoiceLookup(
            is_standard_name=True,
            error=(
                f"Voice '{name}' is a standard voice pack name, but model '{requested}' "
                f"does not support voice cloning.{detail} Use mode 'single_speaker_tts' "
                "with one of its native voices, or select a voice-cloning model."
            ),
        )

    native = request_native_voices(
        configs, requested, requested, language_id=language_id
    )
    native_default = "default" if "default" in native else (native[0] if native else None)
    resolution = resolve_request_voice(
        configs, requested, name, language_id=language_id, model_name=requested
    )
    if resolution is None:  # pragma: no cover - capability checked above
        return StandardVoiceLookup(
            is_standard_name=True,
            error=f"Voice '{name}' could not be resolved for model '{requested}'.",
        )
    return StandardVoiceLookup(
        is_standard_name=True, resolution=resolution, native_default=native_default
    )


