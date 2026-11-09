"""Declarative pipeline (toolchain) registry — the routing extension point.

Every model family Harmony Speech Engine can serve is declared once, here, as a
:class:`PipelineSpec`: an ordered list of :class:`Step` descriptors plus the
:class:`EntryRule` set that maps an incoming request shape onto a starting step.

This module is the single source of truth for:

* **routing** — :func:`next_step` resolves the concrete ``ModelConfig`` for the
  request's current step (the old ``reroute_request_*`` if/elif ladders);
* **forwarding** — :func:`advance` maps a finished step's typed artifacts onto the
  next step's request fields (the old output-class-sniffing branch ladder);
* **startup validation** — :func:`validate_registry` names every incomplete
  pipeline and the step that is missing;
* **model cards** — :func:`specs_for_capabilities` /
  :func:`model_types_advertising` drive ``/v1/models`` instead of the hand
  maintained per-serving-module type lists.

Adding a new toolchain therefore means adding one :class:`PipelineSpec` entry to
:data:`PIPELINE_SPECS` (see ``docs/routing.md``).

Design notes
------------
``needs`` is checked against the request's **artifact store first** (Phase 2
precedence); the wire-encoded string field only counts at the boundary, where
``from_wire``/``prefill_audio_artifacts`` populate the store.

``advance()`` never inspects output classes. Each step declares ``produces`` as
``(output attribute, request field)`` pairs, e.g.
``(("vad_artifact", "input_vad_data"), ...)``, so there is exactly one
unambiguous mapping per hop. A step whose output is missing a declared artifact
raises :class:`MissingStepArtifactError`, which the engine turns into a typed
error naming the step instead of silently mis-routing.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.inputs import (
    AudioConversionRequestInput,
    SpeechEmbeddingRequestInput,
    SpeechTranscribeRequestInput,
    SynthesisRequestInput,
    TextToSpeechRequestInput,
    VocodeRequestInput,
    VoiceConversionRequestInput,
)

__all__ = [
    "CAPABILITIES",
    "EntryRule",
    "FAMILY_REQUIREMENTS",
    "MissingStepArtifactError",
    "MissingStepInputError",
    "PIPELINE_SPECS",
    "PipelineExecution",
    "PipelineRoutingError",
    "PipelineSpec",
    "PipelineState",
    "Step",
    "advance",
    "available_model_types",
    "capabilities_for_model_type",
    "completed_step_state",
    "field_present",
    "format_missing_step",
    "get_spec",
    "is_family_available",
    "missing_family_requirements",
    "model_types_advertising",
    "next_step",
    "resolve_entry",
    "specs_for_capabilities",
    "validate_families",
    "validate_registry",
]

# ---------------------------------------------------------------------------
# Capability vocabulary
# ---------------------------------------------------------------------------

TTS = "tts"
EMBED = "embed"
VC = "vc"
STT = "stt"
TRANSLATION = "translation"
VAD = "vad"
RESTORE = "restore"

#: Every capability a pipeline can advertise on ``/v1/models``.
CAPABILITIES: frozenset[str] = frozenset({TTS, EMBED, VC, STT, TRANSLATION, VAD, RESTORE})


# ---------------------------------------------------------------------------
# Typed routing errors
# ---------------------------------------------------------------------------


class PipelineRoutingError(Exception):
    """Base class for spec-driven routing/forwarding failures.

    The engine converts these into per-request error outputs that name the
    pipeline and the step involved, so a misconfigured deployment fails loudly
    instead of mis-routing or hanging.
    """

    def __init__(self, spec_id: str, step_name: str, message: str) -> None:
        super().__init__(message)
        self.spec_id = spec_id
        self.step_name = step_name
        self.message = message


class MissingStepInputError(PipelineRoutingError):
    """A step was entered without one of the fields it declares in ``needs``."""


class MissingStepArtifactError(PipelineRoutingError):
    """A step's output did not carry an artifact declared in ``produces``."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Where a request currently is inside a pipeline.

    Lives on :class:`harmonyspeech.common.request.EngineRequest` and survives
    forwarding, which is what makes the routing decision unambiguous (no more
    inferring the position from which request fields happen to be set).

    ``stop_after`` lets an entry rule finish early (e.g. an embedding-only
    request stops after the ``embed`` step) without inventing a second spec.
    """

    spec_id: str
    step_index: int
    started_at: float = field(default_factory=time.monotonic)
    stop_after: str | None = None

    @property
    def stop_step(self) -> str | None:
        return self.stop_after


@dataclass(frozen=True)
class Step:
    """One executor invocation inside a pipeline.

    Attributes:
        name: Step name, used in errors, entry rules and ``stop_after``.
        requires: ``model_type`` values that can execute this step, in
            preference order — the first one backed by a configured model wins.
        needs: Request fields/artifacts that must be present to enter the step.
            The artifact store is consulted first (Phase 2 precedence).
        produces: ``(output attribute, request field)`` pairs. The finished
            step's typed artifact (e.g. ``vad_artifact``) is stored under the
            request-field name (e.g. ``input_vad_data``).
        clears: Fields the produced artifact supersedes; they are removed from
            the artifact store *and* reset on the request object so a stale
            wire field can never be re-decoded at the next admission.
        optional: Step may be absent from the configuration without making the
            pipeline incomplete (e.g. the dedicated Chatterbox embedding
            executor, which falls back to the TTS model).
        language_matched: A candidate model only matches when its configured
            ``language`` equals ``request.language_id`` (OpenVoice/Melo variants).
        advertises: Capabilities for which this step's model types are
            advertised as individual entries on ``/v1/models``.
        preference_field: Request field that selects between alternative
            backends (``input_vad_mode`` → Silero vs Whisper).
        requires_by_preference: ``(value, model_type tuple)`` pairs used when
            ``preference_field`` is set on the request.
    """

    name: str
    requires: tuple[str, ...]
    needs: frozenset[str] = frozenset()
    produces: tuple[tuple[str, str], ...] = ()
    clears: frozenset[str] = frozenset()
    optional: bool = False
    language_matched: bool = False
    advertises: tuple[str, ...] = ()
    preference_field: str | None = None
    requires_by_preference: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def candidates_for(self, request: Any) -> tuple[str, ...]:
        """Return the ``model_type`` preference list for this request."""
        if self.preference_field is not None:
            value = getattr(request, self.preference_field, None)
            if value:
                preferred = dict(self.requires_by_preference).get(str(value).lower())
                if preferred:
                    return preferred
        return self.requires


@dataclass(frozen=True)
class EntryRule:
    """Maps an incoming request shape onto a starting step.

    ``applies`` receives ``(request_input, artifacts)`` so presence checks are
    artifact aware. Rules are evaluated in declaration order; the first match
    wins and forward requests never re-enter resolution (they carry state).

    ``requires_model_type`` makes a rule configuration dependent: it only applies
    when that ``model_type`` is configured. That is how the optional Chatterbox
    embedding executor is expressed — with a dedicated embedding executor the
    request starts at ``embed``, without one the TTS executor embeds inline and
    the request starts at ``synthesize``.
    """

    name: str
    applies: Callable[[Any, Mapping[str, Any]], bool]
    start_step: str
    stop_after: str | None = None
    requires_model_type: str | None = None


@dataclass(frozen=True)
class PipelineSpec:
    """A complete toolchain: ordered steps, entry rules and capabilities."""

    id: str
    steps: tuple[Step, ...]
    final_step: str
    capabilities: frozenset[str] = frozenset()
    entry_rules: tuple[EntryRule, ...] = ()
    description: str = ""
    #: Whether the pipeline id itself is a client-facing toolchain alias that
    #: ``/v1/models`` should advertise (``openvoice_v2``, ``chatterbox_vc``...).
    #: Single-step executors are addressed by their configured model name, so
    #: they set this to False and only produce individual cards.
    advertise_toolchain: bool = True

    def __post_init__(self) -> None:
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError(f"Pipeline '{self.id}' declares duplicate step names: {names}")
        if self.final_step not in names:
            raise ValueError(f"Pipeline '{self.id}' final_step '{self.final_step}' is not a declared step")
        for rule in self.entry_rules:
            if rule.start_step not in names:
                raise ValueError(f"Pipeline '{self.id}' entry rule '{rule.name}' starts at unknown step")
            if rule.stop_after is not None and rule.stop_after not in names:
                raise ValueError(f"Pipeline '{self.id}' entry rule '{rule.name}' stops at unknown step")

    def step_index(self, name: str) -> int:
        for index, step in enumerate(self.steps):
            if step.name == name:
                return index
        raise KeyError(f"Pipeline '{self.id}' has no step '{name}'")

    def step_at(self, index: int) -> Step | None:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    @property
    def single_step(self) -> bool:
        return len(self.steps) == 1

    @property
    def member_types(self) -> tuple[str, ...]:
        """All ``model_type`` values any step of this pipeline can consume."""
        seen: list[str] = []
        for step in self.steps:
            for model_type in step.requires:
                if model_type not in seen:
                    seen.append(model_type)
        return tuple(seen)


@dataclass
class PipelineExecution:
    """Minimal forwarding context accepted by :func:`advance`.

    The engine normally passes the real :class:`EngineRequest`; this adapter
    exists for callers that only have a request input (unit tests, direct
    executor use) so the forwarding contract stays identical.
    """

    request_data: Any
    artifacts: dict[str, Any] | None = None
    pipeline_state: PipelineState | None = None


# ---------------------------------------------------------------------------
# Presence helpers (artifact-first, Phase 2 precedence)
# ---------------------------------------------------------------------------


def field_present(request: Any, artifacts: Mapping[str, Any] | None, field_name: str) -> bool:
    """True when ``field_name`` is cached as an artifact or set on the request.

    The artifact store wins: a forwarded request carries the typed object and
    may legitimately leave the wire field unset.
    """
    if artifacts is not None and artifacts.get(field_name) is not None:
        return True
    return getattr(request, field_name, None) is not None


def _missing(request: Any, artifacts: Mapping[str, Any] | None, field_name: str) -> bool:
    return not field_present(request, artifacts, field_name)


# ---------------------------------------------------------------------------
# Entry-rule predicates
# ---------------------------------------------------------------------------


def _is_tts(request: Any) -> bool:
    return isinstance(request, TextToSpeechRequestInput)


def _is_cloning(request: Any) -> bool:
    return getattr(request, "mode", None) == "voice_cloning"


def _is_single_speaker(request: Any) -> bool:
    return getattr(request, "mode", None) == "single_speaker_tts"


def _is_embedding(request: Any) -> bool:
    return isinstance(request, SpeechEmbeddingRequestInput)


def _is_transcribe(request: Any) -> bool:
    return isinstance(request, SpeechTranscribeRequestInput)


def _is_synthesis(request: Any) -> bool:
    return isinstance(request, SynthesisRequestInput)


def _is_vocode(request: Any) -> bool:
    return isinstance(request, VocodeRequestInput)


def _is_voice_conversion(request: Any) -> bool:
    return isinstance(request, VoiceConversionRequestInput)


def _is_audio_conversion(request: Any) -> bool:
    return isinstance(request, AudioConversionRequestInput)


# ---------------------------------------------------------------------------
# Step definitions shared by several pipelines
# ---------------------------------------------------------------------------


def _vad_step(model_types: tuple[str, ...] = ("SileroVAD", "FasterWhisper")) -> Step:
    """Voice-activity step used by the OpenVoice pipelines.

    The ``requires`` tuple encodes the **default** preference (Silero first,
    owner decision 2026-11-10: dedicated VAD is more reliable and much faster
    than running a Whisper model only to segment); ``input_vad_mode`` reorders
    it at request time (Phase 3-4), and ``"whisper"`` keeps FasterWhisper
    explicitly.
    """
    return Step(
        name="vad",
        requires=model_types,
        needs=frozenset({"input_audio"}),
        produces=(("vad_artifact", "input_vad_data"),),
        preference_field="input_vad_mode",
        requires_by_preference=(
            ("silero", ("SileroVAD", "FasterWhisper")),
            ("whisper", ("FasterWhisper", "SileroVAD")),
        ),
    )


def _chatterbox_embed_step(tts_type: str) -> Step:
    return Step(
        name="embed",
        requires=("ChatterboxEmbedding", tts_type),
        needs=frozenset({"input_audio"}),
        produces=(("embedding_artifact", "input_embedding"),),
        clears=frozenset({"input_audio"}),
        optional=True,
        advertises=(EMBED,),
    )


def _chatterbox_synthesize_step(tts_type: str) -> Step:
    return Step(
        name="synthesize", requires=(tts_type,), produces=(("audio_artifact", "input_audio"),), advertises=(TTS,)
    )


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------

_HARMONYSPEECH = PipelineSpec(
    id="harmonyspeech",
    description="Harmony Speech V1: encoder → synthesizer → vocoder",
    steps=(
        Step(
            name="encoder",
            requires=("HarmonySpeechEncoder",),
            needs=frozenset({"input_audio"}),
            produces=(("embedding_artifact", "input_embedding"),),
            clears=frozenset({"input_audio"}),
            advertises=(EMBED,),
        ),
        Step(
            name="synthesizer",
            requires=("HarmonySpeechSynthesizer",),
            needs=frozenset({"input_embedding"}),
            produces=(("mel_artifact", "input_audio"),),
        ),
        Step(
            name="vocoder",
            requires=("HarmonySpeechVocoder",),
            needs=frozenset({"input_audio"}),
            produces=(("audio_artifact", "input_audio"),),
        ),
    ),
    final_step="vocoder",
    capabilities=frozenset({TTS, EMBED}),
    entry_rules=(
        EntryRule("embed_request", lambda request, artifacts: _is_embedding(request), "encoder", "encoder"),
        EntryRule(
            "tts_cloning_from_audio",
            lambda request, artifacts: (
                _is_tts(request)
                and field_present(request, artifacts, "input_audio")
                and _missing(request, artifacts, "input_embedding")
            ),
            "encoder",
        ),
        EntryRule("synthesis_request", lambda request, artifacts: _is_synthesis(request), "synthesizer", "synthesizer"),
        EntryRule(
            "tts_with_embedding",
            lambda request, artifacts: (
                _is_tts(request)
                and _missing(request, artifacts, "input_audio")
                and field_present(request, artifacts, "input_embedding")
            ),
            "synthesizer",
        ),
        EntryRule("vocode_request", lambda request, artifacts: _is_vocode(request), "vocoder", "vocoder"),
        EntryRule(
            "tts_vocode",
            lambda request, artifacts: (
                _is_tts(request)
                and field_present(request, artifacts, "input_audio")
                and field_present(request, artifacts, "input_embedding")
            ),
            "vocoder",
            "vocoder",
        ),
    ),
)


def _openvoice_spec(
    spec_id: str, *, encoder_type: str, synthesizer_type: str, converter_type: str, description: str
) -> PipelineSpec:
    """OpenVoice V1/V2 share the same shape; only model types differ."""
    return PipelineSpec(
        id=spec_id,
        description=description,
        steps=(
            _vad_step(),
            Step(
                name="embed",
                requires=(encoder_type,),
                needs=frozenset({"input_audio", "input_vad_data"}),
                produces=(("embedding_artifact", "input_embedding"),),
                clears=frozenset({"input_audio"}),
            ),
            Step(
                # The synthesizer only needs text + voice: a single-speaker
                # request legitimately starts here without an embedding, which
                # is what replaced the old FIXME field-coercion hack.
                name="synthesize",
                requires=(synthesizer_type,),
                produces=(("audio_artifact", "input_audio"),),
                language_matched=True,
                advertises=(TTS,),
            ),
            Step(
                name="tone_convert",
                requires=(converter_type,),
                # Needs are intentionally empty: a voice-conversion request
                # carries ``source_audio``/``target_embedding`` instead of the
                # TTS cloning fields, so the step validates its own inputs.
                produces=(("audio_artifact", "input_audio"),),
                advertises=(VC,),
            ),
        ),
        final_step="tone_convert",
        capabilities=frozenset({TTS, EMBED, VC}),
        entry_rules=(
            EntryRule("transcribe", lambda request, artifacts: _is_transcribe(request), "vad", "vad"),
            EntryRule(
                "tts_cloning_from_audio",
                lambda request, artifacts: (
                    _is_tts(request)
                    and _is_cloning(request)
                    and field_present(request, artifacts, "input_audio")
                    and _missing(request, artifacts, "input_vad_data")
                    and _missing(request, artifacts, "input_embedding")
                ),
                "vad",
            ),
            EntryRule(
                "embed_from_audio",
                lambda request, artifacts: (
                    _is_embedding(request)
                    and field_present(request, artifacts, "input_audio")
                    and _missing(request, artifacts, "input_vad_data")
                ),
                "vad",
                "embed",
            ),
            EntryRule("embed", lambda request, artifacts: _is_embedding(request), "embed", "embed"),
            EntryRule(
                "tts_cloning_from_vad",
                lambda request, artifacts: (
                    _is_tts(request)
                    and _is_cloning(request)
                    and field_present(request, artifacts, "input_audio")
                    and field_present(request, artifacts, "input_vad_data")
                    and _missing(request, artifacts, "input_embedding")
                ),
                "embed",
            ),
            EntryRule(
                "synthesis_request", lambda request, artifacts: _is_synthesis(request), "synthesize", "synthesize"
            ),
            EntryRule(
                # Single-speaker synthesis stops there: there is no cloned
                # speaker to transfer (this is the old FIXME path, now a rule).
                "tts_single_speaker",
                lambda request, artifacts: (
                    _is_tts(request) and _missing(request, artifacts, "input_audio") and _is_single_speaker(request)
                ),
                "synthesize",
                "synthesize",
            ),
            EntryRule(
                "tts_cloning_from_embedding",
                lambda request, artifacts: (
                    _is_tts(request)
                    and _is_cloning(request)
                    and _missing(request, artifacts, "input_audio")
                    and field_present(request, artifacts, "input_embedding")
                ),
                "synthesize",
            ),
            EntryRule(
                "vc_or_final_conversion",
                lambda request, artifacts: (
                    _is_voice_conversion(request)
                    or (
                        _is_tts(request)
                        and _is_cloning(request)
                        and field_present(request, artifacts, "input_audio")
                        and field_present(request, artifacts, "input_embedding")
                    )
                ),
                "tone_convert",
                "tone_convert",
            ),
        ),
    )


_OPENVOICE_V1 = _openvoice_spec(
    "openvoice_v1",
    encoder_type="OpenVoiceV1ToneConverterEncoder",
    synthesizer_type="OpenVoiceV1Synthesizer",
    converter_type="OpenVoiceV1ToneConverter",
    description="OpenVoice V1: vad → embed → synthesize → tone_convert",
)

_OPENVOICE_V2 = _openvoice_spec(
    "openvoice_v2",
    encoder_type="OpenVoiceV2ToneConverterEncoder",
    synthesizer_type="MeloTTSSynthesizer",
    converter_type="OpenVoiceV2ToneConverter",
    description="OpenVoice V2 (MeloTTS): vad → embed → synthesize → tone_convert",
)

_VOICEFIXER = PipelineSpec(
    id="voicefixer",
    description="VoiceFixer: restore → vocode",
    steps=(
        Step(
            name="restore",
            requires=("VoiceFixerRestorer",),
            needs=frozenset({"source_audio"}),
            produces=(("mel_artifact", "input_mel_spectrogram"),),
        ),
        Step(
            name="vocode",
            requires=("VoiceFixerVocoder",),
            needs=frozenset({"input_mel_spectrogram"}),
            produces=(("audio_artifact", "input_audio"),),
        ),
    ),
    final_step="vocode",
    capabilities=frozenset({RESTORE}),
    entry_rules=(
        EntryRule(
            "restore",
            lambda request, artifacts: (
                _is_audio_conversion(request) and _missing(request, artifacts, "input_mel_spectrogram")
            ),
            "restore",
        ),
        EntryRule(
            "vocode",
            lambda request, artifacts: (
                _is_audio_conversion(request) and field_present(request, artifacts, "input_mel_spectrogram")
            ),
            "vocode",
            "vocode",
        ),
    ),
)


def _chatterbox_spec(spec_id: str, tts_type: str, description: str) -> PipelineSpec:
    return PipelineSpec(
        id=spec_id,
        description=description,
        steps=(_chatterbox_embed_step(tts_type), _chatterbox_synthesize_step(tts_type)),
        final_step="synthesize",
        capabilities=frozenset({TTS, EMBED}),
        entry_rules=(
            EntryRule("embed_request", lambda request, artifacts: _is_embedding(request), "embed", "embed"),
            EntryRule(
                # Only when a dedicated embedding executor is configured; the
                # TTS executor can also embed from reference audio inline,
                # which is why the plain TTS rule below is the fallback.
                "tts_cloning_with_embed_executor",
                lambda request, artifacts: (
                    _is_tts(request)
                    and field_present(request, artifacts, "input_audio")
                    and _missing(request, artifacts, "input_embedding")
                ),
                "embed",
                requires_model_type="ChatterboxEmbedding",
            ),
            EntryRule(
                "tts_synthesize",
                lambda request, artifacts: (
                    _is_tts(request)
                    and not (
                        field_present(request, artifacts, "input_audio")
                        and field_present(request, artifacts, "input_embedding")
                    )
                ),
                "synthesize",
            ),
        ),
    )


_CHATTERBOX = _chatterbox_spec("chatterbox", "ChatterboxTTS", "Chatterbox: embed (optional) → synthesize")
_CHATTERBOX_TURBO = _chatterbox_spec(
    "chatterbox_turbo", "ChatterboxTurboTTS", "Chatterbox Turbo: embed (optional) → synthesize"
)
_CHATTERBOX_MULTILINGUAL = _chatterbox_spec(
    "chatterbox_multilingual", "ChatterboxMultilingualTTS", "Chatterbox Multilingual: embed (optional) → synthesize"
)

_CHATTERBOX_VC = PipelineSpec(
    id="chatterbox_vc",
    description="Chatterbox voice conversion",
    steps=(
        Step(
            name="convert", requires=("ChatterboxVC",), produces=(("audio_artifact", "input_audio"),), advertises=(VC,)
        ),
    ),
    final_step="convert",
    capabilities=frozenset({VC}),
    entry_rules=(EntryRule("convert", lambda request, artifacts: _is_voice_conversion(request), "convert", "convert"),),
)

_KITTEN_TTS = PipelineSpec(
    id="kitten_tts",
    description="KittenTTS single-step synthesis",
    steps=(
        Step(
            name="synthesize",
            requires=("KittenTTSSynthesizer",),
            produces=(("audio_artifact", "input_audio"),),
            advertises=(TTS,),
        ),
    ),
    final_step="synthesize",
    capabilities=frozenset({TTS}),
    entry_rules=(EntryRule("synthesize", lambda request, artifacts: True, "synthesize", "synthesize"),),
    advertise_toolchain=False,
)

_FASTERWHISPER = PipelineSpec(
    id="fasterwhisper",
    description="Faster-Whisper transcription / translation / VAD single-step pipeline",
    steps=(
        Step(
            name="transcribe",
            requires=("FasterWhisper",),
            needs=frozenset({"input_audio"}),
            produces=(("vad_artifact", "input_vad_data"),),
            advertises=(STT, TRANSLATION, VAD),
        ),
    ),
    final_step="transcribe",
    capabilities=frozenset({STT, TRANSLATION, VAD}),
    entry_rules=(EntryRule("transcribe", lambda request, artifacts: True, "transcribe", "transcribe"),),
    advertise_toolchain=False,
)

_SILERO_VAD = PipelineSpec(
    id="silero_vad",
    description="Silero VAD single-step pipeline",
    steps=(
        Step(
            name="detect",
            requires=("SileroVAD",),
            needs=frozenset({"input_audio"}),
            produces=(("vad_artifact", "input_vad_data"),),
            advertises=(VAD,),
        ),
    ),
    final_step="detect",
    capabilities=frozenset({VAD}),
    entry_rules=(EntryRule("detect", lambda request, artifacts: True, "detect", "detect"),),
    advertise_toolchain=False,
)

#: Every registered pipeline, in stable (documentation) order.
PIPELINE_SPECS: dict[str, PipelineSpec] = {
    spec.id: spec
    for spec in (
        _HARMONYSPEECH,
        _OPENVOICE_V1,
        _OPENVOICE_V2,
        _VOICEFIXER,
        _CHATTERBOX,
        _CHATTERBOX_TURBO,
        _CHATTERBOX_MULTILINGUAL,
        _CHATTERBOX_VC,
        _KITTEN_TTS,
        _FASTERWHISPER,
        _SILERO_VAD,
    )
}


def _family_specs() -> list[PipelineSpec]:
    """Specs that are a client-facing toolchain family (not a bare executor)."""
    return [spec for spec in PIPELINE_SPECS.values() if spec.advertise_toolchain]


def _flat_requirements(spec: PipelineSpec) -> list[str]:
    """First (preferred) model type of every mandatory step — legacy view."""
    return [step.requires[0] for step in spec.steps if not step.optional]


#: Backwards-compatible view of the declarative registry (Phase 0-4 API).
#:
#: Derived from :data:`PIPELINE_SPECS` — never edit this dict by hand. Single
#: step executions (``kitten_tts``, ``fasterwhisper``, ``silero_vad``) are
#: excluded: they are addressed by their configured model name, not a family id.
FAMILY_REQUIREMENTS: dict[str, list[str]] = {spec.id: _flat_requirements(spec) for spec in _family_specs()}


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def get_spec(requested_model: str | None) -> PipelineSpec | None:
    """Return the pipeline registered under ``requested_model``, if any.

    Only sentinel family ids resolve here. A concrete configured model name
    (``"kitten-tts-mini"``, ``"ov1-synthesizer-en"``) intentionally returns
    ``None``: those requests are handled by the explicit-model path.
    """
    if not requested_model:
        return None
    return PIPELINE_SPECS.get(requested_model)


def specs_for_capabilities(capabilities: str | Iterable[str]) -> list[PipelineSpec]:
    """Return every spec offering at least one of ``capabilities``."""
    wanted = {capabilities} if isinstance(capabilities, str) else set(capabilities)
    return [spec for spec in PIPELINE_SPECS.values() if spec.capabilities & wanted]


def model_types_advertising(capabilities: str | Iterable[str]) -> set[str]:
    """Return the ``model_type`` values advertised individually for a capability."""
    wanted = {capabilities} if isinstance(capabilities, str) else set(capabilities)
    types: set[str] = set()
    for spec in PIPELINE_SPECS.values():
        for step in spec.steps:
            if set(step.advertises) & wanted:
                types.update(step.requires)
    return types


def capabilities_for_model_type(model_type: str) -> set[str]:
    """Return every capability a configured ``model_type`` advertises.

    Individual model cards carry this set additively (Phase 10-4), so a card
    never needs a hand-written capability list: a Whisper executor advertises
    ``stt``, ``translation`` and ``vad`` because the registry's step says so.
    """
    capabilities: set[str] = set()
    for spec in PIPELINE_SPECS.values():
        for step in spec.steps:
            if model_type in step.requires:
                capabilities.update(step.advertises)
    return capabilities


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_entry(
    spec: PipelineSpec,
    request: Any,
    artifacts: Mapping[str, Any] | None = None,
    model_configs: Sequence[ModelConfig] | None = None,
) -> PipelineState | None:
    """Return the starting :class:`PipelineState` for ``request``, or ``None``.

    ``None`` means the request shape is not supported by this pipeline; the
    engine reports it as a typed error instead of queueing the request. Rules
    with ``requires_model_type`` are skipped when that model type is not among
    ``model_configs`` (pass the configured models to enable them).
    """
    artifacts = artifacts or {}
    available_types = available_model_types(model_configs)
    for rule in spec.entry_rules:
        if rule.requires_model_type is not None and rule.requires_model_type not in available_types:
            continue
        if rule.applies(request, artifacts):
            return PipelineState(
                spec_id=spec.id,
                step_index=spec.step_index(rule.start_step),
                stop_after=rule.stop_after or spec.final_step,
            )
    return None


def completed_step_state(spec: PipelineSpec, model_type: str, *, stop_after: str | None = None) -> PipelineState | None:
    """State pointing at the step executed by ``model_type`` (forwarding fallback).

    Used when an ``EngineRequest`` cannot be located by id (defensive path in
    ``check_forward_processing``); the completed step is unambiguous because a
    model type appears in exactly one step of a pipeline.
    """
    for index, step in enumerate(spec.steps):
        if model_type in step.requires:
            return PipelineState(spec_id=spec.id, step_index=index, stop_after=stop_after or spec.final_step)
    return None


def check_step_inputs(step: Step, request: Any, artifacts: Mapping[str, Any] | None = None) -> None:
    """Raise :class:`MissingStepInputError` when ``step.needs`` is unsatisfied."""
    missing = sorted(field_name for field_name in step.needs if not field_present(request, artifacts, field_name))
    if missing:
        raise MissingStepInputError(
            getattr(request, "requested_model", "") or "",
            step.name,
            f"Pipeline step '{step.name}' cannot run — missing input(s): {', '.join(missing)}",
        )


def _warn_preference_fallback(step: Step, request: Any, available_types: set[str]) -> None:
    if step.preference_field is None:
        return
    value = getattr(request, step.preference_field, None)
    if not value:
        return
    preferred = dict(step.requires_by_preference).get(str(value).lower())
    if not preferred or preferred[0] in available_types:
        return
    fallback = next((candidate for candidate in preferred[1:] if candidate in available_types), None)
    if fallback is not None:
        logger.warning(
            f"Pipeline '{step.name}': {step.preference_field}='{value}' requested "
            f"'{preferred[0]}' but no such model is configured — falling back to '{fallback}'."
        )


def next_step(
    spec: PipelineSpec,
    state: PipelineState | None,
    request: Any,
    model_configs: Sequence[ModelConfig] | None,
    artifacts: Mapping[str, Any] | None = None,
) -> ModelConfig | None:
    """Resolve the concrete model for ``spec.steps[state.step_index]``.

    ``requires`` is a preference list: the first entry backed by a configured
    model wins. Language-variant steps additionally require the configured
    ``language`` to equal ``request.language_id``. Returns ``None`` when nothing
    matches — the engine renders that as the "pipeline not fully configured"
    error.
    """
    if spec is None or state is None:
        return None
    step = spec.step_at(state.step_index)
    if step is None:
        return None
    check_step_inputs(step, request, artifacts)

    configs = list(model_configs or [])
    available_types = {config.model_type for config in configs}
    candidates = step.candidates_for(request)
    _warn_preference_fallback(step, request, available_types)

    for model_type in candidates:
        for config in configs:
            if config.model_type != model_type:
                continue
            if step.language_matched and getattr(config, "language", None) != getattr(request, "language_id", None):
                continue
            return config
    return None


def _artifact_store(engine_request: Any) -> dict[str, Any]:
    artifacts = getattr(engine_request, "artifacts", None)
    if not isinstance(artifacts, dict):
        artifacts = {}
        with contextlib.suppress(AttributeError):  # pragma: no cover - immutable doubles
            engine_request.artifacts = artifacts
    return artifacts


def advance(spec: PipelineSpec, engine_request: Any, result_output: Any) -> bool:
    """Move the pipeline one step forward; return True when it is finished.

    The step's typed output artifacts are stored under the request-field names
    declared in :attr:`Step.produces` (no output-class sniffing), fields listed
    in :attr:`Step.clears` are dropped, and ``pipeline_state.step_index`` is
    incremented.

    Raises :class:`MissingStepArtifactError` when the output does not carry a
    declared artifact — the engine turns that into a typed error naming the step.

    **Phase 6 extension point.** Post-generation processing (extra filters after
    the final step) plugs in here: a caller may append steps to a per-request
    copy of the spec and let ``advance()`` walk them exactly like the built-in
    ones. Nothing else in the engine needs to change.
    """
    if spec is None:
        return True
    state = getattr(engine_request, "pipeline_state", None)
    request_input = getattr(engine_request, "request_data", None)
    artifacts = _artifact_store(engine_request)
    if state is None:
        raise MissingStepArtifactError(
            spec.id, "<unknown>", f"Pipeline '{spec.id}' has no pipeline state; cannot advance it."
        )
    step = spec.step_at(state.step_index)
    if step is None:
        raise MissingStepArtifactError(
            spec.id, "<unknown>", f"Pipeline '{spec.id}' state points past its last step ({state.step_index})."
        )

    produced: dict[str, Any] = {}
    for output_attribute, field_name in step.produces:
        artifact = getattr(result_output, output_attribute, None)
        if artifact is None:
            raise MissingStepArtifactError(
                spec.id,
                step.name,
                f"Pipeline '{spec.id}' step '{step.name}' did not produce its declared artifact "
                f"'{output_attribute}' (for request field '{field_name}').",
            )
        produced[field_name] = artifact

    artifacts.update(produced)
    for field_name in step.clears:
        artifacts.pop(field_name, None)
        if request_input is not None and getattr(request_input, field_name, None) is not None:
            with contextlib.suppress(AttributeError):  # pragma: no cover - read-only doubles
                setattr(request_input, field_name, None)

    state.step_index += 1
    return step.name == (state.stop_after or spec.final_step)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


def available_model_types(model_configs: Iterable[ModelConfig] | None) -> set[str]:
    """Return the set of ``model_type`` values present in the configuration."""
    return {config.model_type for config in (model_configs or [])}


def format_missing_step(step: Step) -> str:
    """Render a missing step as ``name (type, alternative)``."""
    return f"{step.name} ({', '.join(step.requires)})"


def validate_registry(model_configs: Iterable[ModelConfig] | None) -> dict[str, list[str]]:
    """Return ``{spec_id: [missing step descriptions]}`` for incomplete pipelines.

    A step resolves when *at least one* of its ``requires`` model types is
    configured (language variants are validated for count only, at request time
    the language must match). Optional steps never make a pipeline incomplete.
    """
    available = available_model_types(model_configs)
    incomplete: dict[str, list[str]] = {}
    for spec_id, spec in PIPELINE_SPECS.items():
        missing = [
            format_missing_step(step)
            for step in spec.steps
            if not step.optional and not any(model_type in available for model_type in step.requires)
        ]
        if missing:
            incomplete[spec_id] = missing
    return incomplete


def validate_families(model_configs: Iterable[ModelConfig] | None) -> list[str]:
    """Return the complete multi-step family ids (Phase 0-4 public surface)."""
    incomplete = validate_registry(model_configs)
    return [spec.id for spec in _family_specs() if spec.id not in incomplete]


def missing_family_requirements(model_configs: Iterable[ModelConfig] | None) -> dict[str, list[str]]:
    """Map each incomplete family to the ``model_type``\\ s it cannot resolve.

    Alternative backends (Whisper vs Silero for the VAD step) are reported as
    alternatives of the same step; the step is only missing when neither is
    configured.
    """
    available = available_model_types(model_configs)
    missing: dict[str, list[str]] = {}
    for spec in _family_specs():
        absent: list[str] = []
        for step in spec.steps:
            if step.optional or any(model_type in available for model_type in step.requires):
                continue
            absent.extend(model_type for model_type in step.requires if model_type not in available)
        if absent:
            missing[spec.id] = absent
    return missing


def is_family_available(family: str, model_configs: Iterable[ModelConfig] | None) -> bool:
    """Return True if ``family`` is a declared, complete multi-step toolchain."""
    if family not in FAMILY_REQUIREMENTS:
        return False
    return family not in validate_registry(model_configs)


def incomplete_registry_summary(model_configs: Iterable[ModelConfig] | None) -> str:
    """Human readable one-liner of every incomplete pipeline (logs/tests)."""
    incomplete = validate_registry(model_configs)
    return "; ".join(f"{spec_id}: {', '.join(steps)}" for spec_id, steps in incomplete.items())
