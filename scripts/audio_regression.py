"""Threshold-based audio regression harness (Phase 4-2).

Compares two *configurations of the current build* along a change axis — dtype
(fp32 vs fp16), weight sharing (deduplicated vs duplicated), padded batching
(B=1 vs B=N, Phase 5-2) or a plain identity run — on fixed inputs, and fails
(non-zero exit) when a family leaves its thresholds.

Design notes
------------
* **Two separate processes per family.** Each side boots its own engine in a
  subprocess, so VRAM is genuinely released between sides and the measurement is
  not contaminated by a resident fp32 engine. The parent orchestrates, keeps the
  metrics code, and is the only place that needs spectra/numpy heavy lifting.
* **No golden audio.** Bit-exactness across torch versions is unrealistic, so
  the harness always derives both sides of the axis in the same run.
* **Deterministic vs stochastic.** Deterministic families (HarmonySpeech,
  VoiceFixer, KittenTTS) are compared sample-wise (waveform correlation +
  frame-aligned log-mel MSE). Stochastic families (Chatterbox sampling,
  OpenVoice/MeloTTS `noise_scale`) are compared with distribution metrics:
  duration delta, RMS delta, and long-term average spectrum (LTAS) distance —
  frame-by-frame comparison of two independent samples of the same distribution
  is meaningless.
* **Sharing axis is an identity gate.** Deduplicating weights must not change a
  single bit: the harness requires identical length and bitwise-identical
  samples.
* **Batching axis is allow-listed.** ``batching:1-vs-N`` runs only for families
  whose handler really forwards a padded batch (:data:`BATCHING_AXIS_ENABLED`).
  The remaining Phase 8 families already carry their mixed-length case sets, so
  enabling one is a one-line edit to that set.
* **Long-form axes are premise-checked (Phase 12-4).** ``longform:off-vs-on``
  only means something when the case text stays inside ONE chunk at the serving
  layer's budget, and when the clone reference is speech the VAD can detect; a
  case that violates either premise is excluded/retargeted with a recorded
  reason instead of failing for the wrong reason. ``determinism:run1-vs-run2``
  asserts sample identity for deterministic families, the documented
  cross-process noise floor for seeded stochastic ones, and a distributional
  gate (hard duration + RMS/LTAS) for a model that samples on every inference
  (:data:`MODEL_SAMPLES_EVERY_RUN`) — the KittenTTS ONNX graph has unseeded
  ``RandomNormalLike``/``RandomUniformLike`` nodes, so its two runs can never be
  sample-identical.
* **Output filters are not a harness axis (Phase 12-3).** ``remove_dc_offset``,
  ``trim_silence`` and ``fix_internal_silence`` are opt-in post-processing whose
  *intent* is to change samples. The honest on-vs-off gate ("same seed, flags on
  vs off => expected difference, valid WAV") lives on synthesized fixtures with
  known DC/silence structure in ``tests/unit/common/test_audio_utils.py`` and
  ``tests/integration/test_tts_output_filters.py``, because a real model may
  legitimately emit no DC offset and no long silence — then a "must differ" rule
  would fail for the wrong reason. The harness therefore treats the filters as
  an explicit opt-in axis outside its drift comparison rather than rebuilding
  its metrics for a non-regression.

Usage
-----
    python scripts/audio_regression.py --families chatterbox,openvoice_v1 \
        --axis dtype:float32-vs-float16 --gpu

    python scripts/audio_regression.py --families chatterbox \
        --axis sharing:dedup-vs-dup --gpu

    # padded-batch equivalence (Phase 5-2): B=1 loop vs one 4-request batch
    python scripts/audio_regression.py --families melotts,openvoice_v1 \
        --axis batching:1-vs-4 --gpu

    # identity self-test (must always pass)
    python scripts/audio_regression.py --families kittentts \
        --axis dtype:float32-vs-float32

    # S3Gen autocast experiment (Phase 11-2): same seeded render with
    # HSE_S3GEN_AUTOCAST=off vs bfloat16 (--s3gen-autocast float16 for fp16)
    python scripts/audio_regression.py \
        --families chatterbox,chatterbox_multilingual,chatterbox_turbo \
        --axis dtype:fp32-vs-autocast --s3gen-autocast bfloat16 --gpu
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Thresholds (initial values, tuned on the first GPU run — see
# docs/benchmarks/audio-regression.md)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "deterministic": {
        "logmel_mse": 1.0,  # dB^2 over aligned mel frames
        "duration_pct": 1.0,
        "rms_pct": 5.0,
        "corr_min": 0.98,
    },
    "stochastic": {
        "ltas_rmse_db": 4.0,  # long-term average spectrum per mel band
        "duration_pct": 12.0,
        "rms_pct": 15.0,
    },
}

#: Per-family overrides. Values are filled in from measured runs; a family whose
#: measured spread is structurally larger (e.g. an AR model whose token stream
#: legitimately diverges between two samples) gets its own numbers rather than
#: hiding the difference in a global loosening.
FAMILY_THRESHOLDS: dict[str, dict] = {}

#: Families whose dtype axis is not meaningful in the shipped configuration
#: (CPU-pinned entries are rewritten to bf16 by the CPU executor, and ONNX/CT2
#: backends ignore torch dtype entirely).
DTYPE_AXIS_EXEMPT = {
    "kittentts": "ONNX runtime ignores torch dtype (variant chosen by model name)",
    "harmonyspeech": "CPU-pinned in the shipped GPU config (torch dtype only selects bf16 on CPU)",
}

#: Batching axis (Phase 5-2). Padded-batch inference must not change what a
#: sample sounds like: with sampling noise disabled both sides compute the same
#: thing, so the comparison is sample-wise. The tolerances cover GPU
#: accumulation-order noise only (measured max |Δsample| ≈ 1.1e-4 on the RTX
#: 5060 Ti with TF32 disabled — see docs/benchmarks/throughput.md).
BATCHING_MAX_ABS_DIFF = 2e-3
BATCHING_LOG_MEL_MSE = 1e-2
BATCHING_DURATION_PCT = 0.1
BATCHING_CORR_MIN = 0.999

#: Families whose batching gate is runnable today — the handler genuinely
#: forwards a padded batch, so a ``batching:1-vs-N`` comparison means something.
#: This is the axis allow-list: a family with ``batching_cases`` registered in
#: :data:`FAMILY_SPECS` but absent here is refused with the activation task's
#: name instead of producing an empty gate. Enabling a family is a one-line
#: edit to this set (Phase 8-2…8-4 flip their own families on when the handler
#: and its config flag land — never before, the flag must not advertise a
#: capability the executor lacks).
BATCHING_AXIS_ENABLED = frozenset(
    {"openvoice_v1", "openvoice_v2", "melotts", "harmonyspeech", "voicefixer", "chatterbox", "chatterbox_turbo"}
)
#: Phase 11-1 extends the `chatterbox` gate with VC cases (`kind: vc` on
#: `chatterbox_vc`): the allow-list is family-keyed and VC is a model inside the
#: `chatterbox` family, so no new name is needed.

#: AR-decode families whose ``batching:1-vs-N`` gate is **distributional**, not
#: sample-wise: the sequential loop samples vocode noise from the global RNG while
#: the AR runner draws each row's AR + S3Gen noise from a per-row seeded
#: generator, so the two paths agree on the token stream (the e2e seeded parity
#: harness gates that) but not sample-for-sample on the audio. The gate still
#: asserts duration/RMS/LTAS stay inside the stochastic thresholds.
BATCHING_AXIS_STOCHASTIC = frozenset({"chatterbox", "chatterbox_turbo"})

#: S3Gen autocast axis (Phase 11-2). ``dtype:fp32-vs-autocast`` compares the same
#: seeded render with ``HSE_S3GEN_AUTOCAST=off`` (side A) and the requested
#: precision (side B). Autocast changes the arithmetic, not the computation: the
#: AR token stream is untouched (T3 stays float32 on both sides), the CFM step
#: count is identical, and only matmul-heavy S3Gen ops drop to bf16/fp16, so the
#: comparison is sample-wise.
#:
#: Thresholds are the harness's existing *audibility* bounds (the deterministic
#: family set: log-mel MSE < 1.0 dB^2, correlation > 0.98, RMS < 5 %), not the
#: batching axis's near-exact bounds — padded batching is a pure reshaping and
#: must be bit-near, while reduced precision is expected to move the arithmetic.
#: The calibration comes from Phase 4: casting S3Gen's *weights* to fp16 measured
#: 8.9 and 141.6 dB^2 with correlations 0.76 and 0.05 and was judged audible
#: (`docs/models.md` §Precision), so 1.0 dB^2 leaves a 9x margin below the known
#: audible class while still being far tighter than "any change passes". For
#: reference, a batching-order bound (1e-2 dB^2) would fail *both* precisions
#: (measured bf16 0.28-3.0, fp16 0.008-0.099 dB^2) — that is too strict for a
#: precision axis and is recorded in the task file rather than used here.
#: Duration must not move: a different frame count means the token stream changed,
#: which is a hard failure for a precision axis.
AUTOCAST_LOG_MEL_MSE = 1.0
AUTOCAST_DURATION_PCT = 0.1
AUTOCAST_RMS_PCT = 5.0
AUTOCAST_CORR_MIN = 0.98
#: The two side labels of the autocast axis (side A is the fp32 reference).
AUTOCAST_AXIS_SIDES = ("fp32", "autocast")

# ---------------------------------------------------------------------------
# Long-form axes (Phase 12-4)
# ---------------------------------------------------------------------------
#: ``longform:off-vs-on`` is the Phase 12-2 byte-identity gate re-run at family
#: breadth: a short text (0/1 chunks) with ``split_text: true`` must take the
#: untouched single-shot path, so the two sides are the same computation. The
#: in-process gate is bit-exact; this cross-process re-run disables the watermark
#: (Perth's conv kernels are not bit-reproducible across processes — the sharing
#: axis hit the same limit) and requires identical length. Deterministic
#: families must be bit-identical; stochastic families are compared with the
#: documented CUDA kernel-selection noise floor (:data:`SHARING_MAX_ABS_DIFF`).
#:
#: ``determinism:run1-vs-run2`` runs the same *chunked* request twice with the
#: same request seed; the chunk seeds are ``seed + chunk_index`` and the stitch
#: is by index, so the stitched bytes must match across the two runs.
LONGFORM_AXIS_SEED = 1234
#: Character budget for the determinism axis: small enough that the fixture text
#: splits into several chunks, so the stitched path really runs.
LONGFORM_AXIS_CHUNK_SIZE = 120
#: A deterministic multi-sentence fixture for the chunked-determinism axis. No
#: abbreviations/digits, so the splitter boundaries are unambiguous.
LONGFORM_CHUNKED_TEXT = (
    "The morning train carries a quiet signal across the river while the harbor wakes. "
    "Each narrow window frames a distant garden and a steady pattern of light. "
    "We gather the simple story before the weather turns and the current moves. "
    "A careful measure of motion helps every traveler find the open meadow. "
    "The engine listens through the long afternoon and promises a gentle return. "
    "Later the lantern marks the footpath where the evening story begins."
)
#: TTS families the long-form axes run at breadth. VoiceFixer has no TTS case
#: (it restores audio), so it is not in this set.
LONGFORM_AXIS_FAMILIES = (
    "chatterbox",
    "chatterbox_turbo",
    "chatterbox_multilingual",
    "openvoice_v1",
    "openvoice_v2",
    "melotts",
    "harmonyspeech",
    "kittentts",
)

#: Character budget the serving layer applies when a client sets ``split_text``
#: without a ``chunk_size`` (``serving_text_to_speech.DEFAULT_CHUNK_SIZE``, the
#: Phase 12-2 owner decision). The identity axis deliberately leaves
#: ``chunk_size`` unset so the OFF side never chunks at all; the ON side
#: therefore splits at this budget. A case text must stay inside ONE chunk or
#: the two sides legitimately differ (different synthesis boundaries) and the
#: byte-identity gate would fail for the wrong reason. The unit self-check pins
#: this value to the serving constant.
LONGFORM_IDENTITY_CHUNK_BUDGET = 300

#: Families whose *model* draws fresh noise on every inference, so two runs of
#: the same seeded request cannot be sample-identical no matter what the engine
#: does. Verified 2026-11-09 on the KittenTTS ONNX graphs: the graph carries
#: ``RandomNormalLike`` / ``RandomUniformLike`` nodes **without a seed
#: attribute**, and two fresh processes synthesising the same sentence differ
#: (max |Δsample| 0.14, corr 0.992, log-mel MSE 2.45 dB²) with identical
#: duration. Their identity/determinism gates are therefore distributional
#: (duration is still a hard invariant; RMS/LTAS use the stochastic bounds).
#: Engine-side chunk→result pairing for these families is pinned separately,
#: in-process, with a deterministic stub (tests/integration/test_tts_chunking.py).
MODEL_SAMPLES_EVERY_RUN = frozenset({"kittentts"})

#: Families whose clone conditioning is carried through the Silero VAD step.
#: A reference clip the VAD cannot detect makes the tone-converter encoder's
#: segment list empty (OpenVoice ``vad → embed``). The identity axis swaps a
#: synthetic reference for a real-speech one on these families only: Chatterbox
#: and HarmonySpeech condition on the clip directly and are happy with the
#: synthetic fixtures.
VAD_CARRIED_FAMILIES = frozenset({"openvoice_v1", "openvoice_v2"})

#: Real-speech replacement for a synthetic (VAD-undetectable) reference clip.
#: Chosen to keep the same role (a/b) so the case still exercises a second
#: timbre; see ``tests.audio_fixtures.VAD_UNDETECTABLE_CLIPS``.
REAL_SPEECH_REFERENCE = {
    "voice_a": "repo_voice_a",
    "voice_b": "repo_voice_b",
}


#: Families with ``batching_cases`` already registered (so a later Phase 8 task
#: only flips the allow-list entry above) and the task that enables them.
#: Used for the refusal message, never to select cases. Empty since 8-4 enabled
#: the last two families; the dict stays as the mechanism's carrier and its test
#: drives it through ``monkeypatch``.
BATCHING_AXIS_PENDING: dict[str, str] = {}

#: Voice-conversion intelligibility gate (post-review 2026-11-05). VC converts
#: the *content* of the source clip, so a converted clip must still transcribe
#: to the source transcript. WER is measured by transcribing both rendered sides
#: with the harness's faster-whisper model and comparing against ``TEXTS``.
#: Reported + warning above 0.4; hard failure only above 0.7. Transcription is
#: language-aware: non-English source clips need their own ``language`` and a
#: transcript in that language (the bundled clips are English).
VC_WER_WARN = 0.4
VC_WER_FAIL = 0.7



# ---------------------------------------------------------------------------
# Family catalogue (mirrors config.gpu.yml entries)
# ---------------------------------------------------------------------------
def model(
    name: str,
    model: str,
    model_type: str,
    *,
    device: str = "cuda",
    language: str | None = None,
    voices: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    entry = {
        "name": name,
        "model": model,
        "model_type": model_type,
        "max_batch_size": 1,
        "device": device,
        "language": language,
        "voices": voices,
    }
    entry.update(extra or {})
    return entry


def _whisper_model():
    return model("faster-whisper", "Systran/faster-whisper-tiny", "FasterWhisper", device="cpu")


def _openvoice_v1_models():
    return [
        _whisper_model(),
        model(
            "ov1-synthesizer-en",
            "myshell-ai/openvoice",
            "OpenVoiceV1Synthesizer",
            language="EN",
            voices=["default"],
        ),
        model("ov1-tone-converter", "myshell-ai/openvoice", "OpenVoiceV1ToneConverter"),
        model("ov1-tone-converter-encoder", "myshell-ai/openvoice", "OpenVoiceV1ToneConverterEncoder"),
    ]


def _openvoice_v2_models():
    return [
        _whisper_model(),
        model(
            "ov2-synthesizer-en",
            "myshell-ai/MeloTTS-English-v3",
            "MeloTTSSynthesizer",
            language="EN",
            voices=["EN-Newest"],
        ),
        model("ov2-tone-converter", "myshell-ai/openvoicev2", "OpenVoiceV2ToneConverter"),
        model("ov2-tone-converter-encoder", "myshell-ai/openvoicev2", "OpenVoiceV2ToneConverterEncoder"),
    ]

FAMILY_SPECS: dict[str, dict] = {
    "chatterbox": {
        "models": [
            model("chatterbox", "resemble-ai/chatterbox", "ChatterboxTTS", extra={"watermark": True}),
            model("chatterbox_vc", "resemble-ai/chatterbox", "ChatterboxVC", extra={"watermark": True}),
            model("chatterbox_embedding", "resemble-ai/chatterbox", "ChatterboxEmbedding"),
        ],
        "group": "chatterbox-base",
        "cases": [
            {
                "name": "tts_bundled_voice",
                "kind": "tts",
                "model": "chatterbox",
                "text": "en_short",
                "label": "bundled voice, plain text",
            },
            {
                "name": "tts_clone_voice_a",
                "kind": "tts",
                "model": "chatterbox",
                "text": "en_numbers",
                "mode": "voice_cloning",
                "reference": "voice_a",
                "label": "synthetic reference: timbre porting only (source is not speech)",
            },
            {
                # Primary VC case: real speech on both sides. A voice conversion
                # must keep the *content* (the source sentence) intelligible while
                # adopting the target timbre; the WER gate below proves it.
                "name": "vc_speech_a_to_b",
                "kind": "vc",
                "model": "chatterbox_vc",
                "reference": "repo_voice_a",  # wanda4: "It's time to go home."
                "target": "repo_voice_b",  # wanda5 (real speech timbre)
                "transcript": "en_wanda4",
                "language": "en",
                "label": "real speech -> real speech timbre: content must stay intelligible (WER gate)",
            },
            {
                # Secondary VC case: real speech -> synthetic vowel timbre. This
                # isolates timbre porting (the target carries no speech content),
                # but the converted output should still say the source sentence.
                "name": "vc_speech_to_synth_voice",
                "kind": "vc",
                "model": "chatterbox_vc",
                "reference": "repo_voice_a",  # wanda4
                "target": "voice_b",  # synthetic vowel-like timbre
                "transcript": "en_wanda4",
                "language": "en",
                # Report WER but do not gate: the target has no speech content,
                # so the converted output is dominated by the synthetic timbre
                # and cannot be expected to transcribe. This case documents the
                # measurand mismatch that the old synthetic-source case hid;
                # only the real-speech pair above gates intelligibility.
                "wer_gate": False,
                "label": "real speech -> synthetic vowel timbre: isolates timbre porting (WER reported, not gated)",
            },
        ],
        # `batching:1-vs-N` axis (Phase 9-6): the AR runner decodes N requests in
        # shared steps. Four mixed-length texts so rows leave the batch at
        # different decode steps. The verdict is distributional (see
        # BATCHING_AXIS_STOCHASTIC); the seeded token equality is the e2e
        # parity harness (tests/e2e/tts/test_chatterbox_ar_parity.py).
        "batching_cases": [
            {
                "name": "batch_short",
                "kind": "tts",
                "model": "chatterbox",
                "text": "batch_1",
                "label": "short text",
            },
            {
                "name": "batch_medium",
                "kind": "tts",
                "model": "chatterbox",
                "text": "batch_2",
                "label": "medium text",
            },
            {
                "name": "batch_long",
                "kind": "tts",
                "model": "chatterbox",
                "text": "batch_3",
                "label": "long text",
            },
            {
                "name": "batch_shortest",
                "kind": "tts",
                "model": "chatterbox",
                "text": "batch_4",
                "label": "shortest text",
            },
            # Phase 11-1: the VC hop (one batched source-tokenize pass + the
            # 9-5 padded S3Gen vocode) rides this same family entry — the
            # allow-list is family-keyed and `chatterbox_vc` is a model inside
            # `chatterbox`. Four mixed-length sources (3.6 s / 4.4 s real speech,
            # two 8 s synthetic) against two shared targets, so the batched step
            # also composes with the target-voice cache.
            {
                "name": "vc_batch_a",
                "kind": "vc",
                "model": "chatterbox_vc",
                "reference": "repo_voice_a",
                "target": "repo_voice_b",
                "label": "3.6 s real speech -> 4.4 s real target",
            },
            {
                "name": "vc_batch_b",
                "kind": "vc",
                "model": "chatterbox_vc",
                "reference": "repo_voice_b",
                "target": "repo_voice_a",
                "label": "4.4 s real speech -> 3.6 s real target",
            },
            {
                "name": "vc_batch_c",
                "kind": "vc",
                "model": "chatterbox_vc",
                "reference": "voice_a",
                "target": "repo_voice_b",
                "label": "8 s synthetic source -> real target",
            },
            {
                "name": "vc_batch_d",
                "kind": "vc",
                "model": "chatterbox_vc",
                "reference": "voice_b",
                "target": "repo_voice_a",
                "label": "8 s synthetic source -> other real target",
            },
        ],
        # AR decode spans one engine step per token; the generic 20-step cap in
        # `_run_batching_worker` is far too small.
        "batching_max_steps": 4000,
    },
    "chatterbox_turbo": {
        "models": [
            model("chatterbox_turbo", "resemble-ai/chatterbox-turbo", "ChatterboxTurboTTS", extra={"watermark": True}),
        ],
        "group": None,  # turbo stays ungrouped (own meanflow S3Gen)
        "cases": [
            {"name": "tts_bundled_voice", "kind": "tts", "model": "chatterbox_turbo", "text": "en_short"},
        ],
        # `batching:1-vs-N` axis (Phase 9-6): Turbo's AR runner (one row per
        # request, no CFG). Distributional verdict; seeded token equality is the
        # e2e parity harness.
        "batching_cases": [
            {
                "name": "batch_short",
                "kind": "tts",
                "model": "chatterbox_turbo",
                "text": "batch_1",
                "label": "short text",
            },
            {
                "name": "batch_medium",
                "kind": "tts",
                "model": "chatterbox_turbo",
                "text": "batch_2",
                "label": "medium text",
            },
            {
                "name": "batch_long",
                "kind": "tts",
                "model": "chatterbox_turbo",
                "text": "batch_3",
                "label": "long text",
            },
            {
                "name": "batch_shortest",
                "kind": "tts",
                "model": "chatterbox_turbo",
                "text": "batch_4",
                "label": "shortest text",
            },
        ],
        "batching_max_steps": 4000,
    },
    "chatterbox_multilingual": {
        "models": [
            model(
                "chatterbox_multilingual",
                "resemble-ai/chatterbox",
                "ChatterboxMultilingualTTS",
                extra={"watermark": True},
            ),
        ],
        "group": "chatterbox-base",  # joins only on the hash evidence in factory.py
        "cases": [
            {
                "name": "tts_en",
                "kind": "tts",
                "model": "chatterbox_multilingual",
                "text": "en_short",
                "language": "en",
                "label": "English text, bundled English conditionals",
            },
            {
                # chatterbox-tts 0.1.6 `mtl_tts.generate(text, language_id,
                # audio_prompt_path, ...)`: `language_id` only selects
                # tokenization; accent/prosody come ENTIRELY from the
                # conditionals — the reference audio, or the bundled conds when
                # none is given (mtl_tts.py: prepare_conditionals only runs for
                # audio_prompt_path; `language_id` is passed to
                # tokenizer.text_to_tokens). The bundled conds are
                # English-accented, so German text with no reference is
                # accent transfer (de text / en accent) by construction — not a
                # native German voice. Named accordingly.
                "name": "tts_de_en_accent",
                "kind": "tts",
                "model": "chatterbox_multilingual",
                "text": "de_short",
                "language": "de",
                "label": "German text + bundled English conditionals: accent transfer, NOT native German",
            },
            {
                # Optional native-accent case: German text with a German
                # reference clip, which supplies the accent through the
                # conditionals. Skipped (and excluded from metrics) until a
                # user-provided clip exists — see fixtures provenance.
                "name": "tts_de_native",
                "kind": "tts",
                "model": "chatterbox_multilingual",
                "text": "de_short",
                "language": "de",
                "reference": "repo_voice_de",
                "optional": True,
                "skip_message": (
                    "native-accent German case skipped: provide tests/audio_fixtures/ref_voice_de.wav "
                    "— any short German speech clip, e.g. self-recorded; see provenance rules in "
                    "tests/audio_fixtures/__init__.py"
                ),
                "label": "German text + German reference: native accent (skipped until reference exists)",
            },
        ],
    },
    "openvoice_v1": {
        "models": _openvoice_v1_models(),
        "group": None,
        "cases": [
            {
                "name": "clone_voice_a",
                "kind": "tts",
                "model": "openvoice_v1",
                "text": "en_short",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "default",
                "reference": "repo_voice_a",
            },
            {
                "name": "clone_voice_b",
                "kind": "tts",
                "model": "openvoice_v1",
                "text": "en_numbers",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "default",
                "reference": "voice_b",
            },
        ],
        # `batching:1-vs-N` axis: four different-length texts through the same
        # clone toolchain, one engine step on the batched side.
        "batching_cases": [
            {
                "name": "batch_short",
                "kind": "tts",
                "model": "openvoice_v1",
                "text": "batch_1",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "default",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_medium",
                "kind": "tts",
                "model": "openvoice_v1",
                "text": "batch_2",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "default",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_long",
                "kind": "tts",
                "model": "openvoice_v1",
                "text": "batch_3",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "default",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_shortest",
                "kind": "tts",
                "model": "openvoice_v1",
                "text": "batch_4",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "default",
                "reference": "repo_voice_a",
            },
        ],
    },
    "openvoice_v2": {
        "models": _openvoice_v2_models(),
        "group": None,
        "cases": [
            {
                "name": "clone_voice_a",
                "kind": "tts",
                "model": "openvoice_v2",
                "text": "en_short",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "EN-Newest",
                "reference": "repo_voice_a",
            },
        ],
        # `batching:1-vs-N` axis (enabled in 8-2; the embed step batches first,
        # 8-3 adds the tone converter): the same four different-length texts as
        # openvoice_v1, so the full V2 clone toolchain (VAD → embed → MeloTTS →
        # tone-converter) is gated end to end.
        "batching_cases": [
            {
                "name": "batch_short",
                "kind": "tts",
                "model": "openvoice_v2",
                "text": "batch_1",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "EN-Newest",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_medium",
                "kind": "tts",
                "model": "openvoice_v2",
                "text": "batch_2",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "EN-Newest",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_long",
                "kind": "tts",
                "model": "openvoice_v2",
                "text": "batch_3",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "EN-Newest",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_shortest",
                "kind": "tts",
                "model": "openvoice_v2",
                "text": "batch_4",
                "mode": "voice_cloning",
                "language": "EN",
                "voice": "EN-Newest",
                "reference": "repo_voice_a",
            },
        ],
    },
    "melotts": {
        "models": [
            model(
                "ov2-synthesizer-en",
                "myshell-ai/MeloTTS-English-v3",
                "MeloTTSSynthesizer",
                language="EN",
                voices=["EN-Newest"],
            )
        ],
        "group": None,
        "cases": [
            {
                "name": "tts_en",
                "kind": "tts",
                "model": "ov2-synthesizer-en",
                "text": "en_short",
                "language": "EN",
                "voice": "EN-Newest",
            },
        ],
        "batching_cases": [
            {
                "name": "batch_short",
                "kind": "tts",
                "model": "ov2-synthesizer-en",
                "text": "batch_1",
                "language": "EN",
                "voice": "EN-Newest",
            },
            {
                "name": "batch_medium",
                "kind": "tts",
                "model": "ov2-synthesizer-en",
                "text": "batch_2",
                "language": "EN",
                "voice": "EN-Newest",
            },
            {
                "name": "batch_long",
                "kind": "tts",
                "model": "ov2-synthesizer-en",
                "text": "batch_3",
                "language": "EN",
                "voice": "EN-Newest",
            },
            {
                "name": "batch_shortest",
                "kind": "tts",
                "model": "ov2-synthesizer-en",
                "text": "batch_4",
                "language": "EN",
                "voice": "EN-Newest",
            },
        ],
    },
    "voicefixer": {
        "models": [
            model("voicefixer-restorer", "jlmarrugom/voice_fixer", "VoiceFixerRestorer"),
            model("voicefixer-vocoder", "jlmarrugom/voice_fixer", "VoiceFixerVocoder"),
        ],
        "group": None,
        "cases": [
            {"name": "restore_voice_a", "kind": "restore", "model": "voicefixer", "reference": "voice_a"},
        ],
        # `batching:1-vs-N` axis (enabled in 8-4): four clips of different
        # lengths/rates so the padded mel-frame batch really has padding to
        # trim. `restore_voice_a` is the primary case; the others exist because
        # a gate needs at least two distinct inputs to compare anything.
        "batching_cases": [
            {
                "name": "restore_voice_a",
                "kind": "restore",
                "model": "voicefixer",
                "reference": "voice_a",
                "label": "8 s synthetic clip",
            },
            {
                "name": "restore_voice_b",
                "kind": "restore",
                "model": "voicefixer",
                "reference": "voice_b",
                "label": "second 8 s synthetic clip",
            },
            {
                "name": "restore_repo_a",
                "kind": "restore",
                "model": "voicefixer",
                "reference": "repo_voice_a",
                "label": "3.6 s real-speech clip",
            },
            {
                "name": "restore_repo_b",
                "kind": "restore",
                "model": "voicefixer",
                "reference": "repo_voice_b",
                "label": "second real-speech clip",
            },
        ],
    },
    "harmonyspeech": {
        "models": [
            model("hs1-encoder", "harmony-ai/harmony-speech-v1", "HarmonySpeechEncoder", device="cpu"),
            model("hs1-synthesizer", "harmony-ai/harmony-speech-v1", "HarmonySpeechSynthesizer", device="cpu"),
            model("hs1-vocoder", "harmony-ai/harmony-speech-v1", "HarmonySpeechVocoder", device="cpu"),
        ],
        "group": None,
        "cases": [
            {
                "name": "clone_voice_a",
                "kind": "tts",
                "model": "harmonyspeech",
                "text": "en_short",
                "mode": "voice_cloning",
                "reference": "repo_voice_a",
            },
        ],
        # `batching:1-vs-N` axis (enabled in 8-4): mixed-length clone requests
        # so the encoder/synthesizer/vocoder steps all see a padded batch.
        "batching_cases": [
            {
                "name": "batch_short",
                "kind": "tts",
                "model": "harmonyspeech",
                "text": "batch_1",
                "mode": "voice_cloning",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_medium",
                "kind": "tts",
                "model": "harmonyspeech",
                "text": "batch_2",
                "mode": "voice_cloning",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_long",
                "kind": "tts",
                "model": "harmonyspeech",
                "text": "batch_3",
                "mode": "voice_cloning",
                "reference": "repo_voice_a",
            },
            {
                "name": "batch_shortest",
                "kind": "tts",
                "model": "harmonyspeech",
                "text": "batch_4",
                "mode": "voice_cloning",
                "reference": "repo_voice_a",
            },
        ],
    },
    "kittentts": {
        "models": [
            model(
                "kitten-tts-mini",
                "KittenML/kitten-tts-mini-0.8",
                "KittenTTSSynthesizer",
                device="cpu",
                voices=["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"],
            )
        ],
        "group": None,
        "cases": [
            {
                "name": "tts_en",
                "kind": "tts",
                "model": "kitten-tts-mini",
                "text": "en_short",
                "voice": "Bella",
                "language": "default",
            }
        ],
    },
}

# Canonical alignment rate for metric computation.
METRIC_SR = 24000


# ---------------------------------------------------------------------------
# Axis definition
# ---------------------------------------------------------------------------
@dataclass
class Axis:
    """A change axis with a label for each side."""

    kind: str  # "dtype" | "sharing" | "baseline" | "batching"
    side_a: str
    side_b: str
    #: Autocast precision for ``dtype:fp32-vs-autocast`` (``None`` for every
    #: other axis); the CLI's ``--s3gen-autocast`` overrides the default.
    precision: str | None = None

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.side_a}-vs-{self.side_b}"

    @property
    def identity(self) -> bool:
        return self.side_a == self.side_b

    @property
    def autocast(self) -> bool:
        """True for the Phase 11-2 ``dtype:fp32-vs-autocast`` experiment axis."""
        return self.kind == "dtype" and self.precision is not None


def parse_axis(text: str) -> Axis:
    try:
        kind, pair = text.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"axis must look like 'dtype:float32-vs-float16', got {text!r}") from exc
    if "-vs-" not in pair:
        raise ValueError(f"axis must contain '-vs-', got {text!r}")
    a, b = pair.split("-vs-", 1)
    if kind not in {"dtype", "sharing", "baseline", "batching", "longform", "determinism"}:
        raise ValueError(
            f"unknown axis kind '{kind}' (expected dtype, sharing, batching, longform, "
            "determinism or baseline)"
        )
    if kind == "batching":
        for side in (a, b):
            if not side.isdigit() or int(side) < 1:
                raise ValueError(f"batching axis sides must be batch sizes >= 1, got {side!r}")
    if kind == "longform":
        if (a, b) != ("off", "on"):
            raise ValueError(f"the long-form axis is 'longform:off-vs-on', got {text!r}")
    if kind == "determinism":
        if (a, b) != ("run1", "run2"):
            raise ValueError(f"the determinism axis is 'determinism:run1-vs-run2', got {text!r}")
    if kind == "dtype" and {a, b} == set(AUTOCAST_AXIS_SIDES):
        # Phase 11-2 experiment axis. Side A must be the fp32 reference so the
        # report's "A vs B" reads as "baseline vs experiment".
        if a != "fp32":
            raise ValueError(
                f"the autocast axis is 'dtype:fp32-vs-autocast', got {text!r} "
                "(fp32 must be side A)"
            )
        return Axis(kind=kind, side_a=a, side_b=b, precision="bfloat16")
    return Axis(kind=kind, side_a=a, side_b=b)


def side_settings(axis: Axis, side: str) -> dict:
    """Translate one side of the axis into config knobs.

    ``dtype`` value is passed verbatim to ``ModelConfig.dtype``; ``sharing``
    values ``dedup`` / ``dup`` select whether grouped families get a
    ``backbone_group``; ``batching`` values are the per-model ``max_batch_size``
    of that side (1 = legacy per-item loop, N = padded batch).

    The Phase 11-2 ``dtype:fp32-vs-autocast`` axis keeps ``ModelConfig.dtype``
    at float32 on **both** sides (so the T3 backbone is identical and the seeded
    token stream cannot diverge) and instead flips the ``HSE_S3GEN_AUTOCAST``
    environment gate per worker: ``off`` for side A, the chosen precision for
    side B. A fixed request seed makes the vocode noise identical on both sides.
    """
    if axis.autocast:
        return {
            "dtype": "float32",
            "share": True,
            "watermark": True,
            "s3gen_autocast": "off" if side == "fp32" else axis.precision,
            "seed": 1234,
        }
    if axis.kind == "dtype":
        return {"dtype": side, "share": True, "watermark": True}
    if axis.kind == "sharing":
        # PerthNet's conv kernels are not bit-reproducible across processes
        # (cuDNN algorithm choice varies with free VRAM), so the identity gate
        # runs with the dummy watermarker: it tests the synthesis path, which is
        # what sharing touches. The dtype axis keeps watermarking enabled.
        return {"dtype": "float32", "share": side == "dedup", "watermark": False}
    if axis.kind == "batching":
        return {
            "dtype": "float32",
            "share": True,
            "watermark": True,
            "max_batch_size": int(side),
            "batched_inference": int(side) > 1,
            # Diagnostic mode: zero the stochastic scales and disable TF32 so the
            # comparison isolates padding/unpacking instead of sampling noise or
            # TF32's batch-shape-dependent kernel choice.
            "deterministic": True,
        }
    if axis.kind == "longform":
        # Short text with split_text on must take the untouched single-shot path.
        # Watermark off: Perth's conv kernels are not bit-reproducible across
        # processes, which would mask the dispatcher identity we are gating.
        return {
            "dtype": "float32",
            "share": True,
            "watermark": False,
            "split_text": side == "on",
            "chunk_size": None,
            "seed": LONGFORM_AXIS_SEED,
            # Disable TF32 so batch-shape-dependent kernel choice cannot add
            # cross-process noise on top of the dispatcher identity gate.
            "deterministic": True,
        }
    if axis.kind == "determinism":
        # Both sides are the same chunked request; only the run index differs.
        return {
            "dtype": "float32",
            "share": True,
            "watermark": False,
            "split_text": True,
            "chunk_size": LONGFORM_AXIS_CHUNK_SIZE,
            "crossfade_ms": 0,
            "sentence_pause_ms": 0,
            "seed": LONGFORM_AXIS_SEED,
            "deterministic": True,
        }
    return {"dtype": side, "share": True, "watermark": True}


# ---------------------------------------------------------------------------
# Worker: boot one engine, run the family's cases, write wavs + manifest
# ---------------------------------------------------------------------------
def _build_model_configs(family: str, settings: dict, use_gpu: bool):
    from harmonyspeech.common.config import DeviceConfig, ModelConfig

    spec = FAMILY_SPECS[family]
    configs = []
    for entry in spec["models"]:
        device = entry["device"]
        if not use_gpu:
            device = "cpu"
        kwargs = {
            "name": entry["name"],
            "model": entry["model"],
            "model_type": entry["model_type"],
            "max_batch_size": entry["max_batch_size"],
            "device_config": DeviceConfig(device=device),
            "dtype": settings["dtype"],
        }
        if entry.get("language"):
            kwargs["language"] = entry["language"]
        if entry.get("voices"):
            kwargs["voices"] = entry["voices"]
        if entry.get("watermark") is not None:
            kwargs["watermark"] = bool(entry["watermark"]) and bool(settings.get("watermark", True))
        if spec.get("group") and settings["share"]:
            kwargs["backbone_group"] = spec["group"]
        if settings.get("max_batch_size") is not None:
            # Batching axis: the axis *is* the batch size, so it wins over the
            # family catalogue's per-entry value.
            kwargs["max_batch_size"] = int(settings["max_batch_size"])
        if settings.get("batched_inference") is not None:
            kwargs["batched_inference"] = bool(settings["batched_inference"])
        configs.append(ModelConfig(**kwargs))
    return configs


def _decode_response_audio(response) -> tuple[np.ndarray, int]:
    import soundfile as sf

    from harmonyspeech.endpoints.openai.protocol import ErrorResponse

    if isinstance(response, ErrorResponse) or not getattr(response, "data", None):
        message = getattr(response, "message", None) or getattr(response, "error", None) or response
        raise RuntimeError(f"request failed: {message}")
    raw = base64.b64decode(response.data)
    samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32), int(rate)


def reference_available(name: str) -> bool:
    """Whether a logical reference clip exists (delegates to the fixtures)."""
    from tests.audio_fixtures import reference_available as _available

    return _available(name)


def partition_optional_cases(cases: list[dict], available=None) -> tuple[list[dict], list[dict]]:
    """Split cases into runnable and skipped.

    A case is skipped when either:

    * it carries an explicit ``skip_reason`` — the case-selection layer proved
      the case cannot gate anything (e.g. a multi-chunk text on the long-form
      identity axis), so running it would produce a misleading FAIL; or
    * it is marked ``optional`` and its ``reference`` clip is absent. It never
      reaches a worker, so it cannot fail, and it is reported separately.

    ``available`` is injectable for tests.
    """
    available = available or reference_available
    runnable: list[dict] = []
    skipped: list[dict] = []
    for case in cases:
        if case.get("skip_reason") or (
            case.get("optional")
            and case.get("reference")
            and not available(case["reference"])
        ):
            skipped.append(case)
        else:
            runnable.append(case)
    return runnable, skipped


# ---------------------------------------------------------------------------
# Voice-conversion intelligibility (WER)
# ---------------------------------------------------------------------------
_WHISPER_MODELS: dict[str, object] = {}


def _get_whisper():
    """Load the harness's faster-whisper model once (CPU is enough for tiny)."""
    from faster_whisper import WhisperModel

    model_id = _whisper_model()["model"]
    if model_id not in _WHISPER_MODELS:
        _WHISPER_MODELS[model_id] = WhisperModel(model_id, device="cpu", compute_type="int8")
    return _WHISPER_MODELS[model_id]


def transcribe_wav(path: str, language: str | None = "en") -> str:
    """Transcribe a rendered clip with the resident faster-whisper model."""
    model = _get_whisper()
    segments, _info = model.transcribe(
        path,
        language=language or None,
        beam_size=5,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def _wer_tokens(text: str) -> list[str]:
    import re

    # Case/punctuation-insensitive words; keep apostrophes so contractions
    # ("it's") stay one token.
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return cleaned.split()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word error rate (Levenshtein over normalised tokens), ``errors/|ref|``.

    An empty reference is a perfect match only when the hypothesis is empty.
    """
    ref = _wer_tokens(reference)
    hyp = _wer_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def score_vc_intelligibility(
    case: dict,
    path_a: str,
    path_b: str,
    transcribe=transcribe_wav,
) -> dict | None:
    """Transcribe both VC outputs and score WER against the known transcript.

    Returns a reported metric (``warn`` flag) and a soft gate (``failed`` only
    above :data:`VC_WER_FAIL`). ``transcribe`` is injectable for unit tests.
    """
    from tests.audio_fixtures import TEXTS

    transcript_key = case.get("transcript")
    if not transcript_key:
        return None
    reference_text = TEXTS.get(transcript_key, transcript_key)
    language = case.get("language", "en")

    scores = {
        "reference_text": reference_text,
        "language": language,
        "thresholds": {"warn": VC_WER_WARN, "fail": VC_WER_FAIL},
    }
    for side, path in (("a", path_a), ("b", path_b)):
        hypothesis = transcribe(path, language)
        scores[f"hypothesis_{side}"] = hypothesis
        scores[f"wer_{side}"] = round(compute_wer(reference_text, hypothesis), 4)

    worst = max(scores["wer_a"], scores["wer_b"])
    scores["wer"] = round(worst, 4)
    scores["warn"] = worst > VC_WER_WARN
    scores["failed"] = worst > VC_WER_FAIL
    scores["gated"] = bool(case.get("wer_gate", True))
    return scores


def _build_batching_request_input(case: dict, request_id: str, text_table: dict, deterministic: bool):
    """Build the wire ``RequestInput`` for one batching-gate case.

    The batching worker adds every case to the scheduler before stepping, so a
    family's gate may mix request kinds. Each kind maps to its own
    ``RequestInput`` — and therefore its own artifact flow, exactly what the
    padded-batch step sees in production:

    * ``tts`` → ``TextToSpeechRequestInput`` (mode/reference/voice from the case);
    * ``vc`` → ``VoiceConversionRequestInput`` (source + target audio);
    * ``restore`` → ``AudioConversionRequestInput`` (source audio only).

    ``deterministic`` zeroes the stochastic generation scales, so both sides of
    the axis compute the same function and only padding/unpacking can differ.
    """
    from harmonyspeech.common.inputs import (
        AudioConversionRequestInput,
        TextToSpeechRequestInput,
        VoiceConversionRequestInput,
    )
    from harmonyspeech.endpoints.openai.protocol import (
        AudioConversionRequest,
        GenerationOptions,
        TextToSpeechRequest,
        VoiceConversionRequest,
    )
    from tests.audio_fixtures import reference_wav_b64

    kind = case.get("kind", "tts")
    if kind == "tts":
        options = None
        if deterministic:
            # Zero every stochastic scale: both sides then compute the same
            # deterministic function, so any difference is padding/unpacking.
            options = GenerationOptions(noise_scale=0.0, noise_scale_w=0.0, seed=0)
        request = TextToSpeechRequest(
            model=case["model"],
            input=_case_text(case, text_table),
            mode=case.get("mode", "single_speaker_tts"),
            input_audio=reference_wav_b64(case["reference"]) if case.get("reference") else None,
            language=case.get("language"),
            voice=case.get("voice"),
            generation_options=options,
        )
        return TextToSpeechRequestInput.from_openai(request_id=request_id, request=request)
    if kind == "vc":
        # Phase 11-1: a seed makes the vocode noise identical on both sides
        # (the batched path draws it from a per-row generator seeded from the
        # request; the loop seeds the global RNG the same way).
        options = GenerationOptions(seed=0) if deterministic else None
        request = VoiceConversionRequest(
            model=case["model"],
            source_audio=reference_wav_b64(case["reference"]),
            target_audio=reference_wav_b64(case["target"]),
            generation_options=options,
        )
        return VoiceConversionRequestInput.from_openai(request_id=request_id, request=request)
    if kind == "restore":
        request = AudioConversionRequest(
            model=case["model"], source_audio=reference_wav_b64(case["reference"])
        )
        return AudioConversionRequestInput.from_openai(request_id=request_id, request=request)
    raise ValueError(f"batching gate: unknown case kind {kind!r}")


def _run_batching_worker(job, engine, text_table) -> dict:
    """Drive one engine step at a time with every case in a single batch.

    The batching axis must compare *one step* of N requests against N steps of
    one request, so the worker bypasses the background loop: it adds every case
    to the scheduler first and then drives ``step_async`` until all of them
    finish. With ``max_batch_size = N`` the explicit model receives all N
    requests in the same step; with ``max_batch_size = 1`` it receives one per
    step — the two sides of the axis.
    """
    import asyncio
    import base64
    import io

    import soundfile as sf

    from harmonyspeech.endpoints.openai.wire import from_wire_audio, to_wire_audio

    deterministic = bool(job["settings"].get("deterministic"))
    cases = {case["name"]: case for case in job["cases"]}
    results: dict[str, dict] = {}

    async def drive() -> None:
        pending: list[str] = []
        for name, case in cases.items():
            request_id = f"batch-gate-{name}"
            request_data = _build_batching_request_input(case, request_id, text_table, deterministic)
            artifacts = from_wire_audio(request_data)
            await engine.engine.add_request_async(
                request_id=request_id, request_data=request_data, artifacts=artifacts
            )
            pending.append(request_id)

        seen: set[str] = set()
        steps = 0
        max_steps = int(job.get("batching_max_steps", 20))
        while pending and steps < max_steps:
            steps += 1
            outputs, _forwarded = await engine.engine.step_async()
            for output in outputs:
                if output.request_id not in pending:
                    continue
                if output.finish_reason == "error":
                    results[output.request_id] = {"error": output.error or "engine error"}
                    pending.remove(output.request_id)
                    continue
                if output.finish_reason in ("stop", None):
                    try:
                        encoded = to_wire_audio(output, None)
                        raw = base64.b64decode(encoded.data)
                        samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
                        name = output.request_id.replace("batch-gate-", "", 1)
                        wav_path = os.path.join(job["out_dir"], f"{name}.wav")
                        sf.write(wav_path, samples, rate, subtype="PCM_16")
                        results[output.request_id] = {
                            "wav": wav_path,
                            "sample_rate": int(rate),
                            "samples": int(samples.size),
                            "seconds": round(float(samples.size) / rate, 4),
                            "steps": steps,
                        }
                    except Exception as exc:  # noqa: BLE001 - reported per case
                        results[output.request_id] = {"error": f"{type(exc).__name__}: {exc}"}
                    pending.remove(output.request_id)
                # "forwarded" keeps the request in the pipeline.
            seen.update(outputs)
        for request_id in pending:
            results[request_id] = {"error": "request did not finish"}

    asyncio.run(drive())

    manifest_cases = {}
    for name in cases:
        request_id = f"batch-gate-{name}"
        entry = results.get(request_id, {"error": "missing output"})
        manifest_cases[name] = entry
        state = "ERROR" if "error" in entry else f"{entry['seconds']}s audio in {entry['steps']} step(s)"
        print(f"[worker:{job['side']}] {job['family']}/{name}: {state}", flush=True)
    return manifest_cases


def _case_text(case: dict, text_table: dict) -> str:
    """Resolve a case's text, preferring an inline ``text_literal`` (Phase 12-4)."""
    if "text_literal" in case:
        return case["text_literal"]
    return text_table[case["text"]]


def _request_generation_options(settings: dict):
    """Build the request-level ``GenerationOptions`` for a worker side.

    Phase 12-4 adds ``split_text``/``chunk_size`` (and the stitch timings) so the
    long-form axes can drive the real serving dispatcher. When none of the
    long-form settings are present this reduces to the pre-12 behaviour: a seed
    only (the autocast axis) or ``None``.
    """
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions

    request_seed = settings.get("seed")
    split_text = bool(settings.get("split_text", False))
    chunk_size = settings.get("chunk_size")
    if request_seed is None and not split_text and chunk_size is None:
        return None
    kwargs: dict = {}
    if request_seed is not None:
        kwargs["seed"] = request_seed
    if split_text:
        kwargs["split_text"] = True
    if chunk_size is not None:
        kwargs["chunk_size"] = int(chunk_size)
    if settings.get("crossfade_ms") is not None:
        kwargs["crossfade_ms"] = int(settings["crossfade_ms"])
    if settings.get("sentence_pause_ms") is not None:
        kwargs["sentence_pause_ms"] = int(settings["sentence_pause_ms"])
    return GenerationOptions(**kwargs)


def run_worker(job_path: str) -> int:
    """Boot one engine for one side of the axis and run the family's cases."""
    import asyncio
    import random

    import torch

    from harmonyspeech.endpoints.openai.protocol import (
        AudioConversionRequest,
        TextToSpeechRequest,
        VoiceConversionRequest,
    )
    from harmonyspeech.endpoints.openai.serving_audio_conversion import OpenAIServingAudioConversion
    from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
    from harmonyspeech.endpoints.openai.serving_voice_conversion import OpenAIServingVoiceConversion
    from harmonyspeech.engine.args_tools import AsyncEngineArgs
    from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech
    from tests.audio_fixtures import reference_wav_b64

    with open(job_path, encoding="utf-8") as handle:
        job = json.load(handle)

    batching_axis = job["settings"].get("batching") is not None
    if job["settings"].get("deterministic"):
        # Diagnostic mode for the batching axis: TF32's conv/matmul kernel choice
        # depends on the batch shape, which would swamp the ~1e-4 padding signal.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    model_configs = _build_model_configs(job["family"], job["settings"], job["use_gpu"])
    engine_args = AsyncEngineArgs(disable_log_stats=True, disable_log_requests=True)
    from harmonyspeech.common.config import EngineConfig

    engine_config = EngineConfig(
        model_configs=model_configs,
        ar_cache_budget_bytes=int(job["settings"].get("ar_cache_budget_bytes", 1073741824)),
    )
    engine = AsyncHarmonySpeech.from_engine_args_and_config(
        engine_args, engine_config, start_engine_loop=not batching_axis
    )
    serving_tts = OpenAIServingTextToSpeech(
        engine, OpenAIServingTextToSpeech.models_from_config(engine_config.model_configs)
    )
    serving_vc = OpenAIServingVoiceConversion(
        engine, OpenAIServingVoiceConversion.models_from_config(engine_config.model_configs)
    )
    serving_ac = OpenAIServingAudioConversion(
        engine, OpenAIServingAudioConversion.models_from_config(engine_config.model_configs)
    )

    class _Request:
        is_disconnected = staticmethod(lambda: asyncio.sleep(0, result=False))

    os.makedirs(job["out_dir"], exist_ok=True)
    manifest = {"family": job["family"], "side": job["side"], "cases": {}}

    text_table = __import__("tests.audio_fixtures", fromlist=["TEXTS"]).TEXTS

    if batching_axis:
        manifest["cases"] = _run_batching_worker(job, engine, text_table)
        manifest["vram"] = engine.engine.get_vram_stats() if hasattr(engine, "engine") else None
        with open(job["manifest"], "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return 0

    for case in job["cases"]:
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        # Phase 11-2: the autocast axis carries an explicit request seed so the
        # vocode noise is identical on both sides (the sequential TTS loop still
        # uses the global seed above; VC/AR paths honour the request seed).
        # Phase 12-4: the long-form axes additionally drive split_text/chunk_size
        # through the real serving dispatcher.
        options = _request_generation_options(job["settings"])
        started = time.time()
        try:
            kind = case["kind"]
            if kind == "tts":
                request = TextToSpeechRequest(
                    model=case["model"],
                    input=_case_text(case, text_table),
                    mode=case.get("mode", "single_speaker_tts"),
                    input_audio=reference_wav_b64(case["reference"]) if case.get("reference") else None,
                    language=case.get("language"),
                    voice=case.get("voice"),
                    generation_options=options,
                )
                response = asyncio.run(serving_tts.create_text_to_speech(request, _Request()))
            elif kind == "vc":
                request = VoiceConversionRequest(
                    model=case["model"],
                    source_audio=reference_wav_b64(case["reference"]),
                    target_audio=reference_wav_b64(case["target"]),
                    generation_options=options,
                )
                response = asyncio.run(serving_vc.convert_voice(request, _Request()))
            elif kind == "restore":
                request = AudioConversionRequest(
                    model=case["model"], source_audio=reference_wav_b64(case["reference"])
                )
                response = asyncio.run(serving_ac.convert_audio(request, _Request()))
            else:
                raise ValueError(f"unknown case kind {kind!r}")
            samples, rate = _decode_response_audio(response)
        except Exception as exc:  # noqa: BLE001 - reported per case
            manifest["cases"][case["name"]] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[worker:{job['side']}] {job['family']}/{case['name']} FAILED: {exc}", flush=True)
            continue

        import soundfile as sf

        wav_path = os.path.join(job["out_dir"], f"{case['name']}.wav")
        sf.write(wav_path, samples, rate, subtype="PCM_16")
        manifest["cases"][case["name"]] = {
            "wav": wav_path,
            "sample_rate": rate,
            "samples": int(samples.size),
            "seconds": round(float(samples.size) / rate, 4),
            "elapsed_s": round(time.time() - started, 2),
        }
        print(
            f"[worker:{job['side']}] {job['family']}/{case['name']}: "
            f"{samples.size / rate:.2f}s audio in {time.time() - started:.1f}s",
            flush=True,
        )

    manifest["vram"] = engine.engine.get_vram_stats() if hasattr(engine, "engine") else None
    with open(job["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _load_wav(path: str) -> tuple[np.ndarray, int]:
    import librosa

    samples, rate = librosa.load(path, sr=None, mono=True)
    return np.asarray(samples, dtype=np.float32), int(rate)


def _resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return samples
    import librosa

    return librosa.resample(samples, orig_sr=rate, target_sr=target)


def _logmel(samples: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max)


def compute_metrics(path_a: str, path_b: str, kind: str) -> dict:
    """Compare two rendered clips and return metrics plus a pass flag."""
    a, sr_a = _load_wav(path_a)
    b, sr_b = _load_wav(path_b)
    sr = METRIC_SR
    a = _resample(a, sr_a, sr)
    b = _resample(b, sr_b, sr)

    rms_a = float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0
    rms_b = float(np.sqrt(np.mean(np.square(b)))) if b.size else 0.0
    duration_pct = abs(a.size - b.size) / max(a.size, b.size, 1) * 100.0
    rms_pct = abs(rms_a - rms_b) / max(rms_a, 1e-6) * 100.0

    metrics = {
        "duration_a": round(a.size / sr, 4),
        "duration_b": round(b.size / sr, 4),
        "duration_pct": round(duration_pct, 3),
        "rms_a": round(rms_a, 6),
        "rms_b": round(rms_b, 6),
        "rms_pct": round(rms_pct, 3),
    }

    mel_a = _logmel(a, sr)
    mel_b = _logmel(b, sr)
    frames = min(mel_a.shape[1], mel_b.shape[1])
    mel_mse = float(np.mean((mel_a[:, :frames] - mel_b[:, :frames]) ** 2))

    # Long-term average spectrum: robust to a different sampled realisation.
    ltas_a = mel_a.mean(axis=1)
    ltas_b = mel_b.mean(axis=1)
    ltas_rmse = float(np.sqrt(np.mean((ltas_a - ltas_b) ** 2)))
    ltas_corr = float(np.corrcoef(ltas_a, ltas_b)[0, 1]) if frames > 1 else 0.0

    common = min(a.size, b.size)
    corr = float(np.corrcoef(a[:common], b[:common])[0, 1]) if common > 1 else 0.0
    identical = a.size == b.size and bool(np.array_equal(a, b))
    max_abs_diff = float(np.max(np.abs(a[:common] - b[:common]))) if common else 0.0

    metrics.update(
        {
            "logmel_mse": round(mel_mse, 6),
            "ltas_rmse_db": round(ltas_rmse, 4),
            "ltas_corr": round(ltas_corr, 4),
            "corr": round(corr, 4),
            "identical": identical,
            "max_abs_diff": round(max_abs_diff, 8),
        }
    )

    thresholds = {**THRESHOLDS[kind], **FAMILY_THRESHOLDS.get(kind, {})}
    failures = []
    if duration_pct > thresholds["duration_pct"]:
        failures.append(f"duration {duration_pct:.2f}% > {thresholds['duration_pct']}%")
    if rms_pct > thresholds["rms_pct"]:
        failures.append(f"rms {rms_pct:.2f}% > {thresholds['rms_pct']}%")
    if kind == "deterministic":
        if mel_mse > thresholds["logmel_mse"]:
            failures.append(f"logmel mse {mel_mse:.3f} > {thresholds['logmel_mse']}")
        if corr < thresholds["corr_min"]:
            failures.append(f"corr {corr:.3f} < {thresholds['corr_min']}")
    else:
        if ltas_rmse > thresholds["ltas_rmse_db"]:
            failures.append(f"ltas rmse {ltas_rmse:.2f}dB > {thresholds['ltas_rmse_db']}dB")

    metrics["thresholds"] = thresholds
    metrics["failures"] = failures
    metrics["passed"] = not failures
    return metrics


#: Weight-sharing gate. Deduplication must not alter computation. Sample-exact
#: equality is asserted in-process by
#: tests/e2e/tts/test_chatterbox_sharing_identity.py; the cross-process harness
#: gate additionally tolerates last-bit CUDA kernel selection noise (measured
#: max |Δsample| ≈ 1.5e-4, −76 dBFS — four orders of magnitude below any wiring
#: change) while requiring identical length and an inaudible mel distance.
SHARING_MAX_ABS_DIFF = 1e-3
SHARING_LOG_MEL_MSE = 1e-2


def verdict_for_axis(axis: Axis, metrics: dict, family: str | None = None) -> tuple[bool, list[str]]:
    """Apply axis-specific rules on top of the family metrics."""
    if axis.identity:
        # Same configuration twice must be bit-identical.
        if not metrics["identical"]:
            return False, ["not bit-identical (identity axis must not change a sample)"]
        return True, []

    if axis.autocast:
        # Phase 11-2: mixed precision changes the arithmetic, not the
        # computation. A duration change means the token stream moved (the AR
        # backbone is fp32 on both sides), so it fails as hard as a spectral
        # change does.
        failures = []
        if metrics["duration_pct"] > AUTOCAST_DURATION_PCT:
            failures.append(
                f"duration changed ({metrics['duration_pct']}%) — autocast altered the token stream"
            )
        if metrics["rms_pct"] > AUTOCAST_RMS_PCT:
            failures.append(f"rms {metrics['rms_pct']:.2f}% > {AUTOCAST_RMS_PCT}%")
        if metrics["logmel_mse"] > AUTOCAST_LOG_MEL_MSE:
            failures.append(
                f"logmel mse {metrics['logmel_mse']} > {AUTOCAST_LOG_MEL_MSE} (audible degradation)"
            )
        if metrics["corr"] < AUTOCAST_CORR_MIN:
            failures.append(f"corr {metrics['corr']} < {AUTOCAST_CORR_MIN}")
        return (not failures), failures

    if axis.kind in {"longform", "determinism"}:
        # Phase 12-4. The dispatcher must not alter the single-shot path, and
        # the seeded chunked path must reproduce its stitched bytes. Length is a
        # hard invariant; the sample comparison is bit-exact for deterministic
        # families and bounded by the documented cross-process CUDA noise floor
        # for stochastic ones (the same limit the sharing axis records).
        #
        # A family whose *model* samples on every inference
        # (:data:`MODEL_SAMPLES_EVERY_RUN`) cannot be sample-compared at all.
        # Duration stays a hard invariant — it proves the chunk count, pairing
        # and stitch length did not move — and the spectral/level distribution
        # must stay inside the stochastic bounds. A crossed or duplicated chunk
        # changes the stitched length (chunks differ in length) and the LTAS, so
        # this gate still catches the failure it exists for.
        failures = []
        if metrics["duration_pct"] != 0.0:
            failures.append(
                f"duration changed ({metrics['duration_pct']}%) — the chunking path altered the frame count"
            )
        if metrics["identical"]:
            return True, []
        if family and family in MODEL_SAMPLES_EVERY_RUN:
            stochastic = THRESHOLDS["stochastic"]
            if metrics["rms_pct"] > stochastic["rms_pct"]:
                failures.append(f"rms {metrics['rms_pct']:.2f}% > {stochastic['rms_pct']}%")
            if metrics["ltas_rmse_db"] > stochastic["ltas_rmse_db"]:
                failures.append(
                    f"ltas rmse {metrics['ltas_rmse_db']:.2f}dB > {stochastic['ltas_rmse_db']}dB"
                )
            return (not failures), failures
        family_kind = _family_kind(family) if family else "deterministic"
        if family_kind == "deterministic":
            failures.append(
                "not bit-identical (the chunking dispatcher must leave the single-shot / "
                "seeded chunked path byte-identical)"
            )
        else:
            if metrics["max_abs_diff"] > SHARING_MAX_ABS_DIFF:
                failures.append(
                    f"max |Δsample| {metrics['max_abs_diff']} > {SHARING_MAX_ABS_DIFF} "
                    "(beyond the documented CUDA kernel-noise floor)"
                )
            if metrics["logmel_mse"] > SHARING_LOG_MEL_MSE:
                failures.append(f"logmel mse {metrics['logmel_mse']} > {SHARING_LOG_MEL_MSE}")
        return (not failures), failures

    if axis.kind == "sharing":
        failures = []
        if metrics["duration_pct"] != 0.0:
            failures.append(f"duration changed ({metrics['duration_pct']}%) — sharing altered the token stream")
        if metrics["max_abs_diff"] > SHARING_MAX_ABS_DIFF:
            failures.append(
                f"max |Δsample| {metrics['max_abs_diff']} > {SHARING_MAX_ABS_DIFF} (sharing changed the output)"
            )
        if metrics["logmel_mse"] > SHARING_LOG_MEL_MSE:
            failures.append(f"logmel mse {metrics['logmel_mse']} > {SHARING_LOG_MEL_MSE}")
        return (not failures), failures

    if axis.kind == "batching":
        if family in BATCHING_AXIS_STOCHASTIC:
            # AR decode: the token stream is seeded, but the vocode noise differs
            # by construction between the loop (global RNG) and the runner
            # (per-row generators). Assert the distribution, not the samples.
            return metrics["passed"], list(metrics["failures"])
        # Padded batching is a pure reshaping of the same computation (noise
        # disabled on both sides), so a sample inside a batch must match the
        # solo run up to GPU accumulation-order noise.
        failures = []
        if metrics["duration_pct"] > BATCHING_DURATION_PCT:
            failures.append(
                f"duration changed ({metrics['duration_pct']}%) — batching altered the frame count"
            )
        if metrics["max_abs_diff"] > BATCHING_MAX_ABS_DIFF:
            failures.append(
                f"max |Δsample| {metrics['max_abs_diff']} > {BATCHING_MAX_ABS_DIFF} (padding/unpacking bug)"
            )
        if metrics["logmel_mse"] > BATCHING_LOG_MEL_MSE:
            failures.append(f"logmel mse {metrics['logmel_mse']} > {BATCHING_LOG_MEL_MSE}")
        if metrics["corr"] < BATCHING_CORR_MIN:
            failures.append(f"corr {metrics['corr']} < {BATCHING_CORR_MIN}")
        return (not failures), failures

    return metrics["passed"], metrics["failures"]


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------
def _gpu_description() -> dict:
    info = {
        "torch": None,
        "cuda": False,
        "gpu": None,
        "driver": None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["total_vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # pragma: no cover
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0:
            info["driver"] = out.stdout.strip().splitlines()[0]
    except Exception:  # pragma: no cover
        pass
    return info


def _run_side(
    family: str,
    side: str,
    axis: Axis,
    use_gpu: bool,
    workdir: str,
    cases: list[dict] | None = None,
) -> dict:
    out_dir = os.path.join(workdir, f"{family}_{side}")
    manifest_path = os.path.join(workdir, f"{family}_{side}.json")
    spec = FAMILY_SPECS[family]
    settings = side_settings(axis, side)
    if axis.kind == "batching" and family in BATCHING_AXIS_STOCHASTIC:
        # AR slabs are large; give the gate enough budget for N requests so the
        # comparison really is N-in-one-batch (not a pool-refused fallback).
        settings["ar_cache_budget_bytes"] = 4 * 1024**3
    job = {
        "family": family,
        "side": side,
        "settings": settings,
        "use_gpu": use_gpu,
        "cases": cases if cases is not None else _cases_for_axis(axis, family),
        "batching_max_steps": spec.get("batching_max_steps", 20),
        "out_dir": out_dir,
        "manifest": manifest_path,
    }
    job_path = os.path.join(workdir, f"{family}_{side}_job.json")
    with open(job_path, "w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", _REPO_ROOT)
    # Phase 11-2: every worker gets an explicit S3Gen autocast state, so an
    # operator's shell value can never leak into another axis' comparison.
    env["HSE_S3GEN_AUTOCAST"] = str(settings.get("s3gen_autocast") or "off")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", job_path],
        cwd=_REPO_ROOT,
        env=env,
    )
    elapsed = round(time.time() - started, 1)
    if proc.returncode != 0 or not os.path.exists(manifest_path):
        return {"error": f"worker exited {proc.returncode}", "elapsed_s": elapsed}
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["elapsed_s"] = elapsed
    return manifest


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, timeout=15
        ).stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def _markdown_table(report: dict) -> str:
    lines = [
        f"## {report['timestamp_utc']} — `{report['axis']}`",
        "",
        f"* commit `{report.get('commit') or 'dirty'}` · torch {report['env'].get('torch')} · "
        f"{report['env'].get('gpu') or 'CPU'} · driver {report['env'].get('driver') or 'n/a'} · "
        f"mode {'GPU' if report['use_gpu'] else 'CPU'}",
        "",
        "| Family | Case | Kind | Verdict | duration Δ% | RMS Δ% | log-mel MSE | LTAS RMSE dB | corr | identical | WER | Expectation |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    skipped: list[str] = []
    for family, detail in report["families"].items():
        if detail.get("kind") == "skipped" and not detail.get("cases"):
            # The gate did not run (dtype-exempt family, or every case was
            # excluded by the axis' premise check). Show it instead of leaving
            # an unexplained gap in the table.
            reason = detail.get("skip_reason") or "skipped"
            lines.append(
                f"| {family} | - | skipped | SKIP | - | - | - | - | - | - | - | {reason[:80]} |"
            )
        for case_name, case in detail.get("cases", {}).items():
            if "metrics" not in case:
                lines.append(
                    f"| {family} | {case_name} | - | ERROR | - | - | - | - | - | - | - | "
                    f"{case.get('error', '')[:60]} |"
                )
                continue
            m = case["metrics"]
            verdict = "PASS" if case["passed"] else "FAIL"
            wer = case.get("wer")
            if wer:
                wer_cell = f"{wer['wer']:.3f}"
                if not wer.get("gated", True):
                    wer_cell += " (report-only)"
                elif wer.get("failed"):
                    wer_cell += " (FAIL)"
                elif wer.get("warn"):
                    wer_cell += " (warn)"
            else:
                wer_cell = "-"
            label = case.get("label") or "-"
            lines.append(
                f"| {family} | {case_name} | {detail['kind']} | {verdict} | {m['duration_pct']} | "
                f"{m['rms_pct']} | {m['logmel_mse']} | {m['ltas_rmse_db']} | {m['corr']} | {m['identical']} | "
                f"{wer_cell} | {label} |"
            )
        for case in detail.get("skipped_cases", []):
            skipped.append(f"- `{family}/{case['name']}` — {case['reason']}")
    if skipped:
        lines.extend(["", "Skipped optional cases (excluded from metrics):", *skipped])
    lines.append("")
    return "\n".join(lines)


def run_harness(args) -> int:
    axis = parse_axis(args.axis)
    if axis.autocast:
        axis.precision = getattr(args, "s3gen_autocast", "bfloat16") or "bfloat16"
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILY_SPECS]
    if unknown:
        raise SystemExit(f"unknown families: {', '.join(unknown)}")

    use_gpu = bool(args.gpu)
    if use_gpu:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--gpu requested but CUDA is not available")

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": axis.label,
        "use_gpu": use_gpu,
        "commit": _git_commit(),
        "env": _gpu_description(),
        "families": {},
        "passed": True,
    }

    for family in families:
        if axis.kind == "dtype" and family in DTYPE_AXIS_EXEMPT and not axis.identity:
            report["families"][family] = {
                "kind": "skipped",
                "skip_reason": DTYPE_AXIS_EXEMPT[family],
                "cases": {},
            }
            print(f"[harness] {family}: SKIP dtype axis — {DTYPE_AXIS_EXEMPT[family]}", flush=True)
            continue

        kind = _family_kind(family)
        cases, skipped_cases = partition_optional_cases(_cases_for_axis(axis, family))
        skipped_entries = [
            {
                "name": case["name"],
                "reason": (
                    case.get("skip_message")
                    or case.get("skip_reason")
                    or "optional fixture missing"
                ),
                "label": case.get("label"),
            }
            for case in skipped_cases
        ]
        for case, entry in zip(skipped_cases, skipped_entries, strict=True):
            print(f"[harness] {family}/{case['name']}: {entry['reason']}", flush=True)
        if not cases and skipped_cases:
            # Every case for this family was excluded. That is not a pass: the
            # gate did not run, so say so instead of reporting an empty green.
            report["families"][family] = {
                "kind": "skipped",
                "skip_reason": "no runnable case for this axis (see skipped_cases)",
                "cases": {},
                "skipped_cases": skipped_entries,
            }
            print(
                f"[harness] {family}: SKIP — no runnable case for {axis.label}",
                flush=True,
            )
            continue
        print(f"[harness] {family}: side A ({axis.side_a})…", flush=True)
        side_a = _run_side(family, axis.side_a, axis, use_gpu, workdir, cases=cases)
        print(f"[harness] {family}: side B ({axis.side_b})…", flush=True)
        side_b = _run_side(family, axis.side_b, axis, use_gpu, workdir, cases=cases)

        if axis.kind == "batching":
            # The family's stochastic thresholds are not what a padded-batch
            # axis asserts (it is sample-wise). AR decode is the exception: its
            # vocode noise differs by construction, so its gate stays
            # distributional.
            kind = "stochastic" if family in BATCHING_AXIS_STOCHASTIC else "deterministic"
        if axis.kind in {"longform", "determinism"} and family in MODEL_SAMPLES_EVERY_RUN:
            # A model that draws fresh noise per inference cannot be compared
            # sample-wise: report/score it with the stochastic distribution
            # metrics (see :data:`MODEL_SAMPLES_EVERY_RUN`).
            kind = "stochastic"

        family_entry = {
            "kind": kind,
            "cases": {},
            "side_a": side_a,
            "side_b": side_b,
            "skipped_cases": skipped_entries,
        }
        if "error" in side_a or "error" in side_b:
            family_entry["error"] = side_a.get("error") or side_b.get("error")
            report["passed"] = False
            report["families"][family] = family_entry
            continue

        for case in cases:
            name = case["name"]
            a = side_a["cases"].get(name, {})
            b = side_b["cases"].get(name, {})
            if a.get("error") or b.get("error"):
                family_entry["cases"][name] = {
                    "error": a.get("error") or b.get("error"),
                    "passed": False,
                }
                report["passed"] = False
                continue
            metrics = compute_metrics(a["wav"], b["wav"], kind)
            passed, failures = verdict_for_axis(axis, metrics, family=family)

            # Content fidelity for VC: both rendered sides must transcribe to
            # the known source sentence. Reported always; soft gate (warn) at
            # 0.4, hard failure only above 0.7.
            wer = None
            if case.get("kind") == "vc":
                wer = score_vc_intelligibility(case, a["wav"], b["wav"])
                if wer is not None and not wer.get("gated", True):
                    print(
                        f"[harness] {family}/{name}: WER reported only "
                        f"({wer['wer']:.3f}) — non-speech target, not gated",
                        flush=True,
                    )
                if wer is not None and wer.get("gated", True):
                    if wer["failed"]:
                        passed = False
                        failures.append(
                            f"WER {wer['wer']:.3f} > {VC_WER_FAIL} (converted speech unintelligible)"
                        )
                    elif wer["warn"]:
                        print(
                            f"[harness] {family}/{name}: WER warning "
                            f"{wer['wer']:.3f} > {VC_WER_WARN}",
                            flush=True,
                        )

            case_entry = {
                "metrics": metrics,
                "passed": passed,
                "failures": failures,
                "label": case.get("label"),
            }
            if wer is not None:
                case_entry["wer"] = wer
            family_entry["cases"][name] = case_entry
            if not passed:
                report["passed"] = False
                _write_listening_pack(workdir, family, name, a["wav"], b["wav"])
            print(
                f"[harness] {family}/{name}: {'PASS' if passed else 'FAIL'}"
                + (f" — {'; '.join(failures)}" if failures else ""),
                flush=True,
            )

        if axis.kind == "dtype" or axis.identity:
            family_entry["vram"] = {
                side: {
                    "allocated_bytes": (manifest.get("vram") or {}).get("allocated_bytes"),
                    "reserved_bytes": (manifest.get("vram") or {}).get("reserved_bytes"),
                }
                for side, manifest in ((axis.side_a, side_a), (axis.side_b, side_b))
            }
        report["families"][family] = family_entry

    report["passed"] = bool(report["passed"]) and all(
        detail.get("kind") == "skipped" or all(c.get("passed") for c in detail["cases"].values())
        for detail in report["families"].values()
    )

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    table = _markdown_table(report)
    if args.md_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
        with open(args.md_out, "a", encoding="utf-8") as handle:
            handle.write(table)

    print(table)
    print(f"[harness] overall: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


def _family_kind(family: str) -> str:
    from tests.audio_fixtures import FAMILY_KIND

    return FAMILY_KIND.get(family, "deterministic")


def _chunks_for_axis_text(text: str, language: str | None, chunk_size: int | None) -> list[str]:
    """Split ``text`` exactly the way the serving layer does (no drift).

    Imported lazily so the harness module stays importable on a bare CPU
    interpreter, and kept in one place so the case-selection guard and its tests
    cannot disagree about the splitter.
    """
    from harmonyspeech.common.audio_utils import split_text_into_chunks

    return split_text_into_chunks(text, chunk_size=chunk_size, language=language)


def _longform_identity_cases(family: str) -> list[dict]:
    """Cases for ``longform:off-vs-on``, made gate-honest (Phase 12-4).

    The axis asserts that a short text with ``split_text: true`` takes the
    untouched single-shot path, so the two sides must be the same computation.
    Two conditions break that premise, and both are handled here instead of
    producing a misleading FAIL:

    * **A text that splits into more than one chunk.** At the serving layer's
      default budget (:data:`LONGFORM_IDENTITY_CHUNK_BUDGET`) the ON side then
      synthesises each chunk separately, and the OFF side one utterance: the
      outputs legitimately differ at every synthesis boundary. The case is
      excluded with a recorded ``skip_reason`` (the report lists it) rather than
      gated.
    * **A reference clip Silero VAD cannot detect** (the synthetic fixtures —
      see ``tests.audio_fixtures.VAD_UNDETECTABLE_CLIPS``). The OpenVoice
      ``vad → embed`` pipelines would get zero segments and error, so on
      :data:`VAD_CARRIED_FAMILIES` the case is pointed at the real-speech clip
      of the same role. The axis gates the dispatcher, not the reference bytes.
    """
    from tests.audio_fixtures import TEXTS, VAD_UNDETECTABLE_CLIPS

    spec = FAMILY_SPECS[family]
    tts_cases = [case for case in spec["cases"] if case.get("kind", "tts") == "tts"]
    if not tts_cases:
        raise SystemExit(
            f"family '{family}' has no short TTS case; the long-form identity axis "
            f"covers {', '.join(LONGFORM_AXIS_FAMILIES)}"
        )

    selected: list[dict] = []
    for original in tts_cases:
        case = dict(original)
        reference = case.get("reference")
        if (
            family in VAD_CARRIED_FAMILIES
            and reference in VAD_UNDETECTABLE_CLIPS
            and reference in REAL_SPEECH_REFERENCE
        ):
            replacement = REAL_SPEECH_REFERENCE[reference]
            case["reference"] = replacement
            note = (
                f"reference {reference} -> {replacement}: the synthetic clip has no "
                "VAD-detectable speech"
            )
            case["label"] = f"{case['label']} [{note}]" if case.get("label") else note

        text = _case_text(case, TEXTS)
        chunks = _chunks_for_axis_text(
            text, case.get("language"), LONGFORM_IDENTITY_CHUNK_BUDGET
        )
        if len(chunks) > 1:
            case["skip_reason"] = (
                f"identity axis: text splits into {len(chunks)} chunks at the "
                f"{LONGFORM_IDENTITY_CHUNK_BUDGET}-character serving budget, so "
                "split_text on/off differ by construction (different synthesis "
                "boundaries); the byte-identity gate is not meaningful"
            )
        selected.append(case)
    return selected


def _cases_for_axis(axis: Axis, family: str) -> list[dict]:
    """Cases to run for this axis: the batching gate uses its own case set.

    The batching axis is allow-listed by :data:`BATCHING_AXIS_ENABLED`: a
    family only reports a meaningful gate once its handler actually forwards a
    padded batch. Families in :data:`BATCHING_AXIS_PENDING` already have their
    case set registered, so the task that lands the handler only adds the name
    to the allow-list.
    """
    spec = FAMILY_SPECS[family]
    if axis.kind == "longform":
        # Phase 12-4 (a): short-text identity at family breadth. A case only
        # gates anything when its text yields 0/1 chunks at the serving layer's
        # budget; see :func:`_longform_identity_cases`.
        return _longform_identity_cases(family)
    if axis.kind == "determinism":
        # Phase 12-4 (b): the same chunked request on both sides. The fixture
        # text is long enough to split into several chunks at the axis budget.
        tts_cases = [case for case in spec["cases"] if case.get("kind", "tts") == "tts"]
        if not tts_cases:
            raise SystemExit(
                f"family '{family}' has no TTS case; the seeded-determinism axis "
                f"covers {', '.join(LONGFORM_AXIS_FAMILIES)}"
            )
        base = dict(tts_cases[0])
        base["name"] = f"chunked_{base['name']}"
        base["text_literal"] = LONGFORM_CHUNKED_TEXT
        base.pop("text", None)
        base["label"] = "chunked determinism (same seed, two runs)"
        if len(
            _chunks_for_axis_text(
                LONGFORM_CHUNKED_TEXT, base.get("language"), LONGFORM_AXIS_CHUNK_SIZE
            )
        ) < 2:
            raise SystemExit(
                "LONGFORM_CHUNKED_TEXT does not split at LONGFORM_AXIS_CHUNK_SIZE="
                f"{LONGFORM_AXIS_CHUNK_SIZE}; the determinism axis would only exercise "
                "the single-shot path"
            )
        return [base]
    if axis.kind != "batching":
        return spec["cases"]
    cases = spec.get("batching_cases")
    if not cases:
        raise SystemExit(
            f"family '{family}' has no batching_cases; the batching axis only covers "
            f"the families registered for it ({', '.join(sorted(BATCHING_AXIS_ENABLED))})"
        )
    if family not in BATCHING_AXIS_ENABLED:
        pending = BATCHING_AXIS_PENDING.get(family, "a later task")
        raise SystemExit(
            f"family '{family}' has batching_cases registered but its handler does not batch "
            f"yet; add it to BATCHING_AXIS_ENABLED when {pending} lands"
        )
    return cases


def _write_listening_pack(workdir: str, family: str, case: str, path_a: str, path_b: str) -> None:
    """Copy A/B clips into out/regression-pack/ for the human listening pass."""
    pack_dir = os.path.join(_REPO_ROOT, "out", "regression-pack", family)
    os.makedirs(pack_dir, exist_ok=True)
    import shutil

    shutil.copyfile(path_a, os.path.join(pack_dir, f"{case}__A.wav"))
    shutil.copyfile(path_b, os.path.join(pack_dir, f"{case}__B.wav"))
    print(f"[harness] listening pack: {pack_dir}/{case}__A.wav vs {case}__B.wav", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audio regression gate (Phase 4-2)")
    parser.add_argument("--families", default="chatterbox", help="comma-separated family ids")
    parser.add_argument(
        "--axis",
        default="dtype:float32-vs-float16",
        help=(
            "change axis: dtype:float32-vs-float16, sharing:dedup-vs-dup, "
            "batching:1-vs-4, longform:off-vs-on, determinism:run1-vs-run2, "
            "dtype:fp32-vs-autocast or dtype:float32-vs-float32 (identity)"
        ),
    )
    parser.add_argument("--gpu", action="store_true", help="require a CUDA host")
    parser.add_argument("--json-out", default="out/audio-regression/latest.json")
    parser.add_argument("--md-out", default=None, help="append the markdown result table here")
    parser.add_argument(
        "--s3gen-autocast",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="precision for side B of the dtype:fp32-vs-autocast axis (Phase 11-2)",
    )
    parser.add_argument("--workdir", default="out/audio-regression")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return run_worker(args.worker)
    return run_harness(args)


if __name__ == "__main__":
    raise SystemExit(main())
