"""Docs-consistency lint for the batching matrix (Phase 8-5).

`docs/batching.md` is the phase's operator contract: one table, one row per
registered model type, each stating the real mechanism, the budget axis, the
fallback and the reason for its `max_batch_size`. This module keeps the table
honest the same way the Phase 2 grep-invariants keep the artifact contract
honest — a new model type cannot be registered without a matrix row, and a type
cannot be listed twice with two different stories.
"""

import re
from pathlib import Path

import pytest
import yaml

from harmonyspeech.modeling.loader import _MODEL_CONFIGS

DOCS = Path(__file__).resolve().parents[2] / "docs" / "batching.md"
REPO_ROOT = DOCS.parents[1]


def _matrix_rows() -> list[str]:
    """Rows of the first table under the "The batching matrix" heading."""
    text = DOCS.read_text(encoding="utf-8")
    heading = "## The batching matrix"
    section = text.split(heading, 1)[1]
    section = section.split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    assert rows, "docs/batching.md: the batching matrix has no model rows"
    return rows


def _mentions(row: str, model_type: str) -> bool:
    """Exact-token match, so ``...ToneConverter`` does not match its Encoder."""
    return re.search(rf"`{re.escape(model_type)}`", row) is not None


@pytest.mark.unit
def test_every_registered_model_type_has_exactly_one_matrix_row():
    rows = _matrix_rows()
    missing = []
    duplicated = []
    for model_type in _MODEL_CONFIGS:
        matches = [row for row in rows if _mentions(row, model_type)]
        if not matches:
            missing.append(model_type)
        elif len(matches) > 1:
            duplicated.append(model_type)
    assert not missing, f"model type(s) missing from docs/batching.md: {missing}"
    assert not duplicated, f"model type(s) listed twice in docs/batching.md: {duplicated}"


@pytest.mark.unit
def test_matrix_rows_state_a_mechanism_and_a_reason():
    """Every row names one of the four mechanisms and a non-empty rationale."""
    mechanisms = ("padded batch", "CT2 batch", "parallel lanes", "loop")
    for row in _matrix_rows():
        assert any(mechanism in row for mechanism in mechanisms), f"no mechanism: {row}"
        # Split off the leading type / entries / size / mechanism cells; the
        # last cell is the rationale and must not be empty.
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 7, f"expected 7 matrix cells: {row}"
        assert cells[-1], f"empty rationale: {row}"


def _matrix_sizes() -> dict[str, int]:
    """The `max_batch_size` cell of every matrix row, keyed by model type."""
    sizes: dict[str, int] = {}
    for row in _matrix_rows():
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        sizes[cells[0].strip("`")] = int(cells[2].strip("`"))
    return sizes


@pytest.mark.unit
@pytest.mark.parametrize("config_name", ["config.gpu.yml", "config.yml"])
def test_shipped_configs_admit_exactly_the_matrix_value(config_name):
    """The operator table and the shipped configs must tell the same story.

    Phase 8-6's contract is that `docs/batching.md` describes the configs an
    operator actually runs: a row saying `8` next to an entry declaring `1`
    would be the old decorative-`max_batch_size` bug in documentation form.
    """
    sizes = _matrix_sizes()
    with open(REPO_ROOT / config_name, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for entry in config["model_configs"]:
        model_type = entry["model_type"]
        assert entry["max_batch_size"] == sizes[model_type], (
            f"{config_name}: {entry['name']} ({model_type}) declares "
            f"{entry['max_batch_size']}, docs/batching.md says {sizes[model_type]}"
        )


@pytest.mark.unit
def test_matrix_documents_the_phase_8_5_skip_evidence():
    """The three evidence-based verdicts are stated where the table points."""
    text = DOCS.read_text(encoding="utf-8")
    # HarmonySpeech synthesizer: measured padding drift.
    assert "0.155 peak" in text
    # KittenTTS: static batch dimension 1.
    assert "static batch dimension of 1" in text
    # KittenTTS: the baked-in ORT constant and the measured matrix.
    assert "KITTEN_TTS_INTRA_OP_NUM_THREADS" in text
    assert "1.71×" in text and "1.91×" in text
    # Silero: sequential verdict with lane numbers.
    assert "0.95×" in text
    # Chatterbox: the product decision, quoted verbatim (allow line wrapping).
    normalized = " ".join(text.replace(">", " ").split())
    assert "explicitly out of scope** per the product decision (2026-11-05)" in normalized


@pytest.mark.unit
def test_matrix_documents_the_five_execution_classes_and_the_whisper_final_row():
    """Phase 10-5: the class table and Whisper's final row are the operator contract."""
    text = DOCS.read_text(encoding="utf-8")
    assert "## Execution-model classes" in text
    for class_name in ("padded_batch", "loop", "queue_depth", "lanes_windows", "ar_decode"):
        assert f"`{class_name}`" in text, f"docs/batching.md is missing the {class_name} class"
    # Whisper's final mechanism wording (lanes M × windows B, intra-audio 16).
    whisper_row = next(row for row in _matrix_rows() if _mentions(row, "FasterWhisper"))
    assert "lanes M × windows B" in whisper_row
    assert "intra-audio CT2 batch is 16" in whisper_row
    assert "opt-in" in whisper_row
    # The window batcher is documented as opt-in until the B=1 parity gate.
    normalized = " ".join(text.split())
    assert "Phase 10-5 kept it opt-in (dated decision 2026-11-07)" in normalized
