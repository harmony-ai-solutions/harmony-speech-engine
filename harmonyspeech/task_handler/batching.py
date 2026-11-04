"""Padded-batch helpers for one-shot synthesis models (Phase 5-2).

OpenVoice V1 ``SynthesizerTrn`` and MeloTTS's ``SynthesizerTrn`` are fixed-graph
one-shot models: text → phones → mel → waveform in a single forward pass. That
makes ``[B, T_max]`` padding with a per-sample ``x_lengths`` vector the standard,
safe batching technique (unlike Chatterbox, whose autoregressive token loop needs
a per-sequence KV cache).

This module holds the pure, testable pieces so the handlers stay readable:

* :func:`pad_phone_batch` / :func:`pad_feature_batch` build the padded tensors
  plus the length vector.
* :func:`plan_batches` does the length-similar sub-batching (greedy, so a
  pathological mix of one long and many short texts does not spend the step
  multiplying padding).
* :func:`trim_decoder_output` unpads a decoder output using the *per-sample*
  frame lengths the model itself produced (never an index assumption).
* :func:`batched_inference_enabled` resolves the per-model config flag.

Nothing here imports a model: the helpers are CPU/GPU and dtype agnostic.
"""

from __future__ import annotations

import statistics

import torch

#: Maximum ``T_max / T_min`` inside one padded sub-batch. A step that mixes a
#: very short and a very long text pays for the difference in compute on the
#: padding, so such a mix is split into length-similar groups instead.
LENGTH_SIMILARITY_RATIO = 4.0

#: Model types whose handler implements a real padded-batch forward.
BATCHED_INFERENCE_FAMILIES = frozenset({"OpenVoiceV1Synthesizer", "MeloTTSSynthesizer"})


def supports_batched_inference(model_config) -> bool:
    """Return True when this executor's model type has a batched forward."""
    return getattr(model_config, "model_type", None) in BATCHED_INFERENCE_FAMILIES


def batched_inference_enabled(model_config) -> bool:
    """Resolve the per-model ``batched_inference`` flag.

    ``None`` (unset) means "use the family default": on for the families with a
    validated batched forward (Phase 5-2 gate), off everywhere else. An explicit
    ``false`` keeps the legacy per-item loop even for a batched family, and a
    ``true`` on a family without a batched forward still falls back to the loop.
    """
    if not supports_batched_inference(model_config):
        return False
    flag = getattr(model_config, "batched_inference", None)
    return True if flag is None else bool(flag)


def pad_phone_batch(sequences, pad_id: int = 0, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad 1-D phone/tone/language id sequences to ``[B, T_max]``.

    Returns ``(padded_long_tensor, lengths_long_tensor)``. ``pad_id`` matches the
    model's own blank symbol (0 for both VITS-family models), and every padded
    position is masked out by the model through ``x_lengths``.
    """
    lengths = [int(seq.numel()) for seq in sequences]
    if not lengths:
        raise ValueError("pad_phone_batch: empty sequence list")
    max_len = max(lengths)
    padded = torch.full((len(sequences), max_len), int(pad_id), dtype=torch.long)
    for row, seq in enumerate(sequences):
        padded[row, : lengths[row]] = seq.to(device="cpu", dtype=torch.long)
    if device is not None:
        padded = padded.to(device)
    return padded, torch.tensor(lengths, dtype=torch.long, device=device)


def pad_feature_batch(features, pad_value: float = 0.0, device=None) -> torch.Tensor:
    """Pad ``[D, T]`` feature matrices (e.g. BERT embeddings) to ``[B, D, T_max]``."""
    if not features:
        raise ValueError("pad_feature_batch: empty feature list")
    channels = features[0].shape[0]
    max_len = max(int(feature.shape[-1]) for feature in features)
    padded = torch.full(
        (len(features), channels, max_len),
        float(pad_value),
        dtype=features[0].dtype,
    )
    for row, feature in enumerate(features):
        if int(feature.shape[0]) != channels:
            raise ValueError(
                f"pad_feature_batch: inconsistent channel count {feature.shape[0]} vs {channels}"
            )
        padded[row, :, : feature.shape[-1]] = feature.to(device="cpu")
    if device is not None:
        padded = padded.to(device)
    return padded


def plan_batches(
    lengths,
    max_batch_size: int,
    token_budget: int | None = None,
    max_ratio: float = LENGTH_SIMILARITY_RATIO,
) -> list[list[int]]:
    """Group item indices into padded sub-batches.

    Rules (in order):

    * never exceed ``max_batch_size`` items per group;
    * keep ``T_max / T_min <= max_ratio`` inside a group (length similarity);
    * keep ``len(group) * T_max <= token_budget`` when a budget is given.

    Items are visited longest-first (greedy), so the long item starts a group and
    short items either join it while the ratio holds or wait for a later,
    shorter group. The returned groups cover every index exactly once.
    """
    if not lengths:
        return []
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")

    order = sorted(range(len(lengths)), key=lambda index: int(lengths[index]), reverse=True)
    groups: list[list[int]] = []
    current: list[int] = []

    def _fits(candidate_index: int) -> bool:
        if len(current) >= max_batch_size:
            return False
        group_max = max(int(lengths[index]) for index in current + [candidate_index])
        group_min = max(1, min(int(lengths[index]) for index in current + [candidate_index]))
        if group_max / group_min > max_ratio:
            return False
        if token_budget is not None and (len(current) + 1) * group_max > token_budget:
            return False
        return True

    for index in order:
        if current and not _fits(index):
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)

    # Stable, predictable output order (request order within a group, groups by
    # their first member) so tests and logs are deterministic.
    return [sorted(group) for group in groups]


def default_token_budget(lengths, max_batch_size: int) -> int | None:
    """Per-step token budget: ``max_batch_size × median length``.

    Derived from the step's own text lengths rather than a fixed constant, so a
    step of long texts is allowed the same *padding* headroom (B × median) as a
    step of short ones. ``None`` for an empty step.
    """
    if not lengths:
        return None
    median = max(1, int(statistics.median(int(length) for length in lengths)))
    return max(1, int(max_batch_size)) * median


def collapse_per_sample(values, tolerance: float = 1e-12):
    """Return a float when every value is equal, else the list of values.

    Keeps the common case (all requests at speed 1.0, one shared noise setting)
    a *scalar*, so the batched forward multiplies by a scalar exactly like the
    single-sample path instead of by a broadcast tensor.
    """
    floats = [float(value) for value in values]
    if not floats:
        return 0.0
    first = floats[0]
    if all(abs(value - first) <= tolerance for value in floats):
        return first
    return floats


def trim_decoder_output(
    output: torch.Tensor,
    y_lengths: torch.Tensor,
    hop_length: int,
) -> list[torch.Tensor]:
    """Split a padded decoder output ``[B, 1, T]`` into per-sample waveforms.

    ``y_lengths`` are *frame* counts produced by the model (sum of the predicted
    durations), so each sample is trimmed to ``y_lengths[i] * hop_length``
    samples. This is the unpad step: it never infers a length from the padded
    tensor shape.
    """
    if output.dim() != 3:
        raise ValueError(f"trim_decoder_output: expected [B, C, T], got {tuple(output.shape)}")
    lengths = [int(value) for value in y_lengths]
    if len(lengths) != output.shape[0]:
        raise ValueError(
            f"trim_decoder_output: {len(lengths)} lengths for a batch of {output.shape[0]}"
        )
    max_samples = output.shape[-1]
    trimmed = []
    for row, frames in enumerate(lengths):
        samples = max(0, frames) * int(hop_length)
        if samples > max_samples:
            raise ValueError(
                f"trim_decoder_output: sample {row} needs {samples} samples, decoder produced {max_samples}"
            )
        trimmed.append(output[row, :, :samples])
    return trimmed
