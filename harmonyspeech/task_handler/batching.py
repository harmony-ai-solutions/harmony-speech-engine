"""Padded-batch helpers for the engine's one-shot model steps (Phase 5-2, 8-1).

Several steps are fixed-graph one-shot models: text or features → tokens → mel →
waveform in a single forward pass (OpenVoice V1 / MeloTTS ``SynthesizerTrn``,
tone converters, restorers, vocoders, speaker embedders). That makes
``[B, T_max]`` padding with a per-sample length vector the standard, safe
batching technique (unlike Chatterbox, whose autoregressive token loop needs a
per-sequence KV cache).

Three padding axes
------------------
Every padded batch is a set of variable-length rows plus a length vector. What
differs between steps is only the *unit* of that length — the axis. "Token" is
the historical name (Phase 5-2 measured phones); the machinery is unit-agnostic:

=============  ==============  ===============================================
axis unit      one row         steps that pad on it
=============  ==============  ===============================================
``token``      phones / ids    synthesizers (OpenVoice V1, MeloTTS, Harmony
                               Speech): phone / tone / language-id sequences;
                               the Phase 9-5 S3Gen hop: speech-token sequences
``frame``      mel frames      tone converters, restorers, vocoders, Harmony
                               Speech encoder: mel or feature ``[bins, T]``
``sample``     audio samples   speaker embedders and any raw-waveform step
=============  ==============  ===============================================

The axis is a naming convention, not a type: :func:`plan_batches`,
:func:`default_token_budget` and ``ModelConfig.batch_token_budget`` only ever
see a length vector, so they apply unchanged on all three axes. Nothing
phone-specific leaks into them — a sample-axis step passes sample counts and
gets sample-count budgeting.

This module holds the pure, testable pieces so the handlers stay readable:

* :func:`pad_phone_batch` / :func:`pad_feature_batch` /
  :func:`pad_waveform_batch` build the padded tensors plus the length vector for
  the token, frame and sample axes respectively.
* :func:`plan_batches` does the length-similar sub-batching (greedy, so a
  pathological mix of one long and many short rows does not spend the step
  multiplying padding).
* :func:`trim_decoder_output` / :func:`trim_feature_output` unpad using the
  *per-sample* lengths the caller passes in (never a padded-shape assumption).
* :func:`batched_inference_enabled` resolves the per-model config flag.

Parallel lanes for steps that cannot batch
-------------------------------------------
Some steps have no batch axis at all: KittenTTS exports a static batch
dimension of 1, and Silero's ``get_speech_timestamps`` is a single-stream
windowing loop. Those steps run one request per *lane* instead — a bounded
thread pool sized ``min(requests, max_batch_size)`` — so the engine still
processes the whole step concurrently. :func:`lane_count` and
:func:`run_in_lanes` are that mechanism; per-request error isolation and input
order are part of their contract.

Nothing here imports a model: the helpers are CPU/GPU and dtype agnostic.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor

import torch

#: Maximum ``T_max / T_min`` inside one padded sub-batch. A step that mixes a
#: very short and a very long text pays for the difference in compute on the
#: padding, so such a mix is split into length-similar groups instead.
LENGTH_SIMILARITY_RATIO = 4.0

#: Model types whose handler implements a real padded-batch forward.
BATCHED_INFERENCE_FAMILIES = frozenset(
    {
        # Phase 5-2: one-shot padded synthesis (token axis).
        "OpenVoiceV1Synthesizer",
        "MeloTTSSynthesizer",
        # Phase 8-2: speaker embedders (mel-frame / partial-frame axis).
        "OpenVoiceV1ToneConverterEncoder",
        "OpenVoiceV2ToneConverterEncoder",
        "HarmonySpeechEncoder",
        # Phase 8-3: one-shot tone conversion (mel-frame axis).
        "OpenVoiceV1ToneConverter",
        "OpenVoiceV2ToneConverter",
        # Phase 8-4: one-shot vocoders / restorer (mel-frame axis).
        "HarmonySpeechVocoder",
        "VoiceFixerRestorer",
        "VoiceFixerVocoder",
    }
)


def supports_batched_inference(model_config) -> bool:
    """Return True when this executor's model type has a batched forward."""
    return getattr(model_config, "model_type", None) in BATCHED_INFERENCE_FAMILIES


#: Model types whose decode is a token-level AR loop on the shared KV pool
#: (Phase 9-3). They are not padded-batch families: one engine step decodes one
#: speech token per active row through :class:`ARBatchRunner`. This set must stay
#: equal to ``processing.scheduler.AR_INFERENCE_MODEL_TYPES`` (the runtime
#: admission flag); both name the same two Phase 9 members.
AR_INFERENCE_FAMILIES = frozenset({"ChatterboxTTS", "ChatterboxTurboTTS"})


def supports_ar_batching(model_config) -> bool:
    """True when this model type has the Phase 9-3 token-level AR runner."""
    return getattr(model_config, "model_type", None) in AR_INFERENCE_FAMILIES


def ar_batching_enabled(model_config) -> bool:
    """Resolve the AR-runner flag for a model.

    ``batched_inference`` is shared with the padded-batch families: ``None``
    (unset) uses the family default (on for the AR families), ``False`` forces
    the sequential loop even when the pool is available. The caller still has to
    check the pool/arena and the ``len == 1`` fallback.
    """
    if not supports_ar_batching(model_config):
        return False
    flag = getattr(model_config, "batched_inference", None)
    return True if flag is None else bool(flag)



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


def lane_count(request_count: int, max_batch_size: int) -> int:
    """Lanes for a parallel-lane step: ``min(requests, max_batch_size)``, at least 1.

    The scheduler already admitted at most ``max_batch_size`` requests for this
    step, so the step's own size is usually the binding term; the clamp keeps
    the pool from ever exceeding the configured admission width (and keeps a
    hand-built larger list from oversubscribing the CPU).
    """
    return max(1, min(int(request_count), max(1, int(max_batch_size))))


def run_in_lanes(function, items, lanes: int, thread_name_prefix: str = "engine-lane"):
    """Run ``function(item)`` for every item, up to ``lanes`` at a time.

    Returns one ``(value, error)`` pair per item **in input order**: a lane that
    raises produces ``(None, error)`` for its own item only, so the caller can
    turn exactly that request into an error result without disturbing its
    neighbours (Phase 5-1 pairing invariants). ``lanes <= 1`` runs inline on the
    calling thread — the single-request fast path, and the fallback for a model
    whose lanes are not worth it (Silero VAD, see ``docs/batching.md``).
    """
    results: list[tuple[object, Exception | None]] = [(None, None)] * len(items)
    if lanes <= 1:
        for index, item in enumerate(items):
            try:
                results[index] = (function(item), None)
            except Exception as error:  # noqa: BLE001 - becomes this request's error result
                results[index] = (None, error)
        return results

    with ThreadPoolExecutor(max_workers=lanes, thread_name_prefix=thread_name_prefix) as pool:
        futures = [pool.submit(function, item) for item in items]
        for index, future in enumerate(futures):
            try:
                results[index] = (future.result(), None)
            except Exception as error:  # noqa: BLE001 - becomes this request's error result
                results[index] = (None, error)
    return results


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


def pad_feature_batch(
    features,
    pad_value: float = 0.0,
    device=None,
    return_lengths: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pad ``[D, T]`` feature matrices (e.g. BERT or mel features) to ``[B, D, T_max]``.

    Returns just ``padded`` by default — the shape the Phase 5-2 MeloTTS call
    sites use, so their source stays unchanged. With ``return_lengths=True`` it
    returns ``(padded, frame_lengths)``, where ``frame_lengths`` is a ``long``
    tensor of the per-row ``T`` values: the frame axis needs those both for
    :func:`plan_batches` and to unpad the model's output with
    :func:`trim_feature_output`.
    """
    if not features:
        raise ValueError("pad_feature_batch: empty feature list")
    channels = features[0].shape[0]
    frame_lengths = [int(feature.shape[-1]) for feature in features]
    max_len = max(frame_lengths)
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
        padded[row, :, : frame_lengths[row]] = feature.to(device="cpu")
    if device is not None:
        padded = padded.to(device)
    if return_lengths:
        return padded, torch.tensor(frame_lengths, dtype=torch.long, device=device)
    return padded


def pad_waveform_batch(
    waveforms,
    sample_rates=None,
    pad_value: float = 0.0,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad 1-D waveforms to ``[B, 1, T_max]``; returns ``(padded, sample_lengths)``.

    Two accepted calling forms, both requiring one common sample rate:

    * ``pad_waveform_batch([(waveform, sample_rate), ...])`` — rows as
      ``(tensor, rate)`` pairs, the shape an embedder handler reads off its
      ``AudioArtifact``s;
    * ``pad_waveform_batch([waveform, ...], sample_rates=[rate, ...])``.

    All rates must match. The Phase 2 boundary decode plus the ``resampled()``
    views put every waveform of a step on one rate, and mixing rates inside one
    padded batch would silently reinterpret sample indices, so a mismatch raises
    a ``ValueError`` naming the rates instead. ``pad_value`` defaults to ``0.0``
    (digital silence); positions past ``sample_lengths[i]`` are masked out by
    the model's own length argument, not by the padding value.
    """
    if not waveforms:
        raise ValueError("pad_waveform_batch: empty waveform list")
    if sample_rates is None:
        pairs = list(waveforms)
        for row, item in enumerate(pairs):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    f"pad_waveform_batch: row {row} is not a (waveform, sample_rate) pair"
                )
    else:
        if len(sample_rates) != len(waveforms):
            raise ValueError(
                f"pad_waveform_batch: {len(sample_rates)} sample rates for {len(waveforms)} waveforms"
            )
        pairs = list(zip(waveforms, sample_rates, strict=True))

    tensors: list[torch.Tensor] = []
    rates: list[int] = []
    for row, (waveform, rate) in enumerate(pairs):
        if waveform.dim() != 1:
            raise ValueError(
                f"pad_waveform_batch: row {row} is {tuple(waveform.shape)}, expected a 1-D waveform"
            )
        tensors.append(waveform)
        rates.append(int(rate))
    if len(set(rates)) > 1:
        raise ValueError(
            f"pad_waveform_batch: mixed sample rates {sorted(set(rates))}; resample before batching"
        )

    lengths = [int(waveform.numel()) for waveform in tensors]
    dtype = tensors[0].dtype
    if not dtype.is_floating_point:
        # A padded batch is only meaningful in a float dtype; integer PCM rows
        # (rare) are widened rather than silently padded with a numeric value.
        dtype = torch.float32
    padded = torch.full((len(tensors), 1, max(lengths)), float(pad_value), dtype=dtype)
    for row, waveform in enumerate(tensors):
        padded[row, 0, : lengths[row]] = waveform.to(device="cpu", dtype=dtype)
    if device is not None:
        padded = padded.to(device)
    return padded, torch.tensor(lengths, dtype=torch.long, device=device)


def plan_batches(
    lengths,
    max_batch_size: int,
    token_budget: int | None = None,
    max_ratio: float = LENGTH_SIMILARITY_RATIO,
) -> list[list[int]]:
    """Group item indices into padded sub-batches.

    ``lengths`` is a length vector in the step's own axis unit — phones,
    mel/feature frames or audio samples (see the module docstring). The values
    are only ever compared and multiplied, never interpreted, so the same
    grouping applies on all three axes.

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
        return not (token_budget is not None and (len(current) + 1) * group_max > token_budget)

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
    """Per-step padded-area budget: ``max_batch_size × median length``.

    Derived from the step's own lengths rather than a fixed constant, so a step
    of long rows is allowed the same *padding* headroom (B × median) as a step
    of short ones. The unit is whatever axis the step pads on (phones, mel
    frames or audio samples); the arithmetic is unit-agnostic. ``None`` for an
    empty step.
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


def trim_feature_output(
    outputs: torch.Tensor,
    frame_lengths,
    dim: int = -1,
) -> list[torch.Tensor]:
    """Split a padded feature batch into per-item slices using explicit lengths.

    ``outputs`` is ``[B, ..., T_max]`` with the batch axis ``0`` and the padded
    time axis at ``dim`` (default ``-1``). Both ranks the handlers produce work
    as-is: ``[B, T_max]`` (2-D, e.g. an embedding over frames) and
    ``[B, bins, T_max]`` (3-D mel/feature maps); the rule generalises to any
    rank ≥ 2. Each returned tensor has the batch axis dropped and every other
    axis intact, so a 3-D input yields ``[bins, T_i]`` and a 2-D input ``[T_i]``.

    ``frame_lengths`` is the *explicit* per-item length vector returned by
    :func:`pad_feature_batch` (or the model's own frame counts). Lengths are
    never inferred from the padded shape — this is the mel-frame sibling of
    :func:`trim_decoder_output` and validates the same invariants: the length
    count must equal the batch size, and no row may ask for more frames than
    the padded axis holds.
    """
    if outputs.dim() < 2:
        raise ValueError(
            f"trim_feature_output: expected [B, ..., T], got {tuple(outputs.shape)}"
        )
    time_dim = dim if dim >= 0 else outputs.dim() + dim
    if time_dim <= 0 or time_dim >= outputs.dim():
        raise ValueError(
            f"trim_feature_output: time axis {dim} is not a non-batch axis of "
            f"{tuple(outputs.shape)}"
        )
    lengths = [int(value) for value in frame_lengths]
    if len(lengths) != outputs.shape[0]:
        raise ValueError(
            f"trim_feature_output: {len(lengths)} lengths for a batch of {outputs.shape[0]}"
        )
    max_frames = outputs.shape[time_dim]
    trimmed = []
    for row, frames in enumerate(lengths):
        frames = max(0, frames)
        if frames > max_frames:
            raise ValueError(
                f"trim_feature_output: sample {row} needs {frames} frames, batch holds {max_frames}"
            )
        index: list[slice | int] = [slice(None)] * outputs.dim()
        index[0] = row
        index[time_dim] = slice(0, frames)
        trimmed.append(outputs[tuple(index)])
    return trimmed

