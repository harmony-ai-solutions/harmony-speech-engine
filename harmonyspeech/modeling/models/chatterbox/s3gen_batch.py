"""Padded-batch S3Gen token→waveform hop (Phase 9-5).

Why this module exists
---------------------
The Chatterbox family's non-autoregressive half is the upstream S3Gen graph:
speech tokens (+ a reference prompt) → CFM mel decoder → HiFT vocoder. Its
docstring says it is "designed for batch_size=1 only" (``s3gen.py:173-229``),
and Phase 9-3's AR runner can only hand finished rows to it one at a time
(``model_runner_base._finalize_ar_row`` → ``_finalize_ar_rows``). This module adds the Phase 8 padded-batch
entry point for that hop **without touching the installed package** — the same
seam discipline as ``t3_ar.py``.

The padding audit (recorded in ``9-5-S3GenPaddedBatch.md``)
-----------------------------------------------------------
Measured on a tiny seeded instance of the real upstream classes:

* **UpsampleConformerEncoder** — *not* safe by construction. The input
  ``LinearNoSubsampling`` embed adds a bias and the causal ``PreLookaheadLayer``
  reads it forward, so a short row drifted **0.4075 peak**. Zeroing each row's
  padding after ``embed``, after the lookahead and before ``up_layer`` brings it
  to the batch-size numerical floor (~3.6e-6).
* **ConditionalDecoder / CausalConditionalCFM** — safe by construction: every
  stage multiplies by the mask and the convs are causal, so trailing padding can
  never reach the valid tail. The only upstream bug is the mask *length*
  (``make_pad_mask(h_lengths)`` builds ``max(valid)`` columns, not the padded
  width) — this module rebuilds it over the padded axis.
* **HiFTGenerator** — *not* safe. The f0 conv stack, the transposed upsample
  convs, the source branch and the resblocks all leak (0.44 peak on a short
  row), and both the STFT and the ISTFT reflect/overlap at the *batch* end.
  Every stage is masked per row, the source phase/noise are drawn per row, and
  the STFT/ISTFT are run per row on the row's own sample count.
* **Noise** — upstream draws four tensors from the global RNG: the CFM start
  noise, the ``SineGen`` phase, the ``SineGen`` noise and the ``SourceModule``
  noise. Each is re-drawn per row from that row's ``torch.Generator`` so a row's
  audio is identical alone or in a batch, and B=1 is bit-identical to upstream
  (proven by the equality gate).

Scope
-----
* **Standard (English) S3Gen only.** The meanflow (Turbo) graph is a different
  solver with two batch-1 assumptions (``flow_inference`` hard-codes
  ``torch.randn(1, 80, …)`` and the ``noised_mels`` placement uses a single
  prompt length); it is the task's documented hop skip (see ``docs/batching.md``).
  ``supports_batched_s3gen`` returns ``False`` for it and the handler keeps the
  per-row upstream call.
* **B=1 / unseeded fast path** calls upstream ``s3gen.inference`` unchanged, so
  the sequential loop and the single-request case stay bit-identical.
* **Mixed precision (Phase 11-2)** is an opt-in experiment
  (``HSE_S3GEN_AUTOCAST``, default off): the padded flow wraps its encoder and
  CFM decoder forwards in ``torch.autocast`` and casts the results back to the
  flow dtype, so the ODE updates, masks, unpadding and ``mel2wav`` stay float32.
  See ``s3gen_precision.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from chatterbox.models.s3gen.utils.mask import make_pad_mask
from torch import Tensor

from harmonyspeech.modeling.models.batch_padding import length_mask, sync_mask
from harmonyspeech.modeling.models.chatterbox.s3gen_precision import (
    S3GEN_PRECISION_FP32,
    install_inference_autocast,
    s3gen_autocast,
)

__all__ = [
    "S3GenBatchError",
    "S3GenRow",
    "inference_batched",
    "supports_batched_s3gen",
    "supports_batched_tokenizer",
    "tokenize_batched",
]

_PI = 3.141592653589793

#: Padding values used to build the internal padded tensors. They are never read
#: on a row's invalid span (every stage masks it), which the unit gate proves by
#: poisoning them. Kept module-level so the test can override them.
PAD_TOKEN_ID = 0
PAD_VALUE = 0.0
#: Padding value for the padded mel batch fed to the source tokenizer. The
#: tokenizer's encoder masks every padded frame before its first conv, so the
#: value is never read; the poison gate proves it.
PAD_MEL_VALUE = 0.0


class S3GenBatchError(RuntimeError):
    """The S3Gen graph cannot be batched for this request set (audit-hostile).

    Raised before any forward so the handler can fall back to the per-row
    upstream call without a partial result.
    """


@dataclass
class S3GenRow:
    """One finished AR row's S3Gen input.

    ``tokens`` is a 1-D long tensor of speech tokens (already filtered by the
    contract); ``ref_dict`` is that request's ``Conditionals.gen`` mapping
    (``prompt_token`` / ``prompt_token_len`` / ``prompt_feat`` /
    ``prompt_feat_len`` / ``embedding``); ``seed`` drives this row's own noise
    stream (``None`` keeps the global-RNG behaviour for an unseeded request).
    """

    tokens: Tensor
    ref_dict: dict
    seed: int | None = None
    _token_len: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tokens = self.tokens
        if tokens.dim() == 2 and tokens.size(0) == 1:
            tokens = tokens[0]
        if tokens.dim() != 1:
            raise S3GenBatchError(f"S3GenRow.tokens must be 1-D, got {tuple(tokens.shape)}")
        self.tokens = tokens
        self._token_len = int(tokens.numel())


# ---------------------------------------------------------------------------
# Capability / normalisation
# ---------------------------------------------------------------------------
def supports_batched_s3gen(s3gen) -> bool:
    """True when this S3Gen has the standard (non-meanflow) audited graph."""
    if s3gen is None:
        return False
    if bool(getattr(s3gen, "meanflow", False)):
        return False
    return hasattr(s3gen, "flow") and hasattr(s3gen, "mel2wav")


def supports_batched_tokenizer(tokenizer) -> bool:
    """True when this source tokenizer exposes the audited mel + quantize split."""
    return tokenizer is not None and hasattr(tokenizer, "log_mel_spectrogram") and hasattr(
        tokenizer, "quantize"
    )


def _flow_device(flow) -> torch.device:
    return next(flow.parameters()).device


def _flow_dtype(flow) -> torch.dtype:
    return next(flow.parameters()).dtype


def _to_flow(value, device, dtype) -> Tensor | None:
    """Move one ``ref_dict`` value to the flow's device/dtype (upstream order).

    ``None`` passes through: real ``embed_ref`` dicts carry ``prompt_feat_len:
    None`` (upstream reads the prompt length from ``prompt_feat.shape``), and the
    B=1 unseeded fast path forwards every key to the upstream call.
    """
    if value is None:
        return None
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def _generators(rows, device) -> list[torch.Generator | None]:
    """One ``torch.Generator`` per row from its request seed (``None`` = global)."""
    gen_device = "cpu" if torch.device(device).type == "cpu" else str(device)
    out: list[torch.Generator | None] = []
    for row in rows:
        if row.seed is None:
            out.append(None)
        else:
            generator = torch.Generator(device=gen_device)
            generator.manual_seed(int(row.seed))
            out.append(generator)
    return out


def _draw(shape, generator, *, device, dtype) -> Tensor:
    if generator is None:
        return torch.randn(*shape, device=device, dtype=dtype)
    return torch.randn(*shape, device=device, dtype=dtype, generator=generator)


# ---------------------------------------------------------------------------
# Flow: encoder (per-stage mask) + CFM decoder (corrected mask)
# ---------------------------------------------------------------------------
def _encode_masked(encoder, token_embeds: Tensor, token_lens: Tensor) -> tuple[Tensor, Tensor]:
    """``UpsampleConformerEncoder`` forward with each row's padding held at zero.

    Upstream's ``LinearNoSubsampling`` embed adds a bias (so padding is no
    longer zero) and ``PreLookaheadLayer`` is a causal conv that reads forward —
    together a 0.4075 peak leak. The conformer attention itself already masks
    padded keys, so the only extra work is zeroing the padding before the
    lookahead, after it, and before the upsample stage.
    """
    padded_len = token_embeds.size(1)
    mask = (~make_pad_mask(token_lens, padded_len)).unsqueeze(1)
    hidden, pos_emb, mask = encoder.embed(token_embeds, mask)
    hidden = hidden * mask.transpose(1, 2)
    hidden = encoder.pre_lookahead_layer(hidden) * mask.transpose(1, 2)
    hidden = encoder.forward_layers(hidden, mask, pos_emb, mask) * mask.transpose(1, 2)

    hidden = hidden.transpose(1, 2).contiguous()
    hidden, up_lens = encoder.up_layer(hidden, token_lens)
    hidden = hidden.transpose(1, 2).contiguous()
    up_len = hidden.size(1)
    up_mask = (~make_pad_mask(up_lens, up_len)).unsqueeze(1)
    hidden, pos_emb, up_mask = encoder.up_embed(hidden, up_mask)
    hidden = hidden * up_mask.transpose(1, 2)
    hidden = encoder.forward_up_layers(hidden, up_mask, pos_emb, up_mask)
    return encoder.after_norm(hidden), up_mask


def _flow_batched(
    flow,
    rows: list[S3GenRow],
    generators: list[torch.Generator | None],
    *,
    n_timesteps: int,
    precision=None,
) -> list[Tensor]:
    """Tokens + per-row prompts → per-row generated mels ``[1, 80, 2·T_i]``."""
    device = _flow_device(flow)
    dtype = _flow_dtype(flow)
    batch = len(rows)

    tokens = [row.tokens.to(device=device, dtype=torch.long) for row in rows]
    token_lens = [row._token_len for row in rows]

    prompt_tokens: list[Tensor] = []
    prompt_lens: list[int] = []
    prompt_feats: list[Tensor] = []
    embeddings: list[Tensor] = []
    for index, row in enumerate(rows):
        ref = row.ref_dict or {}
        prompt_token = _to_flow(ref["prompt_token"], device, dtype).reshape(-1).long()
        prompt_token_len = ref.get("prompt_token_len")
        if prompt_token_len is None:
            p_len = int(prompt_token.numel())
        else:
            p_len = int(torch.as_tensor(prompt_token_len).reshape(-1)[0].item())
        if p_len > prompt_token.numel():
            raise S3GenBatchError(
                f"prompt_token_len {p_len} exceeds prompt_token width {prompt_token.numel()}"
            )
        prompt_feat = _to_flow(ref["prompt_feat"], device, dtype)
        if prompt_feat.dim() == 3:
            if prompt_feat.size(0) != 1:
                raise S3GenBatchError("prompt_feat must be one row per ref_dict")
            prompt_feat = prompt_feat[0]
        if prompt_feat.dim() != 2 or prompt_feat.size(-1) != 80:
            raise S3GenBatchError(f"prompt_feat must be [frames, 80], got {tuple(prompt_feat.shape)}")
        # Upstream ``CausalMaskedDiffWithXvec.inference`` takes the prompt mel
        # length from ``prompt_feat.shape[1]`` itself (not from the token count)
        # and derives the generated span as ``h.shape[1] - mel_len1``. The prompt
        # is therefore only required to leave at least one generated frame in the
        # row's combined encoded length (2 frames per token). It is **not**
        # exactly ``2 * prompt_token_len``: upstream ``S3Gen.embed_ref`` trims the
        # tokens to ``mel_frames // 2`` and logs a warning, so a real reference
        # measures e.g. 181 frames for 90 tokens. The old strict 2× check made
        # every real VC target fall back to the loop (Phase 11-1 finding).
        combined_frames = 2 * (p_len + token_lens[index])
        if prompt_feat.size(0) >= combined_frames:
            raise S3GenBatchError(
                f"prompt_feat has {prompt_feat.size(0)} frames; the row's "
                f"{combined_frames} encoded frames leave no generated span"
            )
        prompt_tokens.append(prompt_token[:p_len])
        prompt_lens.append(p_len)
        prompt_feats.append(prompt_feat)
        embedding = _to_flow(ref["embedding"], device, dtype)
        embeddings.append(embedding.reshape(-1))

    # Combined per-row token sequence [prompt | speech], padded on the token axis.
    combined_len = max(prompt_lens[i] + token_lens[i] for i in range(batch))
    combined = torch.full((batch, combined_len), PAD_TOKEN_ID, dtype=torch.long, device=device)
    for i in range(batch):
        p_len, t_len = prompt_lens[i], token_lens[i]
        combined[i, :p_len] = prompt_tokens[i]
        combined[i, p_len : p_len + t_len] = tokens[i]

    total_lens = torch.tensor(
        [prompt_lens[i] + token_lens[i] for i in range(batch)], dtype=torch.long, device=device
    )
    embedding = torch.nn.functional.normalize(torch.stack(embeddings), dim=1)
    embedding = flow.spk_embed_affine_layer(embedding)

    token_mask = (~make_pad_mask(total_lens, combined_len)).unsqueeze(-1).to(embedding)
    token_embeds = flow.input_embedding(combined) * token_mask

    # Phase 11-2/11-3: the surgical wrap. Autocast covers the encoder and CFM
    # decoder forwards only; the returned tensors are cast back to the flow
    # dtype so every ODE state update, mask, unpad and `mel2wav` call below
    # stays float32. With the effective precision fp32 (config default and/or
    # `HSE_S3GEN_AUTOCAST` unset) the context is a no-op and the casts are
    # skipped, so this path is bit-identical.
    with s3gen_autocast(device, precision=precision) as autocast:
        hidden, up_mask = _encode_masked(flow.encoder, token_embeds, total_lens)
        if autocast is not None:
            hidden = hidden.to(dtype)
            if up_mask.is_floating_point():
                up_mask = up_mask.to(dtype)
    hidden_lens = up_mask.sum(dim=-1).squeeze(-1)
    hidden = flow.encoder_proj(hidden)

    # The conditioning mel is the prompt prefix followed by zeros for the
    # generated span (upstream leaves the rest of ``conds`` at zero). Only the
    # invalid padding past ``hidden_lens`` is masked, so the zeros there are not
    # a "poisonable" input padding.
    conds = torch.zeros(batch, hidden.size(1), 80, device=device, dtype=hidden.dtype)
    for i, feat in enumerate(prompt_feats):
        conds[i, : feat.size(0)] = feat

    positions = torch.arange(hidden.size(1), device=device)
    mask = (positions[None, :] < hidden_lens[:, None]).unsqueeze(1).to(hidden)

    # Per-row start noise, drawn at the row's own valid length so it matches the
    # solo draw; the padded tail is never read (the decoder masks it).
    noise = torch.full(
        (batch, 80, hidden.size(1)), PAD_VALUE, device=device, dtype=hidden.dtype
    )
    for i in range(batch):
        valid = int(hidden_lens[i])
        noise[i, :, :valid] = _draw(
            (80, valid), generators[i], device=device, dtype=hidden.dtype
        )

    with s3gen_autocast(device, precision=precision) as autocast:
        feat, _ = flow.decoder(
            mu=hidden.transpose(1, 2).contiguous(),
            mask=mask,
            spks=embedding,
            cond=conds.transpose(1, 2),
            n_timesteps=n_timesteps,
            noised_mels=noise,
            meanflow=False,
        )
        if autocast is not None:
            feat = feat.to(dtype)
    mels: list[Tensor] = []
    for i in range(batch):
        start = int(prompt_feats[i].size(0))
        end = int(hidden_lens[i])
        mels.append(feat[i : i + 1, :, start:end].contiguous())
    return mels


# ---------------------------------------------------------------------------
# HiFT mel→waveform: per-stage masks, per-row source RNG, per-row STFT/ISTFT
# ---------------------------------------------------------------------------
def _masked_resblock(block, hidden: Tensor, mask: Tensor) -> Tensor:
    """``ResBlock.forward`` with each conv's output held at zero on padding."""
    for index in range(len(block.convs1)):
        branch = block.activations1[index](hidden)
        branch = block.convs1[index](branch) * mask
        branch = block.activations2[index](branch)
        branch = block.convs2[index](branch) * mask
        hidden = branch + hidden
    return hidden * mask


def _masked_f0_predictor(predictor, mel: Tensor, mask: Tensor) -> Tensor:
    """``ConvRNNF0Predictor`` with the padding zeroed after every conv."""
    hidden = mel
    for module in predictor.condnet:
        hidden = module(hidden)
        if getattr(module, "kernel_size", None) is not None:
            hidden = hidden * mask
    f0 = torch.abs(predictor.classifier(hidden.transpose(1, 2)).squeeze(-1))
    return f0 * mask[:, 0, :]


def _source_per_row(m_source, source: Tensor, generators, valid_lens) -> Tensor:
    """Re-implementation of ``SourceModuleHnNSF.forward`` with per-row RNG.

    Upstream draws three tensors from the global RNG (``SineGen`` phase, its
    noise, and ``SourceModule``'s extra noise). Each is drawn at the row's own
    valid length so the row reproduces its solo stream exactly.
    """
    sine_gen = m_source.l_sin_gen
    batch, _, length = source.shape
    device = source.device
    harmonics = sine_gen.harmonic_num

    f_mat = torch.zeros(batch, harmonics + 1, length, device=device, dtype=source.dtype)
    for harmonic in range(harmonics + 1):
        f_mat[:, harmonic : harmonic + 1, :] = source * (harmonic + 1) / sine_gen.sampling_rate
    theta = 2 * _PI * (torch.cumsum(f_mat, dim=-1) % 1)

    phase = torch.zeros(batch, harmonics + 1, 1, device=device, dtype=source.dtype)
    for row in range(batch):
        generator = generators[row]
        if generator is None:
            phase[row] = torch.empty(harmonics + 1, 1, device=device).uniform_(-_PI, _PI)
        else:
            phase[row] = torch.empty(harmonics + 1, 1, device=device).uniform_(
                -_PI, _PI, generator=generator
            )
    phase[:, 0, :] = 0

    sine = sine_gen.sine_amp * torch.sin(theta + phase)
    voiced = (source > sine_gen.voiced_threshold).to(source.dtype)
    noise_amp = voiced * sine_gen.noise_std + (1 - voiced) * sine_gen.sine_amp / 3

    noise = torch.zeros_like(sine)
    for row in range(batch):
        valid = int(valid_lens[row])
        noise[row, :, :valid] = _draw(
            (harmonics + 1, valid), generators[row], device=device, dtype=source.dtype
        )
    sine = sine * voiced + noise_amp * noise

    # SourceModuleHnNSF draws one more tensor and discards it; reproduce the draw
    # so the per-row stream stays aligned with the solo run.
    for row in range(batch):
        valid = int(valid_lens[row])
        _draw((1, valid), generators[row], device=device, dtype=source.dtype)

    merged = torch.tanh(m_source.l_linear(sine.transpose(1, 2)))
    return merged.transpose(1, 2)  # [B, 1, L]


def _hift_batched(mel2wav, mels: list[Tensor], generators) -> list[Tensor]:
    """Per-row HiFT inference; returns one 1-D waveform per row."""
    batch = len(mels)
    mel_lens = [int(mel.size(-1)) for mel in mels]
    max_frames = max(mel_lens)
    device = mels[0].device
    dtype = mels[0].dtype

    padded = torch.full((batch, 80, max_frames), PAD_VALUE, device=device, dtype=dtype)
    for i, mel in enumerate(mels):
        padded[i, :, : mel_lens[i]] = mel[0]

    mask = length_mask(mel_lens, max_frames, device=device, dtype=dtype)
    padded = padded * mask

    up_factor = 1
    for up in mel2wav.ups:
        up_factor *= int(up.stride[0])
    hop = int(mel2wav.istft_params["hop_len"])
    sample_lens = [frames * up_factor * hop for frames in mel_lens]

    f0 = _masked_f0_predictor(mel2wav.f0_predictor, padded, mask)
    source = mel2wav.f0_upsamp(f0[:, None])  # [B, 1, L]
    source_len = source.shape[-1]
    source_mask = length_mask(sample_lens, source_len, device=device, dtype=dtype)
    source = source * source_mask
    source = _source_per_row(mel2wav.m_source, source, generators, sample_lens) * source_mask

    # torch.stft reflects at the signal end, so a padded batch would not match a
    # solo row. Run the STFT per row on the row's own samples and stitch.
    stft_lens = [1 + samples // hop for samples in sample_lens]
    stft_max = max(stft_lens)
    # The STFT buffer's padding stands in for the convolution's implicit zero
    # padding, so it must stay zero (a poisoned value there would be read by the
    # source convs at a row's boundary). It is an internal buffer, not input
    # padding; the input padded regions are the ones the gate poisons.
    stft = torch.zeros(
        batch, mel2wav.istft_params["n_fft"] + 2, stft_max, device=device, dtype=dtype
    )
    for row in range(batch):
        real, imag = mel2wav._stft(source[row, 0, : sample_lens[row]])
        stft[row, :, : stft_lens[row]] = torch.cat([real, imag], dim=0)

    hidden = mel2wav.conv_pre(padded) * mask
    for stage in range(mel2wav.num_upsamples):
        hidden = torch.nn.functional.leaky_relu(hidden, mel2wav.lrelu_slope)
        hidden = mel2wav.ups[stage](hidden)
        mask = sync_mask(mask, hidden.shape[-1])
        if stage == mel2wav.num_upsamples - 1:
            hidden = mel2wav.reflection_pad(hidden)
            # ReflectionPad1d((1, 0)) mirrors the row's own first frame, so the
            # prepended frame is valid (mask 1), not padding (mask 0).
            mask = torch.nn.functional.pad(mask, (1, 0), value=1.0)
        source_part = mel2wav.source_downs[stage](stft) * mask
        source_part = _masked_resblock(mel2wav.source_resblocks[stage], source_part, mask)
        hidden = (hidden + source_part) * mask
        residual = None
        for kernel in range(mel2wav.num_kernels):
            block = mel2wav.resblocks[stage * mel2wav.num_kernels + kernel]
            branch = _masked_resblock(block, hidden, mask)
            residual = branch if residual is None else residual + branch
        hidden = (residual / mel2wav.num_kernels) * mask

    hidden = torch.nn.functional.leaky_relu(hidden)
    hidden = mel2wav.conv_post(hidden) * mask
    magnitude = torch.exp(hidden[:, : mel2wav.istft_params["n_fft"] // 2 + 1, :]) * mask
    phase = torch.sin(hidden[:, mel2wav.istft_params["n_fft"] // 2 + 1 :, :]) * mask

    waveforms: list[Tensor] = []
    for row in range(batch):
        frames = stft_lens[row]
        piece = mel2wav._istft(magnitude[row : row + 1, :, :frames], phase[row : row + 1, :, :frames])
        piece = torch.clamp(piece[0, : sample_lens[row]], -mel2wav.audio_limit, mel2wav.audio_limit)
        waveforms.append(piece)
    return waveforms


# ---------------------------------------------------------------------------
# Source tokenizer: per-row mel (STFT is padding-hostile) + one quantize
# ---------------------------------------------------------------------------
def _as_clip(clip, index: int) -> Tensor:
    """Normalise one source clip to a 1-D float tensor (no device move yet)."""
    tensor = clip if torch.is_tensor(clip) else torch.as_tensor(clip)
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor[0]
    if tensor.dim() != 1:
        raise S3GenBatchError(
            f"source clip {index} must be 1-D, got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor


@torch.inference_mode()
def tokenize_batched(tokenizer, clips) -> list[Tensor]:
    """Tokenize a batch of 16 kHz source clips in one encoder/quantizer forward.

    The padding audit (``11-1-ChatterboxVCBatching.md``) measured, on a tiny
    seeded instance of the real classes:

    * **STFT / log-mel — padding-hostile.** ``torch.stft`` reflects at the
      signal end, so a padded row's last valid mel frame reads zeros instead of
      the row's own reflection (measured **0.21–0.51** max |Δmel| at that one
      frame). The STFT is therefore run **per row**, exactly like upstream's own
      per-row ``log_mel_spectrogram`` call.
    * **``AudioEncoderV2`` + ``FSQVectorQuantization`` — safe by construction.**
      The encoder masks each row's padding before ``conv1`` and again before
      ``conv2``; the FSMN block zeroes padded inputs/outputs and the attention
      masks padded keys. With the per-row mels padded on the frame axis, the
      valid hidden states were bit-identical (0.0) and every token matched its
      solo run (0 mismatches); poisoned mel padding (``1e3``) changes nothing.

    ``clips`` is a list of 1-D float tensors (or numpy arrays) at
    ``S3_SR`` (16 kHz). Returns one 1-D long token tensor per clip, sliced with
    the tokenizer's **own** per-row token length (never a padded-shape
    assumption). With one clip this is bit-identical to upstream
    ``tokenizer(clip[None])`` — the sequential fast path.

    Raises :class:`S3GenBatchError` before any forward when the tokenizer lacks
    the audited split, so the caller can fall back to the per-row call.
    """
    if not clips:
        return []
    if not supports_batched_tokenizer(tokenizer):
        raise S3GenBatchError("source tokenizer has no audited mel/quantize split")

    device = tokenizer.device
    mels: list[Tensor] = []
    for index, clip in enumerate(clips):
        waveform = _as_clip(clip, index)
        mel = tokenizer.log_mel_spectrogram(waveform[None, :].to(device))
        mels.append(mel.squeeze(0))

    channels = int(mels[0].size(0))
    frame_lens = [int(mel.size(-1)) for mel in mels]
    width = max(frame_lens)
    padded = torch.full(
        (len(mels), channels, width), PAD_MEL_VALUE, device=device, dtype=mels[0].dtype
    )
    for row, mel in enumerate(mels):
        if int(mel.size(0)) != channels:
            raise S3GenBatchError(
                f"source clip {row} produced {mel.size(0)} mel bins, expected {channels}"
            )
        padded[row, :, : frame_lens[row]] = mel.to(device)

    mel_lens = torch.tensor(frame_lens, dtype=torch.long, device=device)
    tokens, token_lens = tokenizer.quantize(padded, mel_lens)
    return [tokens[row, : int(token_lens[row])].clone() for row in range(len(mels))]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
@torch.inference_mode()
def inference_batched(
    s3gen,
    rows: list[S3GenRow],
    *,
    n_cfm_timesteps: int | None = None,
) -> list[Tensor]:
    """Vocode ``rows`` in one padded S3Gen forward; returns one 1-D waveform each.

    A single unseeded row calls upstream ``s3gen.inference`` unchanged (the
    sequential fast path / legacy behaviour). Otherwise the padded graph runs
    with per-row seeded noise and per-row trimming; every row's waveform is the
    same as if it had been vocoded alone with that seed (up to float32
    batch-kernel accumulation; the tiny gate measured ≤2.1e-7, asserted ≤1e-6).

    Raises :class:`S3GenBatchError` before any forward when the graph cannot be
    batched (meanflow/turbo, missing modules, inconsistent ref_dict) so the
    handler can fall back cleanly.
    """
    if not rows:
        return []
    if not supports_batched_s3gen(s3gen):
        raise S3GenBatchError("S3Gen graph is not audited for padded batching")

    # Phase 11-2/11-3: the loader installs the autocast seam at model load;
    # installing again is idempotent and keeps a hand-built S3Gen (tests,
    # probes) covered. It matters here only for the B=1 unseeded fast path
    # below, which calls the upstream `s3gen.inference`; the padded path wraps
    # its own forwards with the same per-model precision.
    install_inference_autocast(s3gen)
    precision = getattr(s3gen, "_hse_s3gen_precision", S3GEN_PRECISION_FP32)

    flow = s3gen.flow
    device = _flow_device(flow)
    dtype = _flow_dtype(flow)
    timesteps = n_cfm_timesteps or (2 if getattr(s3gen, "meanflow", False) else 10)

    if len(rows) == 1 and rows[0].seed is None:
        row = rows[0]
        ref = {
            key: _to_flow(value, device, dtype) for key, value in (row.ref_dict or {}).items()
        }
        wav, _ = s3gen.inference(
            speech_tokens=row.tokens.to(device=device, dtype=torch.long).view(1, -1),
            ref_dict=ref,
            n_cfm_timesteps=timesteps,
        )
        return [wav[0].detach()]

    generators = _generators(rows, device)
    mels = _flow_batched(flow, rows, generators, n_timesteps=timesteps, precision=precision)
    waveforms = _hift_batched(s3gen.mel2wav, mels, generators)

    trim_fade = getattr(s3gen, "trim_fade", None)
    if trim_fade is not None and not getattr(s3gen, "training", False):
        fade = trim_fade.to(device=waveforms[0].device if waveforms else device)
        for waveform in waveforms:
            span = min(fade.numel(), waveform.numel())
            waveform[:span] = waveform[:span] * fade[:span]
    return waveforms
