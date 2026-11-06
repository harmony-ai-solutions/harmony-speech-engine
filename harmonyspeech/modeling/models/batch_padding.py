"""Padding isolation for the batched conv vocoders (Phase 8-4).

The Phase 8-3 lesson generalised: a padded row inside a convolutional decoder is
**not** harmless extra context. A solo run only ever sees its own frames plus the
conv's implicit zero padding, while a padded row hands the next layer a full
frame of activations that were computed from somebody else's padding — and every
transposed/dilated conv then spreads that difference back into the valid tail.
The measured effect on the VoiceFixer and MelGAN vocoders is large (0.28 and
2e-3 peak on random weights, before the fix).

Two rules make a padded batch reproduce each row's solo output:

* **Mask after every stage.** Each row's activations are multiplied by a
  ``[B, 1, T]`` mask of its valid span after every leaf module, so the padding
  region holds exactly the zeros a solo conv would pad with.
* **Reflect with the row's own tail.** ``ReflectionPad1d`` mirrors values that
  are already inside the tensor; for a padded row it would mirror the padding
  instead of the row's own last frames. The pad is therefore handled together
  with the convolution that consumes it: the tensor is extended by the pad
  amount, each row's mirrored tail is copied into its first padding positions,
  and the pair is applied as one step.

Masks are built here rather than in the models so both vocoders (and their
tests) share one implementation. Nothing here runs a model: callers pass their
own module lists in through :func:`run_masked`.
"""

from __future__ import annotations

import torch
import torch.nn.functional

__all__ = ["length_mask", "sync_mask", "fill_reflection_tail", "run_masked"]


def length_mask(lengths, max_len: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return ``[B, 1, max_len]`` with 1.0 on each row's valid span, else 0.0."""
    values = [int(value) for value in lengths]
    if not values:
        raise ValueError("length_mask: empty length list")
    if max(values) > int(max_len):
        raise ValueError(f"length_mask: a row declares {max(values)} frames, batch holds {max_len}")
    mask = torch.zeros((len(values), 1, int(max_len)), device=device, dtype=dtype)
    for row, value in enumerate(values):
        mask[row, :, : max(0, value)] = 1.0
    return mask


def sync_mask(mask: torch.Tensor, frame_length: int) -> torch.Tensor:
    """Resize a mask to the time length a strided layer just produced.

    Only the two shapes a strided (transposed) convolution can produce are
    accepted: unchanged (stride 1) or an integer multiple (stride ``u``), where
    each input frame owns ``u`` output frames. Anything else means the model
    chain is not a plain up/down sampling stack, and the caller must fall back to
    the per-item loop rather than guess an alignment.
    """
    current = int(mask.shape[-1])
    target = int(frame_length)
    if target == current:
        return mask
    if target > current and target % current == 0:
        return mask.repeat_interleave(target // current, dim=-1)
    raise ValueError(f"sync_mask: cannot align a {current}-frame mask with a {target}-frame stage")


def row_lengths(mask: torch.Tensor) -> list[int]:
    """Per-row valid frame counts of a 0/1 ``[B, 1, T]`` mask."""
    return [int(round(float(value))) for value in mask[:, 0, :].sum(dim=-1)]


def fill_reflection_tail(x: torch.Tensor, mask: torch.Tensor, pad: int) -> torch.Tensor:
    """Copy each row's mirrored tail into its first ``pad`` padding positions.

    ``ReflectionPad1d`` mirrors values from inside the tensor, so on a padded
    batch a short row would mirror its padding (zeros) where its solo run mirrors
    its own last frames. Filling those positions first makes the following
    convolution see the row's solo window. The caller extends the tensor by
    ``pad`` frames first, so every row has room for its mirror.
    """
    pad = int(pad)
    if pad <= 0:
        return x
    total = int(x.shape[-1])
    for row, length in enumerate(row_lengths(mask)):
        end = min(length + pad, total)
        if end <= length:
            continue
        offset = torch.arange(1, end - length + 1, device=x.device)
        source = torch.clamp(length - 1 - offset, min=0)
        x[row, :, length:end] = x[row, :, source]
    return x


def _as_pad(padding) -> tuple[int, int]:
    if isinstance(padding, int):
        return padding, padding
    left, right = padding
    return int(left), int(right)


def run_masked(modules, x: torch.Tensor, mask: torch.Tensor, *, custom=None):
    """Apply ``modules`` in order, holding each row's padding at zero.

    ``custom`` maps a module type to ``(module, x, mask) -> (x, mask)`` for the
    blocks whose forward is not a plain leaf op (the vocoders' upsampling and
    residual blocks, which mix per-frame and strided paths). Everything else is
    applied as a leaf: elementwise activations keep the mask, convolution-style
    layers resync it to their output length (:func:`sync_mask`) and a
    ``ReflectionPad1d`` is folded into the ``Conv1d`` that consumes it so each
    row reflects its own tail (:func:`fill_reflection_tail`).

    Returns ``(x, mask)`` with ``mask`` matching ``x``'s time axis.
    """
    custom = custom or {}
    modules = list(modules)
    index = 0
    while index < len(modules):
        module = modules[index]
        handler = custom.get(type(module))
        if handler is not None:
            x, mask = handler(module, x, mask)
            index += 1
            continue
        if isinstance(module, torch.nn.ReflectionPad1d):
            if index + 1 >= len(modules) or not isinstance(modules[index + 1], torch.nn.Conv1d):
                raise ValueError("run_masked: a reflection pad must be followed by its Conv1d")
            left, right = _as_pad(module.padding)
            if left != right:
                raise ValueError("run_masked: asymmetric reflection pads are not batched")
            conv = modules[index + 1]
            original = int(x.shape[-1])
            x = torch.nn.functional.pad(x, (0, left))
            mask = torch.nn.functional.pad(mask, (0, left))
            x = fill_reflection_tail(x, mask, left)
            x = conv(module(x))
            if int(x.shape[-1]) != original + left:
                raise ValueError(
                    f"run_masked: reflection pad {left} with kernel {conv.kernel_size} "
                    f"produced {x.shape[-1]} frames for {original}"
                )
            x = x[:, :, :original]
            mask = mask[:, :, :original]
            x = x * mask
            index += 2
            continue
        if isinstance(module, (torch.nn.ConstantPad1d, torch.nn.ReplicationPad1d)):
            # Asymmetric (causal) padding: the pad amount is not part of the
            # valid span, and the matching causal conv trims it back.
            x = module(x)
            index += 1
            continue
        x = module(x)
        mask = sync_mask(mask, x.shape[-1])
        x = x * mask
        index += 1
    return x, mask
