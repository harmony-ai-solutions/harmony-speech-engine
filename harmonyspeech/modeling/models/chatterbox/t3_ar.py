"""Chatterbox T3 with a hygiene-cleaned autoregressive decode loop (Phase 9-1, AR-0).

Why this module exists
----------------------
``T3`` lives in the installed upstream package (``chatterbox-tts==0.1.6``); HSE
cannot edit site-packages. Study #14 found two pieces of pure-loss work in the
hottest loop of the engine's slowest family (Chatterbox TTS, flat at ~0.6 req/s
from c=1 to c=8):

* ``chatterbox/models/t3/t3.py`` asks the HF Llama forward for attention tensors
  on **every** step — ``output_attentions=True`` at ``t3.py:344`` (prefill) and
  ``t3.py:403`` (per decode step) — although nobody reads them on the English
  path. The only consumer is the multilingual ``AlignmentStreamAnalyzer``, which
  reads the attention *hooks* (and therefore needs the eager attention path).
* a ``tqdm`` progress bar wraps the per-token loop (``t3.py:352``), writing to
  stderr once per request.

Verified 2026-11-07 in the installed package: the decode loop is
``t3.py:352-408`` (Python per-token loop over ``t3.tfmr`` with an HF
``DynamicCache``); the CFG combine / sampler calls are ``t3.py:353-382``.

The seam
--------
This module defines :class:`ChatterboxT3AR`, a subclass of upstream ``T3`` that
overrides **only** the inference path. ``factory.py`` instantiates it in place of
the plain class for the Chatterbox TTS role (the embedding role shares that T3
and never decodes). Nothing is monkey-patched: the process-wide ``torch.load``
patch in ``chatterbox.py`` is the anti-pattern this deliberately does not
extend.

The decode path is deliberately *not* one opaque method. Phase 9-3 drives the
same modules with its own batched loop against a KV-cache pool, so each piece is
overridable and parameterised:

``ar_prefill`` / ``ar_forward``
    conditioning + text + BOS composition and the first (uncached) forward.
``ar_token_embedding`` / ``ar_cfg_expand`` / ``ar_step``
    one decode step: embed the sampled token at its position, duplicate it for
    the CFG rows, forward it against the cache.
``T3ARSampler`` / ``ar_sample``
    the sampler call: CFG combine, analyzer hook (multilingual only),
    repetition penalty, temperature, min-p/top-p warpers, multinomial.
``ar_progress``
    the loop's iterator; a no-op unless the debug env var is set.

Numerical inertness
-------------------
Dropping ``output_attentions`` only stops the forward from *returning* extra
tensors. On this path the returned tensors are never read (the English model has
no analyzer), so outputs must stay bit-identical; the equality gate in
``tests/unit/modeling/test_chatterbox_ar0.py`` proves it on a tiny seeded model
for both ``eager`` and ``sdpa`` attention. Note for the record: with
``attn_implementation="sdpa"`` (the shipped English config) transformers 4.57
already ignores ``output_attentions=True`` and returns ``attentions=None`` with a
warning — so on the shipped TTS path the flag costs a per-step warning rather
than an attention materialisation, and the tqdm hook is the measurable loss.
With ``eager`` (the multilingual path, which HSE forces) the flag really does
materialise attention weights.

Scope
-----
* **TTS role only.** Turbo uses a structurally different loop
  (``T3.inference_turbo``, ``t3.py:415-490``: raw ``self.tfmr``, a
  ``LogitsProcessorList``, no CFG, no ``T3HuggingfaceBackend``) whose only
  hygiene issue is its own ``tqdm`` (``t3.py:457``); it is left to 9-3.
* **Multilingual untouched.** Its analyzer needs attention, and per-request
  analyzer state on a shared Llama is exactly what 9-3 documents as a skip. This
  class still supports it faithfully if ever used (attention stays on when an
  analyzer is attached).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from chatterbox.models.t3 import T3
from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from loguru import logger
from torch import Tensor
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

#: Opt-in debug flag: set to ``1``/``true``/``yes``/``on`` to bring back
#: upstream's per-token progress bar. Default off — the bar is a per-request
#: stderr hook in the hottest loop and has no place in serving.
PROGRESS_ENV = "HSE_CHATTERBOX_T3_PROGRESS"

_TRUTHY = {"1", "true", "yes", "on"}


def _ensure_bot_eot(text_tokens: Tensor, hp) -> None:
    """Mirror of upstream ``t3._ensure_BOT_EOT`` (same assertions, same order).

    Re-implemented locally instead of importing the private upstream helper so a
    patch release that renames it cannot break this module's import.
    """
    batch = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= batch, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= batch, "missing stop_text_token"


@dataclass
class T3ARDecodeState:
    """Mutable state of one decode loop.

    ``generated_ids`` starts as the BOS token and grows one column per step;
    ``logits`` is the last position's logits (``(2, V)`` — CFG conditional and
    unconditional rows) and ``past_key_values`` the HF cache after that forward.
    """

    generated_ids: Tensor
    logits: Tensor
    past_key_values: object
    len_cond: int


@dataclass
class T3ARSampler:
    """Per-request sampler: CFG combine → penalties/warpers → multinomial.

    Constructed once per ``inference`` call (upstream builds its warpers there
    too). Phase 9-3 can reuse it as-is for its batched per-row sampler, or
    override :meth:`ChatterboxT3AR.ar_sample`.
    """

    cfg_weight: float = 0.5
    temperature: float = 0.8
    top_p: float = 0.95
    min_p: float = 0.05
    repetition_penalty: float = 1.2
    _min_p_warper: MinPLogitsWarper = field(init=False, repr=False)
    _top_p_warper: TopPLogitsWarper = field(init=False, repr=False)
    _repetition_penalty: RepetitionPenaltyLogitsProcessor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._min_p_warper = MinPLogitsWarper(min_p=self.min_p)
        self._top_p_warper = TopPLogitsWarper(top_p=self.top_p)
        self._repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=float(self.repetition_penalty))

    def combine_cfg(self, logits_step: Tensor) -> Tensor:
        """Classifier-free-guidance combine of the conditional/unconditional rows."""
        cond = logits_step[0:1, :]
        uncond = logits_step[1:2, :]
        cfg = torch.as_tensor(self.cfg_weight, device=cond.device, dtype=cond.dtype)
        return cond + cfg * (cond - uncond)

    def pick(self, logits: Tensor, generated_ids: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        """Apply the processors and draw one token (upstream order, one row).

        ``generator`` is the Phase 9-3 per-row RNG: a generator seeded exactly
        like the global RNG produces the same draw as the sequential loop, while
        two rows keep independent streams (a row's tokens are identical alone or
        batched). ``None`` keeps upstream's global-RNG behaviour.
        """
        ids_for_proc = generated_ids[:1, ...]  # batch = 1
        logits = self._repetition_penalty(ids_for_proc, logits)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        logits = self._min_p_warper(ids_for_proc, logits)
        logits = self._top_p_warper(ids_for_proc, logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)  # (1, 1)


class ChatterboxT3AR(T3):
    """Upstream ``T3`` with the AR-0 decode-loop hygiene applied.

    Behavioural contract: for every input the sampled token sequence is
    bit-identical to upstream ``T3.inference`` (proven by the equality gate);
    the only differences are that attention tensors are not requested when no
    analyzer is attached and that no progress bar is drawn by default.
    """

    #: See :data:`PROGRESS_ENV`.
    PROGRESS_ENV = PROGRESS_ENV

    # ------------------------------------------------------------------
    # Loop plumbing
    # ------------------------------------------------------------------
    @classmethod
    def progress_enabled(cls) -> bool:
        """True only when the operator opted into the debug progress bar."""
        return str(os.environ.get(cls.PROGRESS_ENV, "")).strip().lower() in _TRUTHY

    def ar_progress(self, iterable: Iterable) -> Iterable:
        """Return ``iterable`` unchanged unless the debug progress bar is enabled."""
        if not self.progress_enabled():
            return iterable
        from tqdm import tqdm

        return tqdm(iterable, desc="Sampling", dynamic_ncols=True)

    def ar_output_attentions(self, backend: T3HuggingfaceBackend) -> bool:
        """Whether the backbone forward must return attention tensors.

        Only the multilingual ``AlignmentStreamAnalyzer`` reads them (through
        attention hooks on the eager path); without it the extra tensors are
        dead weight — and on the ``sdpa`` path transformers ignores the request
        anyway.
        """
        return getattr(backend, "alignment_stream_analyzer", None) is not None

    def build_ar_backend(
        self, *, len_cond: int | None = None, text_len: int | None = None
    ) -> T3HuggingfaceBackend:
        """Build the HF-generation backend over the existing modules.

        Mirrors upstream ``t3.py:277-298``. Multilingual models need the text
        slice, so ``len_cond``/``text_len`` are required for them.
        """
        analyzer = None
        if self.hp.is_multilingual:
            assert len_cond is not None and text_len is not None, "multilingual prefill needs len_cond/text_len"
            analyzer = AlignmentStreamAnalyzer(
                self.tfmr,
                None,
                text_tokens_slice=(len_cond, len_cond + text_len),
                alignment_layer_idx=9,  # upstream hard-codes this too
                eos_idx=self.hp.stop_speech_token,
            )
            assert analyzer.eos_idx == self.hp.stop_speech_token

        return T3HuggingfaceBackend(
            config=self.cfg,
            llama=self.tfmr,
            speech_enc=self.speech_emb,
            speech_head=self.speech_head,
            alignment_stream_analyzer=analyzer,
        )

    def get_ar_backend(
        self, *, len_cond: int | None = None, text_len: int | None = None
    ) -> T3HuggingfaceBackend:
        """Return the backend, rebuilding it when ``self.compiled`` is cleared.

        Upstream clears ``self.compiled`` at the top of every ``inference`` call
        (``t3.py:273``), which gives multilingual requests a fresh analyzer. This
        keeps that behaviour and lets 9-3 memoise by overriding ``build_ar_backend``.
        """
        if not self.compiled:
            self.patched_model = self.build_ar_backend(len_cond=len_cond, text_len=text_len)
            self.compiled = True
        return self.patched_model

    # ------------------------------------------------------------------
    # Prefill / step / sample (the Phase 9-3 seam)
    # ------------------------------------------------------------------
    def ar_forward(self, inputs_embeds: Tensor, past_key_values: object) -> object:
        """One backbone forward with the attention request derived from the backend."""
        backend = self.get_ar_backend()
        return backend(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=self.ar_output_attentions(backend),
            output_hidden_states=True,
            return_dict=True,
        )

    def ar_cfg_expand(self, embed: Tensor) -> Tensor:
        """Duplicate a last-position embedding for the CFG conditional/unconditional rows."""
        return torch.cat([embed, embed])

    def ar_token_embedding(self, token: Tensor, position: int) -> Tensor:
        """Embed a sampled token at ``position`` (speech embedding + learned position)."""
        embed = self.speech_emb(token)
        return embed + self.speech_pos_emb.get_fixed_embedding(position)

    def ar_prefill(
        self,
        *,
        t3_cond,
        text_tokens: Tensor,
        initial_speech_tokens: Tensor,
        cfg_weight: float = 0.0,
    ) -> T3ARDecodeState:
        """Compose conditioning + text + BOS and run the uncached prefill forward."""
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # Build the backend right after the embeds are known (the multilingual
        # analyzer needs `len_cond`/text length); mirrors upstream's ordering.
        self.get_ar_backend(len_cond=len_cond, text_len=text_tokens.size(-1))

        device = embeds.device
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # (1, 1, dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = self.ar_cfg_expand(bos_embed)
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        output = self.ar_forward(inputs_embeds, None)

        return T3ARDecodeState(
            generated_ids=bos_token.clone(),
            logits=output.logits[:, -1, :],
            past_key_values=output.past_key_values,
            len_cond=len_cond,
        )

    def ar_step(self, state: T3ARDecodeState, next_token: Tensor, position: int) -> tuple[Tensor, object]:
        """Feed one sampled token through the backbone and return its logits + new cache."""
        embed = self.ar_token_embedding(next_token, position)
        embed = self.ar_cfg_expand(embed)
        output = self.ar_forward(embed, state.past_key_values)
        return output.logits[:, -1, :], output.past_key_values

    def ar_sample(
        self,
        logits_step: Tensor,
        generated_ids: Tensor,
        *,
        sampler: T3ARSampler,
        backend: T3HuggingfaceBackend | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Turn the last position's logits into one sampled token (upstream ``t3.py:353-382``)."""
        if backend is None:
            backend = self.get_ar_backend()
        logits = sampler.combine_cfg(logits_step)

        analyzer = getattr(backend, "alignment_stream_analyzer", None)
        if analyzer is not None:
            if logits.dim() == 1:  # guard in case something upstream squeezed
                logits = logits.unsqueeze(0)  # (1, V)
            # Pass the last generated token for repetition tracking.
            last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
            logits = analyzer.step(logits, next_token=last_token)  # (1, V)

        return sampler.pick(logits, generated_ids, generator=generator)

    # ------------------------------------------------------------------
    # Upstream-compatible entry point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond,
        text_tokens: Tensor,
        initial_speech_tokens: Tensor | None = None,
        # misc conditioning
        prepend_prompt_speech_tokens: Tensor | None = None,
        # HF generate args (kept for signature compatibility; upstream ignores
        # `num_return_sequences`, `do_sample` and `length_penalty` here too)
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0.5,
    ):
        """Generate speech tokens with the cleaned loop (see module docstring)."""
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_bot_eot(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token.
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Upstream rebuilds the backend on every call (fresh analyzer state).
        self.compiled = False

        sampler = T3ARSampler(
            cfg_weight=cfg_weight,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
        state = self.ar_prefill(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            initial_speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        predicted = []
        for i in self.ar_progress(range(max_new_tokens)):
            next_token = self.ar_sample(state.logits, state.generated_ids, sampler=sampler)
            predicted.append(next_token)
            state.generated_ids = torch.cat([state.generated_ids, next_token], dim=1)

            # Upstream always stops on EOS (`stop_on_eos` is accepted but unused
            # there); preserved verbatim so outputs stay identical.
            if next_token.view(-1) == self.hp.stop_speech_token:
                logger.info(f"✅ EOS token detected! Stopping generation at step {i + 1}")
                break

            state.logits, state.past_key_values = self.ar_step(state, next_token, position=i + 1)

        return torch.cat(predicted, dim=1)  # (1, num_tokens)


__all__ = [
    "ChatterboxT3AR",
    "T3ARDecodeState",
    "T3ARSampler",
    "PROGRESS_ENV",
]
