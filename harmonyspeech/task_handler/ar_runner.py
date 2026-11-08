"""Token-level autoregressive batch runner for Chatterbox T3 (Phase 9-3).

This module is the "mini-vLLM" core: it drives ``t3.tfmr`` (an HF ``LlamaModel``
for English Chatterbox, a ``GPT2Model`` for Turbo) **directly** against a 9-2
``ARCacheArena``, decodes one speech token per row per engine step, and drops
rows on EOS. Upstream's ``T3.inference`` / ``T3.inference_turbo`` stay untouched
and remain the ``len == 1`` / fallback path.

Why a runner instead of editing ``inference``
---------------------------------------------
The upstream loops own a per-request HF ``DynamicCache`` and one forward per
token. Batching across requests needs one forward over N rows and a cache whose
storage is the shared pool, which the upstream loops cannot express. The 9-1 seam
(``ChatterboxT3AR``) already decomposed the English loop; this runner reuses its
sampler/warpers and modules but supplies its own batched forward and per-row
sampling.

The two decode contracts (verified 2026-11-07, see the task file)
-----------------------------------------------------------------
``EnglishContract``
    two CFG rows per request (cond / uncond, row 1's text embedding zeroed when
    ``cfg_weight > 0``), prefill = cond + text + start-speech + BOS, learned
    speech position ``k`` on token ``k``, sampler = CFG combine → repetition
    penalty → temperature → min_p → top_p → multinomial, EOS stops immediately
    (the EOS token stays in the stream; the handler runs upstream's
    ``drop_invalid_tokens`` filter). Up to 1000 sampled tokens.
``TurboContract``
    one row per request, prefill = cond + text + start-speech (no BOS, no learned
    positions), sampler = ``LogitsProcessorList`` (temperature → top_k → top_p →
    repetition penalty) over the generated ids, EOS is checked **after** the
    first token and the trailing EOS is stripped, up to 1001 sampled tokens.
    ``tfmr.wte`` is deleted by the Turbo loader, which is safe because the runner
    always passes ``inputs_embeds`` (positions come from ``wpe(position_ids)``).

Reproducibility
---------------
Every row owns a ``torch.Generator`` seeded from the request seed. A generator
seeded exactly like the global RNG produces the same draw as the sequential
loop, so B=1 is bit-identical to 9-1's cleaned loop and a row's tokens are
identical whether it decodes alone or in a batch (the equality gates). RNG is
consumed **per row**, never as one batched draw, which is what makes that
statement well-defined.

KV-cache wiring
---------------
:class:`ArenaBatchCache` is an HF ``Cache`` over one forward's row set: each
``update`` writes the new K/V into the row's ``ARCacheSlot`` (write-before-read)
and returns a per-layer ``[R, H, len, D]`` view. When the batch's slots are a
contiguous ascending run the view is zero-copy into the arena; otherwise it is
gathered (still correct, and only the active length is copied). Stale bytes past
a row's current length are never read: the forward gets a per-row
``attention_mask`` whenever rows have different lengths.

Per-row isolation (Phase 8-3 Deviation 6 discipline)
----------------------------------------------------
Sampling is per row, so a bad row fails alone. A batched forward that raises is
retried row-by-row to isolate the poisoned row; its neighbours keep their logits
and continue. A refused pool grant (``acquire`` → ``None``) makes ``begin``
return ``None`` and the handler takes the sequential fallback for that request
only.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from loguru import logger
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from harmonyspeech.common.device import devices_match, resolve_device
from harmonyspeech.engine.ar_cache import ARCacheBudgetError, ARCacheDescriptor, ARCachePool, ARCacheSlot
from harmonyspeech.modeling.models.chatterbox.t3_ar import T3ARSampler

__all__ = [
    "ARBatchRunner",
    "ArenaBatchCache",
    "ARRow",
    "ARStepOutcome",
    "ARPrefillTooLongError",
    "EnglishContract",
    "TurboContract",
    "cast_conditionals",
    "english_text_tokens",
    "turbo_text_tokens",
]

#: Upstream English ``tts.py`` hard-codes ``max_new_tokens=1000``.
ENGLISH_MAX_NEW_TOKENS = 1000

#: Turbo samples one token before its loop, then loops ``max_gen_len=1000``.
TURBO_MAX_NEW_TOKENS = 1001


class ARPrefillTooLongError(ValueError):
    """A request's prefill does not fit one arena slot.

    Raised before any KV write. ``ARBatchRunner.begin`` turns it into a graceful
    ``None`` (the handler's sequential fallback), because the sequential loop has
    no per-slot cap and can still serve an unusually long text.
    """


# ---------------------------------------------------------------------------
# Tokenisation helpers (the wrapper's pre-decode text step)
# ---------------------------------------------------------------------------
def english_text_tokens(model, text: str) -> Tensor:
    """BOT + EnTokenizer(text) + EOT, as ``ChatterboxTTS.generate`` builds it.

    Returns ``[1, T]``. The CFG duplication happens in
    :meth:`EnglishContract.build_prefill`, not here (one text row in, two CFG
    rows out).
    """
    from chatterbox.tts import punc_norm

    tokens = model.tokenizer.text_to_tokens(punc_norm(text)).to(model.device)
    tokens = torch.atleast_2d(tokens).to(dtype=torch.long)
    start = int(model.t3.hp.start_text_token)
    stop = int(model.t3.hp.stop_text_token)
    tokens = torch.nn.functional.pad(tokens, (1, 0), value=start)
    tokens = torch.nn.functional.pad(tokens, (0, 1), value=stop)
    return tokens.to(device=model.t3.device)


def turbo_text_tokens(model, text: str) -> Tensor:
    """``AutoTokenizer`` ids for Turbo (no BOT/EOT, as ``tts_turbo.py`` does)."""
    from chatterbox.tts_turbo import punc_norm

    encoded = model.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True)
    return encoded.input_ids.to(device=model.t3.device)


def cast_conditionals(conds, t3):
    """Cast a request's ``T3Cond`` to the T3 weight dtype/device, in place.

    The AR path never assigns conditionals to ``model.conds``; this is the local
    equivalent of :meth:`ModelRunnerBase._align_chatterbox_conditionals` (Phase
    4-1 dtype alignment) applied to the request-owned object.
    """
    t3_cond = getattr(conds, "t3", None)
    if t3_cond is None:
        return conds
    dtype = next(t3.parameters()).dtype
    conds.t3 = t3_cond.to(device=t3.device, dtype=dtype)
    return conds


# ---------------------------------------------------------------------------
# Arena-backed HF cache
# ---------------------------------------------------------------------------
class ArenaBatchCache(Cache):
    """One forward's K/V view over a set of 9-2 arena slots.

    ``write_positions[r]`` is the KV index of the first new token for row ``r``
    and ``write_lengths[r]`` how many contiguous tokens it writes (the prefill
    length, or 1 for a decode step). ``length`` is the maximum sequence length
    the attention may see; shorter rows are masked by the caller.
    """

    def __init__(
        self,
        arena,
        slots: list[ARCacheSlot],
        *,
        write_positions: list[int],
        write_lengths: list[int],
        length: int,
    ) -> None:
        # `layers=[]` keeps the base Cache bookkeeping valid (`is_sliding` empty,
        # `is_compileable` False); `update` is fully overridden.
        super().__init__(layers=[])
        self.arena = arena
        self.slots = list(slots)
        self.write_positions = [int(value) for value in write_positions]
        self.write_lengths = [int(value) for value in write_lengths]
        self.length = int(length)
        indices = [slot.slot_index for slot in self.slots]
        self._contiguous = indices == list(range(indices[0], indices[0] + len(indices))) if indices else False
        self._slot_index = torch.tensor(indices, device=arena.tensor.device) if indices else None

    # -- HF Cache interface ---------------------------------------------
    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int, cache_kwargs=None):
        rows = key_states.shape[0]
        for row in range(rows):
            tokens = self.write_lengths[row]
            if tokens <= 0:
                continue
            keys = key_states[row, :, :tokens, :].transpose(0, 1)  # [T, H, D]
            values = value_states[row, :, :tokens, :].transpose(0, 1)
            self.slots[row].write(layer_idx, self.write_positions[row], keys, values)
        return self._layer_kv(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def get_mask_sizes(self, cache_position, layer_idx: int) -> tuple[int, int]:
        return self.length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.arena.cap_tokens

    # -- views -----------------------------------------------------------
    def _layer_kv(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        rows = len(self.slots)
        if self._contiguous:
            base = self.slots[0].slot_index
            keys = self.arena.tensor[layer_idx, 0, base : base + rows, : self.length]
            values = self.arena.tensor[layer_idx, 1, base : base + rows, : self.length]
        else:
            keys = self.arena.tensor[layer_idx, 0].index_select(0, self._slot_index)[:, : self.length]
            values = self.arena.tensor[layer_idx, 1].index_select(0, self._slot_index)[:, : self.length]
        # [R, len, H, D] -> [R, H, len, D] (the layout every attention path wants)
        return keys.transpose(1, 2), values.transpose(1, 2)


# ---------------------------------------------------------------------------
# Row state
# ---------------------------------------------------------------------------
@dataclass
class ARRow:
    """One request's decode state, owned by the runner (never by the model)."""

    request_id: str
    contract: object
    slots: list[ARCacheSlot]
    conds: object
    text_tokens: Tensor
    generator: torch.Generator
    seed: int | None = None
    sampler: T3ARSampler | None = None
    processors: LogitsProcessorList | None = None
    start_token: Tensor | None = None
    generated_ids: Tensor | None = None
    tokens: list[Tensor] = field(default_factory=list)
    logits: Tensor | None = None
    prefill_len: int = 0
    length: int = 0
    position: int = 0
    max_new_tokens: int = ENGLISH_MAX_NEW_TOKENS
    steps: int = 0
    finished: bool = False
    finish_reason: str | None = None
    error: Exception | None = None

    @property
    def rows_per_request(self) -> int:
        return len(self.slots)

    @property
    def tokens_emitted(self) -> int:
        return len(self.tokens)

    def progress(self) -> dict:
        return {"tokens": self.tokens_emitted, "step": self.steps}

    def release_slots(self) -> None:
        for slot in self.slots:
            if not slot.released:
                slot.release()


@dataclass
class ARStepOutcome:
    """Per-request result of one decode step."""

    request_id: str
    finished: bool = False
    progress: dict = field(default_factory=dict)
    error: Exception | None = None


# ---------------------------------------------------------------------------
# Decode contracts
# ---------------------------------------------------------------------------
class _BaseContract:
    rows_per_request = 1
    max_new_tokens = ENGLISH_MAX_NEW_TOKENS

    def build_prefill(self, t3, conds_t3, text_tokens: Tensor, cfg_weight: float) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def embed_token(self, t3, row: ARRow) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def sample(self, t3, row: ARRow) -> Tensor | None:  # pragma: no cover
        raise NotImplementedError

    def stop_after_sample(self, t3, row: ARRow, token: Tensor) -> bool:  # pragma: no cover
        raise NotImplementedError

    def final_tokens(self, t3, row: ARRow) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class EnglishContract(_BaseContract):
    """Upstream ``T3.inference`` semantics (two CFG rows, immediate EOS)."""

    rows_per_request = 2
    max_new_tokens = ENGLISH_MAX_NEW_TOKENS

    def build_prefill(self, t3, conds_t3, text_tokens: Tensor, cfg_weight: float) -> Tensor:
        # `prepare_input_embeds` is upstream's own composition (cond + text +
        # initial speech token, row 1's text zeroed when cfg > 0); the extra BOS
        # is appended exactly as `inference` does.
        text = torch.cat([text_tokens, text_tokens], dim=0)
        start = int(t3.hp.start_speech_token)
        initial = start * torch.ones_like(text[:, :1])
        embeds, _len_cond = t3.prepare_input_embeds(
            t3_cond=conds_t3,
            text_tokens=text,
            speech_tokens=initial,
            cfg_weight=cfg_weight,
        )
        bos = torch.full((1, 1), start, dtype=torch.long, device=embeds.device)
        bos = t3.speech_emb(bos)
        bos = bos + t3.speech_pos_emb.get_fixed_embedding(0)
        bos = torch.cat([bos, bos])
        return torch.cat([embeds, bos], dim=1)

    def embed_token(self, t3, row: ARRow) -> Tensor:
        embed = t3.speech_emb(row.tokens[-1])
        embed = embed + t3.speech_pos_emb.get_fixed_embedding(row.position)
        return torch.cat([embed, embed])

    def sample(self, t3, row: ARRow) -> Tensor | None:
        logits = row.sampler.combine_cfg(row.logits)
        token = row.sampler.pick(logits, row.generated_ids, generator=row.generator)
        row.generated_ids = torch.cat([row.generated_ids, token], dim=1)
        row.tokens.append(token)
        return token

    def stop_after_sample(self, t3, row: ARRow, token: Tensor) -> bool:
        return int(token.view(-1)[0]) == int(t3.hp.stop_speech_token)

    def final_tokens(self, t3, row: ARRow) -> Tensor:
        stream = torch.cat(row.tokens, dim=1)[0]  # 1-D, includes a sampled EOS
        stream = drop_invalid_tokens(stream)
        return stream[stream < int(t3.hp.start_speech_token)].to(dtype=torch.long)


class TurboContract(_BaseContract):
    """Upstream ``T3.inference_turbo`` semantics (one row, logits processors)."""

    rows_per_request = 1
    max_new_tokens = TURBO_MAX_NEW_TOKENS

    def build_prefill(self, t3, conds_t3, text_tokens: Tensor, cfg_weight: float) -> Tensor:
        start = int(t3.hp.start_speech_token)
        initial = start * torch.ones_like(text_tokens[:, :1])
        embeds, _len_cond = t3.prepare_input_embeds(
            t3_cond=conds_t3,
            text_tokens=text_tokens,
            speech_tokens=initial,
            cfg_weight=0.0,
        )
        return embeds

    def embed_token(self, t3, row: ARRow) -> Tensor:
        # Turbo has no learned position embedding (`input_pos_emb=None`).
        return t3.speech_emb(row.tokens[-1])

    def sample(self, t3, row: ARRow) -> Tensor | None:
        input_ids = torch.cat(row.tokens, dim=1) if row.tokens else row.start_token
        processed = row.processors(input_ids, row.logits)
        if bool(torch.all(processed == -float("inf"))):
            return None
        probs = torch.softmax(processed, dim=-1)
        token = torch.multinomial(probs, num_samples=1, generator=row.generator)
        row.tokens.append(token)
        return token

    def stop_after_sample(self, t3, row: ARRow, token: Tensor) -> bool:
        # Upstream checks EOS only for draws inside its loop (token 2 onwards).
        return len(row.tokens) >= 2 and int(token.view(-1)[0]) == int(t3.hp.stop_speech_token)

    def final_tokens(self, t3, row: ARRow) -> Tensor:
        stream = torch.cat(row.tokens, dim=1)[0]
        stop = int(t3.hp.stop_speech_token)
        if stream.numel() and int(stream[-1]) == stop:
            stream = stream[:-1]
        return stream[stream < int(t3.hp.start_speech_token)].to(dtype=torch.long)


def _build_processors(temperature, top_k, top_p, repetition_penalty) -> LogitsProcessorList:
    """Turbo's processor list, in upstream's order (t3.py:418-426)."""
    processors = LogitsProcessorList()
    if temperature and temperature > 0 and temperature != 1.0:
        processors.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k > 0:
        processors.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p))
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    return processors


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
class ARBatchRunner:
    """Batched T3 decode over a 9-2 arena; one token per row per step.

    The runner owns every piece of per-request decode state (row, slot, sampler,
    generator, token stream). It never reads or writes ``model.conds`` — the
    conditionals are projected into the prefill, so the Phase 0-2
    ``finally``-restore is superseded on this path (the sequential fallback keeps
    it).
    """

    def __init__(
        self,
        *,
        t3,
        pool: ARCachePool | None,
        model_key: str,
        model_config=None,
        contract: _BaseContract | None = None,
        model_type: str | None = None,
    ) -> None:
        self.t3 = t3
        self.pool = pool
        self.model_key = model_key
        self.model_config = model_config
        # Canonical indexed device (Phase 11-0): ``t3.device`` is already
        # indexed in practice, but resolving keeps a mocked/CPU t3 and an
        # explicit ``cuda:1`` consistent with the pool's accounting.
        self.device = resolve_device(getattr(t3, "device", None))
        model_type = model_type or getattr(model_config, "model_type", None)
        self.contract = contract or (TurboContract() if model_type == "ChatterboxTurboTTS" else EnglishContract())
        self.arena = None
        self.rows: dict[str, ARRow] = {}
        self._lock = threading.RLock()
        self.default_seed = int(getattr(model_config, "seed", 0) or 0)
        self._seed_counter = 0
        self._registration_attempted = False
        self._registration_error: str | None = None

    # -- registration ----------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.arena is not None

    def _geometry(self) -> tuple[int, int, int, torch.dtype]:
        cfg = self.t3.cfg
        if getattr(self.t3, "is_gpt", False):
            layers = int(cfg.n_layer)
            heads = int(cfg.n_head)
            head_dim = int(cfg.n_embd) // heads
        else:
            layers = int(cfg.num_hidden_layers)
            heads = int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
            head_dim = int(getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads))
        dtype = next(self.t3.parameters()).dtype
        return layers, heads, head_dim, dtype

    def register(self) -> bool:
        """Register this family's arena under the scheduler's model key.

        ``model_key`` must be the scheduler's name (``ModelConfig.name`` /
        ``request_data.model``): 9-4's capacity admission reads the pool by that
        key and treats any other key as "no signal". A disabled pool, a
        non-CUDA pool for a CUDA model, or a budget refusal disables the runner
        (sequential fallback) instead of failing boot.

        Phase 11-0: the arena is allocated on **this model's resolved device**
        (``cuda`` and ``cuda:0`` are the same device; an explicit ``cuda:1`` is
        honoured), and the budget check uses that device's remaining headroom.
        The old code compared ``pool.device`` with ``self.device`` directly and
        disabled the runner for every CUDA model because ``torch.device("cuda")
        != torch.device("cuda:0")``.
        """
        with self._lock:
            if self.arena is not None:
                return True
            if self._registration_attempted:
                return False
            self._registration_attempted = True
            pool = self.pool
            if pool is None or not getattr(pool, "enabled", False):
                self._registration_error = "no AR cache pool"
                return False
            model_device = resolve_device(self.device)
            if model_device.type == "cuda" and not torch.cuda.is_available():
                self._registration_error = f"model device {model_device} but CUDA is unavailable"
                logger.warning(
                    f"AR runner for '{self.model_key}' disabled: {self._registration_error}; "
                    "Chatterbox requests use the sequential loop"
                )
                return False
            if not devices_match(pool.device, model_device):
                # Not a refusal (Phase 11-0): the pool allocates per device, so a
                # model on a secondary device still gets a real arena there. Log
                # it because the placement is worth seeing in the boot table.
                logger.info(
                    f"AR runner for '{self.model_key}': arena on {model_device} "
                    f"(pool default {pool.device})"
                )
            try:
                layers, kv_heads, head_dim, dtype = self._geometry()
                # Bound the arena to the maximum rows the scheduler can admit
                # (`max_batch_size` requests × rows per request), clamped to what
                # the remaining budget fits. This keeps one family from eating
                # the whole shared pool and makes a runtime refusal impossible
                # when the budget is adequate; an arena that still cannot fit a
                # slab is a graceful sequential fallback, never a boot failure.
                cap = int(getattr(pool, "default_cap_tokens", 768) or 768)
                slab = ARCacheDescriptor(
                    model_key=self.model_key,
                    layers=layers,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    cap_tokens=cap,
                ).slab_bytes(cap)
                desired = max(
                    1,
                    int(getattr(self.model_config, "max_batch_size", 1) or 1)
                    * self.contract.rows_per_request,
                )
                remaining = (
                    pool.remaining_bytes_for(model_device)
                    if hasattr(pool, "remaining_bytes_for")
                    else pool.remaining_bytes
                )
                fit = max(1, int(remaining) // slab)
                slots = max(1, min(desired, fit))
                arena = pool.register(
                    ARCacheDescriptor(
                        model_key=self.model_key,
                        layers=layers,
                        kv_heads=kv_heads,
                        head_dim=head_dim,
                        dtype=dtype,
                        cap_tokens=None,  # pool default (`ar_inference.default_cap_tokens`)
                        slots=slots,
                    ),
                    device=model_device,
                )
            except ARCacheBudgetError as error:
                self._registration_error = str(error)
                logger.warning(f"AR runner for '{self.model_key}' disabled: {error}")
                return False
            except Exception as error:  # noqa: BLE001 - never fail boot for an optimization
                self._registration_error = str(error)
                logger.warning(f"AR runner registration failed for '{self.model_key}': {error}")
                return False
            if arena is None:
                self._registration_error = "disabled pool"
                return False
            self.arena = arena
            logger.info(
                f"AR decode arena '{self.model_key}' on {arena.device}: {arena.slots} slot(s) × "
                f"{arena.cap_tokens} tokens, {arena.slab_bytes / 2**20:.1f} MiB/slot, dtype={dtype}"
            )
            return True

    # -- row lifecycle ---------------------------------------------------
    def has_row(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self.rows

    def active_rows(self) -> int:
        with self._lock:
            return len(self.rows)

    def _next_seed(self) -> int:
        self._seed_counter += 1
        return (self.default_seed * 1000003 + self._seed_counter) % (2**31 - 1)

    def _resolved_seed(self, seed: int | None) -> int:
        """The seed a row actually decodes with (``_next_seed`` when unset).

        Stored on the row so the Phase 9-5 S3Gen hop can re-create the same
        deterministic noise stream after the AR decode has consumed the row's
        generator.
        """
        return self._next_seed() if seed is None else int(seed)

    def _generator(self, seed: int | None) -> torch.Generator:
        device = "cpu" if torch.device(self.device).type == "cpu" else str(self.device)
        generator = torch.Generator(device=device)
        generator.manual_seed(self._resolved_seed(seed))
        return generator

    def begin(
        self,
        request_id: str,
        *,
        conds,
        text_tokens: Tensor,
        cfg_weight: float = 0.5,
        seed: int | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        top_k: int = 1000,
        max_new_tokens: int | None = None,
    ) -> ARRow | None:
        """Admit a request: acquire slots, prefill, sample its first token later.

        Returns ``None`` when the pool refuses a slot (or the runner is
        disabled), which is the handler's cue to run the sequential loop for
        that request. An exception (bad prefill) propagates so the handler can
        error just that request.
        """
        with self._lock:
            existing = self.rows.get(request_id)
            if existing is not None:
                return existing
            if self.arena is None and not self.register():
                return None

            slots: list[ARCacheSlot] = []
            try:
                for _ in range(self.contract.rows_per_request):
                    slot = self.pool.acquire(self.model_key)
                    if slot is None:
                        for granted in slots:
                            if not granted.released:
                                granted.release()
                        return None
                    slots.append(slot)
            except KeyError:
                for granted in slots:
                    if not granted.released:
                        granted.release()
                return None

            resolved_seed = self._resolved_seed(seed)
            row = ARRow(
                request_id=request_id,
                contract=self.contract,
                slots=slots,
                conds=conds,
                text_tokens=text_tokens,
                generator=self._generator(resolved_seed),
                seed=resolved_seed,
            )
            if isinstance(self.contract, TurboContract):
                row.processors = _build_processors(temperature, top_k, top_p, repetition_penalty)
                row.start_token = torch.full(
                    (1, 1),
                    int(self.t3.hp.start_speech_token),
                    dtype=torch.long,
                    device=self.t3.device,
                )
            else:
                row.sampler = T3ARSampler(
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                )
                row.generated_ids = torch.full(
                    (1, 1),
                    int(self.t3.hp.start_speech_token),
                    dtype=torch.long,
                    device=self.t3.device,
                )

            try:
                self._prefill(row, conds)
            except ARPrefillTooLongError as error:
                for granted in slots:
                    if not granted.released:
                        granted.release()
                logger.debug(f"AR prefill for {request_id} does not fit a slot ({error}); sequential fallback")
                return None
            except Exception:
                for granted in slots:
                    if not granted.released:
                        granted.release()
                raise
            if max_new_tokens is not None:
                row.max_new_tokens = max(0, min(row.max_new_tokens, int(max_new_tokens)))
            self.rows[request_id] = row
            return row

    def _prefill(self, row: ARRow, conds) -> None:
        embeds = self.contract.build_prefill(
            self.t3,
            getattr(conds, "t3", conds),
            row.text_tokens,
            float(getattr(row.sampler, "cfg_weight", 0.0)) if row.sampler is not None else 0.0,
        )
        length = int(embeds.shape[1])
        if length > self.arena.cap_tokens:
            raise ARPrefillTooLongError(
                f"prefill of {length} tokens exceeds the arena's cap_tokens={self.arena.cap_tokens} "
                f"for '{self.model_key}'"
            )
        rows = len(row.slots)
        cache = ArenaBatchCache(
            self.arena,
            row.slots,
            write_positions=[0] * rows,
            write_lengths=[length] * rows,
            length=length,
        )
        cache_position = torch.arange(length, dtype=torch.long, device=self.t3.device)
        position_ids = cache_position.unsqueeze(0).expand(rows, -1)
        output = self._forward(
            embeds,
            cache,
            cache_position=cache_position,
            attention_mask=None,
            position_ids=position_ids,
        )
        logits = self.t3.speech_head(output.last_hidden_state)  # [R, length, V]
        row.logits = logits[:, -1, :]
        row.prefill_len = length
        row.length = length
        row.position = 1
        row.max_new_tokens = max(0, min(self.contract.max_new_tokens, self.arena.cap_tokens - length))

    def _forward(self, inputs_embeds, cache, *, cache_position, attention_mask, position_ids):
        return self.t3.tfmr(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

    def _advance(self, rows: list[ARRow]) -> None:
        """One batched forward for the rows that still need their token cached."""
        try:
            self._advance_batch(rows)
        except Exception as error:  # noqa: BLE001 - isolate the poisoned row
            logger.warning(f"AR batched decode failed ({error}); retrying row-by-row to isolate it")
            for row in rows:
                try:
                    self._advance_batch([row])
                except Exception as row_error:  # noqa: BLE001 - errors this row only
                    row.error = row_error

    def _advance_batch(self, rows: list[ARRow]) -> None:
        rows_flat = sum((row.slots for row in rows), [])
        embeds = torch.cat([row.contract.embed_token(self.t3, row) for row in rows], dim=0)
        length_after = max(row.length for row in rows) + 1
        write_positions = [row.length for row in rows for _ in row.slots]
        common_length = all(row.length == rows[0].length for row in rows)
        if common_length:
            attention_mask = None
        else:
            attention_mask = torch.zeros(
                len(rows_flat), length_after, dtype=torch.bool, device=self.t3.device
            )
            index = 0
            for row in rows:
                for _ in row.slots:
                    attention_mask[index, : row.length + 1] = True
                    index += 1
        cache = ArenaBatchCache(
            self.arena,
            rows_flat,
            write_positions=write_positions,
            write_lengths=[1] * len(rows_flat),
            length=length_after,
        )
        cache_position = torch.tensor([length_after - 1], dtype=torch.long, device=self.t3.device)
        position_ids = torch.tensor(
            [[row.length] for row in rows for _ in row.slots], dtype=torch.long, device=self.t3.device
        )
        output = self._forward(
            embeds,
            cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.t3.speech_head(output.last_hidden_state)  # [R, 1, V]
        index = 0
        for row in rows:
            width = len(row.slots)
            row.logits = logits[index : index + width, -1, :]
            index += width
            row.length += 1
            row.position += 1

    # -- decode ----------------------------------------------------------
    def step(self, request_ids) -> list[ARStepOutcome]:
        """Advance every named row by one speech token (one batched forward)."""
        outcomes: list[ARStepOutcome] = []
        with self._lock:
            rows = [self.rows[request_id] for request_id in request_ids if request_id in self.rows]
            advance: list[ARRow] = []
            for row in rows:
                if row.finished or row.error is not None:
                    # A finished/errored row is never sampled again (Phase 11-0).
                    # The engine pops finished rows per step, but the seeded
                    # parity harness passes its whole row set every step; without
                    # this guard a row that already hit EOS would draw an extra
                    # token and its stream would no longer match the solo run.
                    continue
                row.steps += 1
                try:
                    token = row.contract.sample(self.t3, row)
                except Exception as error:  # noqa: BLE001 - this row alone
                    row.error = error
                    continue
                if token is None or row.contract.stop_after_sample(self.t3, row, token):
                    row.finished = True
                    row.finish_reason = "stop"
                elif row.tokens_emitted >= row.max_new_tokens or row.length >= self.arena.cap_tokens:
                    # Upstream stops after `max_new_tokens` draws; the arena cap
                    # bounds the KV write, so a longer request truncates here.
                    row.finished = True
                    row.finish_reason = "length"
                else:
                    advance.append(row)
            if advance:
                self._advance(advance)
            for row in rows:
                outcomes.append(
                    ARStepOutcome(
                        request_id=row.request_id,
                        finished=row.finished and row.error is None,
                        progress=row.progress(),
                        error=row.error,
                    )
                )
        return outcomes

    # -- finish / release ------------------------------------------------
    def final_tokens(self, request_id: str) -> Tensor | None:
        """Return the S3Gen-ready token stream for a finished row."""
        with self._lock:
            row = self.rows.get(request_id)
            if row is None:
                return None
            return row.contract.final_tokens(self.t3, row)

    def row(self, request_id: str) -> ARRow | None:
        with self._lock:
            return self.rows.get(request_id)

    def raw_tokens(self, request_id: str) -> Tensor | None:
        """The sampled token stream **before** the contract's EOS/vocab filtering."""
        with self._lock:
            row = self.rows.get(request_id)
            if row is None or not row.tokens:
                return None
            return torch.cat(row.tokens, dim=1)

    def pop(self, request_id: str) -> ARRow | None:
        """Remove a row and release its slots (finish, error, abort, deadline)."""
        with self._lock:
            row = self.rows.pop(request_id, None)
        if row is not None:
            row.release_slots()
        return row

    def release_request(self, request_id: str) -> None:
        """Duck-typed executor hook (9-4): drop a row that will never run again."""
        self.pop(request_id)

    def release_all(self) -> None:
        with self._lock:
            rows = list(self.rows.values())
            self.rows.clear()
        for row in rows:
            row.release_slots()
