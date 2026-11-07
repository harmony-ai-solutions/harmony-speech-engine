"""Cross-request Whisper window batching (Phase 10-2).

``BatchedInferencePipeline`` (faster-whisper 1.2.1) batches the ≤30 s clips of
**one** audio. This module owns the missing cross-request step: collect the
windows of every admitted Whisper request in a step and run them through **one**
CTranslate2 ``encode(batch)`` + **one** ``generate(encoder_output, prompts, …)``
call per formed window batch. CT2 keeps the KV cache and native beam search
inside the call — the engine only forms batches (study #14 D7/D10).

Design notes
------------
* **Reuse, not re-implementation.** Window formation reuses faster-whisper's
  own VAD, ``collect_chunks``, ``pad_or_trim`` and feature extractor; timestamp
  restoration reuses ``_split_segments_by_timestamps`` and
  ``restore_speech_timestamps``. The thin new code is the *cross-request* row
  table and the per-request stitching around it.
* **One encoder pass.** Language detection calls CT2's ``detect_language`` on
  the encoder output that ``generate`` is about to use, instead of the second
  encoder pass ``WhisperModel.detect_language`` performs. The one honest parity
  edge (first VAD window shorter than 30 s on a multi-window request) is
  documented in ``.current_work/engine-modernization/10-2-WhisperWindowBatcher.md``.
* **Per-request semantics.** Each request gets its own tokenizer (detected
  language) and prompt; a window batch may mix languages because CT2 takes a
  list of prompts, one per row.
* **Isolation.** A request whose collection or decoding fails errors alone.
  A batch-level CT2 failure propagates so the handler can fall back to the
  sequential ``pipeline.transcribe`` path for the whole step (the Phase 8-3
  deviation-6 discipline: never de-batch neighbours silently).
* **Bounded memory.** At most :data:`WHISPER_WINDOW_BATCH_SIZE` windows per
  formed batch (owner-approved initial 8, 2026-11-07); batches are formed and
  consumed one at a time, so peak activations are bounded by the cap, not by
  the step's queue depth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import (
    Segment,
    TranscriptionInfo,
    TranscriptionOptions,
    collect_chunks,
    get_compression_ratio,
    get_suppressed_tokens,
    pad_or_trim,
    restore_speech_timestamps,
)
from faster_whisper.vad import VadOptions, get_speech_timestamps
from loguru import logger

#: Owner-approved initial cap (2026-11-07) on windows per formed CT2 batch.
#: This is a *memory* bound, not an admission width: the scheduler admits whole
#: requests, the batcher folds their windows into batches of this size.
WHISPER_WINDOW_BATCH_SIZE = 8

#: faster-whisper's windowing scale. Kept here (and in the scheduler) so the
#: admission estimate and the collector agree on what "one window" means.
WHISPER_WINDOW_SECONDS = 30.0

#: Defaults mirrored from ``BatchedInferencePipeline.transcribe`` (see the API
#: table in the phase file). The handler passes none of these explicitly today;
#: they are named so a future request option cannot silently diverge.
_WHISPER_BEAM_SIZE = 5
_WHISPER_BEST_OF = 5
_WHISPER_PATIENCE = 1
_WHISPER_LENGTH_PENALTY = 1
_WHISPER_REPETITION_PENALTY = 1
_WHISPER_NO_REPEAT_NGRAM_SIZE = 0
_WHISPER_LOG_PROB_THRESHOLD = -1.0
_WHISPER_NO_SPEECH_THRESHOLD = 0.6
_WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
_WHISPER_TEMPERATURE = 0.0
_WHISPER_VAD_MIN_SILENCE_MS = 160
_WHISPER_PREPEND_PUNCTUATIONS = "\"'“¿([{-"
_WHISPER_APPEND_PUNCTUATIONS = "\"'.。,，!！?？:：”)]}、"


class WhisperRequestAbortedError(Exception):
    """A Whisper request was aborted by the engine before it completed.

    Lives here (Phase 10-2) because both the lane pool and the window batcher
    need to signal it; ``model_runner_base`` re-exports the name for callers and
    tests that imported it from there since Phase 10-1.
    """


def whisper_batched_inference_enabled(model_config) -> bool:
    """Resolve the Whisper cross-request batching flag.

    Whisper is intentionally **not** in ``BATCHED_INFERENCE_FAMILIES``: that set
    means "the handler forwards a padded ``[B, T_max]`` batch", while Whisper's
    win is window formation on the CT2 public API (different mechanism, same
    flag name). The cross-request path is **opt-in** for Whisper: only an
    explicit ``batched_inference: true`` enables it, so an existing deployment
    (and every pre-10-2 test) keeps today's per-request pipeline path until the
    operator turns the window batcher on. This matches the phase spec's
    "batched path when >1 request and `batched_inference: true`" and its
    "flag off → always sequential" fallback.
    """
    return bool(getattr(model_config, "batched_inference", False))


def whisper_request_options(request) -> tuple[str, str | None, str | None]:
    """Return ``(task, language_hint, prompt)`` for a Whisper request.

    Accepts an ``EngineRequest`` (the handler paths) or a bare
    ``SpeechTranscribeRequestInput`` (direct callers/tests). faster-whisper
    encodes ``task`` into the prompt through the tokenizer's
    ``<|transcribe|>`` / ``<|translate|>`` token, so both the sequential
    ``pipeline.transcribe`` call and the window batcher must use the same value.
    """
    request_data = getattr(request, "request_data", None)
    data = request_data if request_data is not None else request
    task = str(getattr(data, "task", None) or "transcribe").strip().lower()
    language = getattr(data, "language", None) or None
    prompt = getattr(data, "prompt", None) or None
    return task, language, prompt


def _request_finished(request) -> bool:
    """True when the engine already aborted/finished ``request``."""
    is_finished = getattr(request, "is_finished", None)
    if not callable(is_finished):
        return False
    try:
        return bool(is_finished())
    except Exception:  # pragma: no cover - defensive on odd request stand-ins
        return False


def _aborted(request) -> WhisperRequestAbortedError:
    return WhisperRequestAbortedError(
        f"request {getattr(request, 'request_id', '?')} was aborted before transcription"
    )


@dataclass
class _Window:
    """One ≤30 s row in a formed batch (request-tagged)."""

    request_index: int
    window_index: int
    offset: float
    duration: float
    segment_size: int
    seek: int
    features: np.ndarray


@dataclass
class _RequestWindows:
    """Everything one request contributes to the step's window table."""

    request_index: int
    request_id: str
    request: object
    audio_seconds: float
    duration_after_vad: float
    clip_timestamps: list[dict]
    vad_options: VadOptions
    #: faster-whisper task (``transcribe`` / ``translate``), resolved by the
    #: handler from ``SpeechTranscribeRequestInput.task`` (Phase 10-4).
    task: str = "transcribe"
    #: Source-language hint; when set, language detection is skipped.
    language_hint: str | None = None
    #: Optional initial prompt forwarded to ``get_prompt`` as previous tokens.
    initial_prompt: str | None = None
    windows: list[_Window] = field(default_factory=list)
    decoded: dict[int, list[dict]] = field(default_factory=dict)
    language: str | None = None
    language_probability: float | int | None = None
    all_language_probs: list | None = None
    tokenizer: object | None = None
    prompt: list[int] | None = None
    error: Exception | None = None


class WhisperWindowBatcher:
    """Collect, batch and stitch the windows of a whole Whisper step.

    ``model`` is the faster-whisper ``WhisperModel`` (not the CT2 binding); the
    public wrapper provides the feature extractor, tokenizer factory, prompt
    builder and timestamp splitter, while ``model.model`` is the CT2
    ``Whisper`` used for the batched ``encode`` / ``generate`` /
    ``detect_language`` calls.

    The batcher is **stateless across steps** (one instance per runner is fine):
    no window, encoder output or prompt survives a call, so nothing can leak
    between engine steps.
    """

    def __init__(
        self,
        model,
        window_batch_size: int = WHISPER_WINDOW_BATCH_SIZE,
    ) -> None:
        self.model = model
        self.window_batch_size = max(1, int(window_batch_size or WHISPER_WINDOW_BATCH_SIZE))
        #: Diagnostics for tests / logging; cheap integers, no per-window state.
        self.batches_formed = 0
        self.windows_batched = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def transcribe_batch(self, audio_artifacts, requests) -> list[tuple[dict | None, Exception | None]]:
        """Transcribe every request in the step; one ``(payload, error)`` per input.

        ``payload`` is the same ``{"text", "segments", "info"}`` dict the
        sequential handler serializes (segments and info already ``asdict``-ed).
        A per-request collection/decode failure is returned as ``(None, error)``
        and never affects its neighbours. A **batch-level** CT2 failure raises,
        which is the handler's signal to fall back to the sequential path for
        the whole step.
        """
        if len(audio_artifacts) != len(requests):
            raise ValueError(
                f"WhisperWindowBatcher: {len(audio_artifacts)} audio artifacts vs {len(requests)} requests"
            )

        # Diagnostics are step-scoped; nothing else survives a call.
        self.batches_formed = 0
        self.windows_batched = 0
        pairs: list[tuple[dict | None, Exception | None]] = [(None, None)] * len(requests)
        collected: list[_RequestWindows] = []

        for index, (audio_artifact, request) in enumerate(zip(audio_artifacts, requests, strict=True)):
            if _request_finished(request):
                pairs[index] = (None, _aborted(request))
                continue
            try:
                collected.append(self._collect(audio_artifact, index, request))
            except Exception as error:  # noqa: BLE001 - this request's failure only
                logger.opt(exception=error).warning(
                    f"Whisper window batcher: request {getattr(request, 'request_id', '?')} "
                    f"failed during window collection; erroring it alone"
                )
                pairs[index] = (None, error)

        # Requests with no speech produce no windows but still need a language
        # and the standard info block (the sequential path does the same).
        for request_windows in collected:
            if request_windows.windows:
                continue
            index = request_windows.request_index
            if _request_finished(request_windows.request):
                pairs[index] = (None, _aborted(request_windows.request))
                continue
            try:
                self._finish_without_windows(request_windows)
                pairs[index] = (self._payload(request_windows), None)
            except Exception as error:  # noqa: BLE001 - this request's failure only
                logger.opt(exception=error).warning(
                    f"Whisper window batcher: request {request_windows.request_id} "
                    f"failed while finishing an empty window set; erroring it alone"
                )
                pairs[index] = (None, error)

        rows = [
            (request_windows, window)
            for request_windows in collected
            if request_windows.windows
            for window in request_windows.windows
        ]
        for start in range(0, len(rows), self.window_batch_size):
            self._run_batch(rows[start : start + self.window_batch_size])

        for request_windows in collected:
            if not request_windows.windows:
                continue
            index = request_windows.request_index
            if request_windows.error is not None:
                pairs[index] = (None, request_windows.error)
                continue
            if _request_finished(request_windows.request):
                pairs[index] = (None, _aborted(request_windows.request))
                continue
            try:
                pairs[index] = (self._payload(request_windows), None)
            except Exception as error:  # noqa: BLE001 - stitching failure for this request only
                pairs[index] = (None, error)

        return pairs

    # ------------------------------------------------------------------
    # Window collection (same steps as BatchedInferencePipeline.transcribe)
    # ------------------------------------------------------------------
    def _collect(self, audio_artifact, request_index: int, request) -> _RequestWindows:
        model = self.model
        extractor = model.feature_extractor
        sampling_rate = int(extractor.sampling_rate)
        chunk_length = float(extractor.chunk_length)
        audio = audio_artifact.numpy
        duration = float(audio.shape[0]) / sampling_rate
        task, language_hint, initial_prompt = whisper_request_options(request)

        vad_options = VadOptions(
            max_speech_duration_s=chunk_length,
            min_silence_duration_ms=_WHISPER_VAD_MIN_SILENCE_MS,
        )
        clip_timestamps = get_speech_timestamps(audio, vad_options)
        audio_chunks, chunks_metadata = collect_chunks(audio, clip_timestamps, max_duration=chunk_length)
        duration_after_vad = (
            sum((clip["end"] - clip["start"]) for clip in clip_timestamps) / sampling_rate
        )

        windows: list[_Window] = []
        if duration_after_vad:
            for window_index, (chunk, metadata) in enumerate(
                zip(audio_chunks, chunks_metadata, strict=True)
            ):
                # `[..., :-1]` drops the trailing STFT column, exactly like
                # BatchedInferencePipeline.transcribe before `pad_or_trim`.
                feature = extractor(chunk)[..., :-1]
                window_duration = float(metadata["duration"])
                offset = float(metadata["offset"])
                windows.append(
                    _Window(
                        request_index=request_index,
                        window_index=window_index,
                        offset=offset,
                        duration=window_duration,
                        segment_size=int(math.ceil(window_duration) * model.frames_per_second),
                        seek=int(offset * model.frames_per_second),
                        features=pad_or_trim(feature),
                    )
                )

        return _RequestWindows(
            request_index=request_index,
            request_id=getattr(request, "request_id", str(request_index)),
            request=request,
            audio_seconds=duration,
            duration_after_vad=duration_after_vad,
            clip_timestamps=clip_timestamps,
            vad_options=vad_options,
            task=task,
            language_hint=language_hint,
            initial_prompt=initial_prompt,
            windows=windows,
        )

    # ------------------------------------------------------------------
    # Language / tokenizer / prompt (per request)
    # ------------------------------------------------------------------
    def _set_language(self, request_windows: _RequestWindows, all_results) -> None:
        if request_windows.language is not None:
            return
        if request_windows.language_hint:
            # A source-language hint skips detection (upstream parity: a
            # provided language reports probability 1 and no candidates). An
            # English-only model forces "en", exactly like WhisperModel.
            if self.model.model.is_multilingual:
                request_windows.language = request_windows.language_hint
            else:
                request_windows.language = "en"
            request_windows.language_probability = 1
            request_windows.all_language_probs = None
            return
        if not self.model.model.is_multilingual:
            request_windows.language = "en"
            # Upstream uses the int ``1`` for English-only models; keep the type
            # so ``json.dumps`` output matches the sequential payload.
            request_windows.language_probability = 1
            request_windows.all_language_probs = None
            return
        probabilities = [(token[2:-2], float(probability)) for token, probability in all_results]
        if not probabilities:
            raise RuntimeError("Whisper language detection returned no candidates")
        request_windows.all_language_probs = probabilities
        request_windows.language, request_windows.language_probability = probabilities[0]

    def _detect_languages(self, rows, encoder_output) -> None:
        """Fill the language of requests whose first window is in this batch."""
        pending: dict[int, tuple[_RequestWindows, int]] = {}
        hinted: dict[int, _RequestWindows] = {}
        for row_index, (request_windows, window) in enumerate(rows):
            if request_windows.language is not None:
                continue
            if request_windows.language_hint:
                hinted.setdefault(request_windows.request_index, request_windows)
                continue
            if window.window_index != 0:
                continue
            pending.setdefault(request_windows.request_index, (request_windows, row_index))
        for request_windows in hinted.values():
            self._set_language(request_windows, [])
        if not pending:
            return
        if not self.model.model.is_multilingual:
            for request_windows, _row in pending.values():
                self._set_language(request_windows, [])
            return
        # One CT2 call for the whole batch; CT2 accepts the encoder output that
        # `generate` is about to consume, so no second encoder pass is paid.
        results = self.model.model.detect_language(encoder_output)
        for request_windows, row_index in pending.values():
            self._set_language(request_windows, results[row_index])

    def _finish_without_windows(self, request_windows: _RequestWindows) -> None:
        """Language + tokenizer for a request whose audio produced no windows."""
        if request_windows.language_hint or not self.model.model.is_multilingual:
            self._set_language(request_windows, [])
        else:
            n_mels = getattr(self.model.model, "n_mels", None)
            if n_mels is None:
                n_mels = int(self.model.feature_extractor.mel_filters.shape[0])
            dummy = np.full((int(n_mels), 1), -1.5, dtype="float32")
            encoder_output = self.model.encode(pad_or_trim(dummy))
            results = self.model.model.detect_language(encoder_output)
            self._set_language(request_windows, results[0])
        self._tokenizer_for(request_windows)

    def _tokenizer_for(self, request_windows: _RequestWindows):
        if request_windows.tokenizer is None:
            tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                bool(self.model.model.is_multilingual),
                task=request_windows.task,
                language=request_windows.language,
            )
            request_windows.tokenizer = tokenizer
            # `condition_on_previous_text` is forced False by the upstream
            # batched pipeline, so the prompt is constant for every window.
            # The request's task token lives in `tokenizer.sot_sequence`; the
            # optional initial prompt travels as previous tokens (upstream
            # parity with `generate_segment_batched`).
            previous_tokens = (
                tokenizer.encode(request_windows.initial_prompt)
                if request_windows.initial_prompt
                else []
            )
            request_windows.prompt = self.model.get_prompt(
                tokenizer,
                previous_tokens=previous_tokens,
                without_timestamps=True,
                prefix=None,
                hotwords=None,
            )
        return request_windows.tokenizer

    # ------------------------------------------------------------------
    # One formed batch: one encode + one generate
    # ------------------------------------------------------------------
    def _run_batch(self, rows) -> None:
        active: list[tuple[_RequestWindows, _Window]] = []
        for request_windows, window in rows:
            if _request_finished(request_windows.request):
                request_windows.error = request_windows.error or _aborted(request_windows.request)
                continue
            active.append((request_windows, window))
        if not active:
            return

        features = np.stack([window.features for _request_windows, window in active])
        encoder_output = self.model.encode(features)
        self._detect_languages(active, encoder_output)

        prompts: list[list[int]] = []
        suppress_tokens = None
        for request_windows, _window in active:
            tokenizer = self._tokenizer_for(request_windows)
            prompts.append(request_windows.prompt)
            if suppress_tokens is None:
                suppress_tokens = get_suppressed_tokens(tokenizer, [-1])

        self.batches_formed += 1
        self.windows_batched += len(active)
        logger.debug(
            f"Whisper window batcher: one encode + one generate for {len(active)} window(s) "
            f"from {len({request_windows.request_index for request_windows, _window in active})} request(s)"
        )
        results = self.model.model.generate(
            encoder_output,
            prompts,
            beam_size=_WHISPER_BEAM_SIZE,
            patience=_WHISPER_PATIENCE,
            length_penalty=_WHISPER_LENGTH_PENALTY,
            repetition_penalty=_WHISPER_REPETITION_PENALTY,
            no_repeat_ngram_size=_WHISPER_NO_REPEAT_NGRAM_SIZE,
            max_length=int(self.model.max_length),
            suppress_blank=True,
            suppress_tokens=suppress_tokens,
            return_scores=True,
            return_no_speech_prob=True,
            sampling_temperature=_WHISPER_TEMPERATURE,
        )
        if len(results) != len(active):
            raise RuntimeError(
                f"CTranslate2 generate returned {len(results)} results for {len(active)} prompts"
            )

        for (request_windows, window), result in zip(active, results, strict=True):
            if _request_finished(request_windows.request):
                request_windows.error = request_windows.error or _aborted(request_windows.request)
                continue
            try:
                request_windows.decoded[window.window_index] = self._decode_window(
                    request_windows, window, result
                )
            except Exception as error:  # noqa: BLE001 - this request's window only
                logger.opt(exception=error).warning(
                    f"Whisper window batcher: request {request_windows.request_id} "
                    f"failed decoding window {window.window_index}; erroring it alone"
                )
                request_windows.error = request_windows.error or error

    def _decode_window(self, request_windows: _RequestWindows, window: _Window, result) -> list[dict]:
        tokenizer = request_windows.tokenizer
        tokens = list(result.sequences_ids[0])
        sequence_length = len(tokens)
        # Mirrors generate_segment_batched: cumulative score scaled by the
        # length penalty (1.0 here), normalized by seq_len + 1.
        cumulative_logprob = float(result.scores[0]) * (sequence_length**_WHISPER_LENGTH_PENALTY)
        average_logprob = cumulative_logprob / (sequence_length + 1)

        subsegments, _seek, _single_timestamp_ending = self.model._split_segments_by_timestamps(
            tokenizer=tokenizer,
            tokens=tokens,
            time_offset=window.offset,
            segment_size=window.segment_size,
            segment_duration=window.duration,
            seek=0,
        )
        decoded: list[dict] = []
        for subsegment in subsegments:
            text = tokenizer.decode(subsegment["tokens"])
            decoded.append(
                {
                    "text": text,
                    "avg_logprob": average_logprob,
                    "no_speech_prob": float(result.no_speech_prob),
                    "tokens": list(subsegment["tokens"]),
                    "start": float(subsegment["start"]),
                    "end": float(subsegment["end"]),
                    "compression_ratio": get_compression_ratio(text),
                    "seek": window.seek,
                }
            )
        return decoded

    # ------------------------------------------------------------------
    # Per-request stitching → the sequential handler's payload shape
    # ------------------------------------------------------------------
    def _payload(self, request_windows: _RequestWindows) -> dict:
        segments: list[Segment] = []
        for window_index in range(len(request_windows.windows)):
            for item in request_windows.decoded.get(window_index, []):
                segments.append(
                    Segment(
                        id=len(segments) + 1,
                        seek=item["seek"],
                        start=round(item["start"], 3),
                        end=round(item["end"], 3),
                        text=item["text"],
                        tokens=item["tokens"],
                        avg_logprob=item["avg_logprob"],
                        compression_ratio=item["compression_ratio"],
                        no_speech_prob=item["no_speech_prob"],
                        words=None,
                        temperature=_WHISPER_TEMPERATURE,
                    )
                )
        if segments and request_windows.clip_timestamps:
            segments = list(
                restore_speech_timestamps(
                    segments,
                    request_windows.clip_timestamps,
                    int(self.model.feature_extractor.sampling_rate),
                )
            )

        info = self._build_info(request_windows)
        return {
            "text": "".join(segment.text for segment in segments),
            "segments": [asdict(segment) for segment in segments],
            "info": asdict(info),
        }

    def _build_info(self, request_windows: _RequestWindows) -> TranscriptionInfo:
        options = TranscriptionOptions(
            beam_size=_WHISPER_BEAM_SIZE,
            best_of=_WHISPER_BEST_OF,
            patience=_WHISPER_PATIENCE,
            length_penalty=_WHISPER_LENGTH_PENALTY,
            repetition_penalty=_WHISPER_REPETITION_PENALTY,
            no_repeat_ngram_size=_WHISPER_NO_REPEAT_NGRAM_SIZE,
            log_prob_threshold=_WHISPER_LOG_PROB_THRESHOLD,
            no_speech_threshold=_WHISPER_NO_SPEECH_THRESHOLD,
            compression_ratio_threshold=_WHISPER_COMPRESSION_RATIO_THRESHOLD,
            condition_on_previous_text=False,
            prompt_reset_on_temperature=0.5,
            temperatures=[_WHISPER_TEMPERATURE],
            initial_prompt=request_windows.initial_prompt,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=get_suppressed_tokens(request_windows.tokenizer, [-1]),
            without_timestamps=True,
            max_initial_timestamp=0.0,
            word_timestamps=False,
            prepend_punctuations=_WHISPER_PREPEND_PUNCTUATIONS,
            append_punctuations=_WHISPER_APPEND_PUNCTUATIONS,
            multilingual=False,
            max_new_tokens=None,
            clip_timestamps=request_windows.clip_timestamps,
            hallucination_silence_threshold=None,
            hotwords=None,
        )
        return TranscriptionInfo(
            language=request_windows.language,
            language_probability=request_windows.language_probability,
            duration=request_windows.audio_seconds,
            duration_after_vad=request_windows.duration_after_vad,
            all_language_probs=request_windows.all_language_probs,
            transcription_options=options,
            vad_options=request_windows.vad_options,
        )


__all__ = [
    "WHISPER_WINDOW_BATCH_SIZE",
    "WHISPER_WINDOW_SECONDS",
    "WhisperRequestAbortedError",
    "WhisperWindowBatcher",
    "whisper_batched_inference_enabled",
    "whisper_request_options",
]
