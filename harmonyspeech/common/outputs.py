from typing import TYPE_CHECKING

from harmonyspeech.common.metrics import RequestMetrics

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact

__all__ = [
    "RequestOutput",
    "UnfinishedRequestOutput",
    "TextToSpeechRequestOutput",
    "SpeechEmbeddingRequestOutput",
    "SpeechSynthesisRequestOutput",
    "VocodeRequestOutput",
    "AudioConversionRequestOutput",
    "VoiceConversionRequestOutput",
    "SpeechTranscriptionRequestOutput",
    "DetectVoiceActivityRequestOutput",
]


class RequestOutput:
    """
    The base class for model output data.
    Args:
        request_id: The unique ID of the request.
        finish_reason: Reason why this request finished, if set.
        error: Error message if finish_reason is "error", otherwise None.
        metrics: Metrics associated with the request.
        unfinished: ``True`` for a partial/progress result from an executor
            that is still decoding this request across engine steps (Phase
            9-4). Such a result is consumed by the engine, never handed to the
            client; it has no ``finish_reason``.
        progress: Per-request progress payload for an ``unfinished`` result
            (convention: ``{"tokens": int, "step": int}``). ``None`` for a
            terminal result.
    """

    def __init__(
        self,
        request_id: str,
        finish_reason: str | None = None,
        error: str | None = None,
        metrics: RequestMetrics | None = None,
        unfinished: bool = False,
        progress: dict | None = None,
    ) -> None:
        self.request_id = request_id
        self.finish_reason = finish_reason
        self.error = error
        self.metrics = metrics
        self.unfinished = bool(unfinished)
        self.progress = progress

    def finished(self) -> bool:
        return self.finish_reason is not None

    def is_unfinished(self) -> bool:
        """True when this is a partial decode result (never client-visible)."""
        return self.unfinished


class UnfinishedRequestOutput(RequestOutput):
    """Partial result for a request that is still decoding (Phase 9-4).

    An executor returns this instead of a final output while its row is still
    producing tokens. The engine records ``progress`` on the in-flight
    ``EngineRequest`` and keeps the request in the scheduler's ``running`` set
    for the next step; the client receives exactly one final response once a
    terminal result arrives.

    ``finish_reason`` is deliberately ``None``: :meth:`RequestOutput.finished`
    stays ``False`` so no code path can mistake progress for completion. The
    ``unfinished`` marker (not a null finish reason) is what the engine keys on,
    because an error result also carries ``finish_reason``.
    """

    def __init__(
        self,
        request_id: str,
        progress: dict | None = None,
        metrics: RequestMetrics | None = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            finish_reason=None,
            metrics=metrics,
            unfinished=True,
            progress=dict(progress or {}),
        )

    def __repr__(self) -> str:
        return f"UnfinishedRequestOutput(request_id={self.request_id}, progress={self.progress})"


class TextToSpeechRequestOutput(RequestOutput):
    """
    The output Data of a Text-to-Speech Request.

    Args:
        request_id: The unique ID of the request.
        text: The text string used to generate the result.
        output: The generated output data, encoded in base64. Empty when
            ``audio_artifact`` carries the result (Phase 2 artifact bus).
        finish_reason: Reason why this request finished, if set.
        metrics: Metrics associated with the request.
        audio_artifact: Optional typed waveform for in-process consumers; ignored
            by wire serialization, which encodes from it at the boundary.
    """

    def __init__(
        self,
        request_id: str,
        text: str,
        output: str,
        finish_reason: str | None = None,
        metrics: RequestMetrics | None = None,
        audio_artifact: "AudioArtifact | None" = None,
    ) -> None:
        super().__init__(request_id=request_id, finish_reason=finish_reason, metrics=metrics)
        self.input_text = text
        self.output = output
        self.audio_artifact = audio_artifact

    def __repr__(self) -> str:
        return f"TextToSpeechRequestOutput(request_id={self.request_id}, data=bytes({len(self.output)})"


class SpeechEmbeddingRequestOutput(RequestOutput):
    """
    The output Data of a Speech Embedding Request.

    Args:
        request_id: The unique ID of the request.
        output: The generated output data, encoded in base64. Empty when
            ``embedding_artifact`` carries the embedding/conditionals.
        finish_reason: Reason why this request finished, if set.
        metrics: Metrics associated with the request.
        embedding_artifact: Optional typed speaker artifact. The serving layer
            serializes it for external embed clients (0-1-safe format).
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: str | None = None,
        metrics: RequestMetrics | None = None,
        embedding_artifact: "SpeakerArtifact | None" = None,
    ) -> None:
        super().__init__(request_id=request_id, finish_reason=finish_reason, metrics=metrics)
        self.output = output
        self.embedding_artifact = embedding_artifact

    def __repr__(self) -> str:
        return f"SpeechEmbeddingRequestOutput(request_id={self.request_id}, data=bytes({len(self.output)})"


class SpeechSynthesisRequestOutput(RequestOutput):
    """
    The output Data of a Speech Synthesis Request.

    ``mel_artifact`` carries HarmonySpeech's mel+breaks; ``audio_artifact``
    carries backends that produce a waveform directly (OpenVoice/Melo). Both are
    additive and ignored by wire serialization.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: str | None = None,
        metrics: RequestMetrics | None = None,
        audio_artifact: "AudioArtifact | None" = None,
        mel_artifact: "MelArtifact | None" = None,
    ) -> None:
        super().__init__(request_id=request_id, finish_reason=finish_reason, metrics=metrics)
        self.output = output
        self.audio_artifact = audio_artifact
        self.mel_artifact = mel_artifact

    def __repr__(self) -> str:
        return f"SpeechSynthesisRequestOutput(request_id={self.request_id}, data=bytes({len(self.output)})"


class VocodeRequestOutput(RequestOutput):
    """
    The output Data of a Vocoder Request.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: str | None = None,
        metrics: RequestMetrics | None = None,
        audio_artifact: "AudioArtifact | None" = None,
    ) -> None:
        super().__init__(request_id=request_id, finish_reason=finish_reason, metrics=metrics)
        self.output = output
        self.audio_artifact = audio_artifact

    def __repr__(self) -> str:
        return f"VocodeRequestOutput(request_id={self.request_id}, data=bytes({len(self.output)})"


class AudioConversionRequestOutput(RequestOutput):
    """
    The output Data of an Audio Conversion Request.

    ``mel_artifact`` (restorer) and ``audio_artifact`` (vocoder) are the internal
    pipeline results; ``output`` keeps the legacy wire representation for
    callers that do not go through the artifact-aware serving layer.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: str | None = None,
        metrics: RequestMetrics | None = None,
        audio_artifact: "AudioArtifact | None" = None,
        mel_artifact: "MelArtifact | None" = None,
    ) -> None:
        super().__init__(request_id=request_id, finish_reason=finish_reason, metrics=metrics)
        self.output = output
        self.audio_artifact = audio_artifact
        self.mel_artifact = mel_artifact

    def __repr__(self) -> str:
        return f"AudioConversionRequestOutput(request_id={self.request_id}, data=bytes({len(self.output)})"


class VoiceConversionRequestOutput(RequestOutput):
    """
    The output Data of a Voice Conversion Request.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: str | None = None,
        metrics: RequestMetrics | None = None,
        audio_artifact: "AudioArtifact | None" = None,
    ) -> None:
        super().__init__(request_id=request_id, finish_reason=finish_reason, metrics=metrics)
        self.output = output
        self.audio_artifact = audio_artifact

    def __repr__(self) -> str:
        return f"VoiceConversionRequestOutput(request_id={self.request_id}, data=bytes({len(self.output)})"


class SpeechTranscriptionRequestOutput(RequestOutput):
    """
    The output Data of a Speech Transcription Request.

    ``vad_artifact`` carries the parsed segments/info so the OpenVoice embedding
    step consumes them directly instead of re-parsing JSON.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: str | None = None,
        metrics: RequestMetrics | None = None,
        vad_artifact: "VadArtifact | None" = None,
    ) -> None:
        super().__init__(request_id=request_id, finish_reason=finish_reason, metrics=metrics)
        self.output = output
        self.vad_artifact = vad_artifact

    def __repr__(self) -> str:
        return f"SpeechTranscriptionRequestOutput(request_id={self.request_id}, data=bytes({len(self.output)})"


class DetectVoiceActivityRequestOutput(RequestOutput):
    """
    The output Data of a VAD Request.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: str | None = None,
        metrics: RequestMetrics | None = None,
        vad_artifact: "VadArtifact | None" = None,
    ) -> None:
        super().__init__(request_id=request_id, finish_reason=finish_reason, metrics=metrics)
        self.output = output
        self.vad_artifact = vad_artifact

    def __repr__(self) -> str:
        return f"DetectVoiceActivityRequestOutput(request_id={self.request_id}, data=bytes({len(self.output)})"
