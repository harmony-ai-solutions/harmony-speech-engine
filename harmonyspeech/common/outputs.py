from typing import TYPE_CHECKING

from harmonyspeech.common.metrics import RequestMetrics

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact

__all__ = [
    "RequestOutput",
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
    """

    def __init__(
        self,
        request_id: str,
        finish_reason: str | None = None,
        error: str | None = None,
        metrics: RequestMetrics | None = None,
    ) -> None:
        self.request_id = request_id
        self.finish_reason = finish_reason
        self.error = error
        self.metrics = metrics

    def finished(self) -> bool:
        return self.finish_reason is not None


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
