from typing import Optional, List

from harmonyspeech.common.metrics import RequestMetrics


class RequestOutput:
    """
    The base class for model output data.
    Args:
        request_id: The unique ID of the request.
        finish_reason: Reason why this request finished, if set.
        metrics: Metrics associated with the request.
    """
    def __init__(
        self,
        request_id: str,
        finish_reason: Optional[str] = None,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        self.request_id = request_id
        self.finish_reason = finish_reason
        self.metrics: metrics

    def finished(self) -> bool:
        return self.finish_reason is not None


class TextToSpeechRequestOutput(RequestOutput):
    """
    The output Data of a Text-to-Speech Request.

    Args:
        request_id: The unique ID of the request.
        text: The text string used to generate the result.
        output: The generated output data, encoded in base64.
        finish_reason: Reason why this request finished, if set.
        metrics: Metrics associated with the request.
    """

    def __init__(
        self,
        request_id: str,
        text: str,
        output: str,
        finish_reason: Optional[str] = None,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            finish_reason=finish_reason,
            metrics=metrics,
        )
        self.input_text = text
        self.output = output

    def __repr__(self) -> str:
        return (f"TextToSpeechRequestOutput(request_id={self.request_id}, "
                f"data=bytes({len(self.output)})")


class SpeechEmbeddingRequestOutput(RequestOutput):
    """
    The output Data of a Speech Embedding Request.

    Args:
        request_id: The unique ID of the request.
        output: The generated output data, encoded in base64.
        finish_reason: Reason why this request finished, if set.
        metrics: Metrics associated with the request.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: Optional[str] = None,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            finish_reason=finish_reason,
            metrics=metrics,
        )
        self.output = output

    def __repr__(self) -> str:
        return (f"SpeechEmbeddingRequestOutput(request_id={self.request_id}, "
                f"data=bytes({len(self.output)})")


class SpeechSynthesisRequestOutput(RequestOutput):
    """
    The output Data of a Speech Synthesis Request.

    Args:
        request_id: The unique ID of the request.
        output: The generated output data, encoded in base64.
        finish_reason: Reason why this request finished, if set.
        metrics: Metrics associated with the request.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: Optional[str] = None,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            finish_reason=finish_reason,
            metrics=metrics,
        )
        self.output = output

    def __repr__(self) -> str:
        return (f"SpeechSynthesisRequestOutput(request_id={self.request_id}, "
                f"data=bytes({len(self.output)})")


class VocodeRequestOutput(RequestOutput):
    """
    The output Data of a Vocoder Request.

    Args:
        request_id: The unique ID of the request.
        output: The generated output data, encoded in base64.
        finish_reason: Reason why this request finished, if set.
        metrics: Metrics associated with the request.
    """

    def __init__(
        self,
        request_id: str,
        output: str,
        finish_reason: Optional[str] = None,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            finish_reason=finish_reason,
            metrics=metrics,
        )
        self.output = output

    def __repr__(self) -> str:
        return (f"VocodeRequestOutput(request_id={self.request_id}, "
                f"data=bytes({len(self.output)})")

