from typing import Optional, List

from harmonyspeech.common.request import RequestMetrics, ExecutorResult


class RequestOutput:
    """
    The base class for model output data.
    Args:
        request_id: The unique ID of the request.
        finished: Whether the whole request is finished.
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


class GeneratedSpeechOutput:
    """
    The output of a TTS Generate Speech request.

    Args:
        index: The index of the output in the request.
        data: The generated output audio data.
    """
    def __init__(
        self,
        index: int,
        data: bytes,
    ) -> None:
        self.index = index
        self.data = data

    def __repr__(self) -> str:
        return (f"GeneratedSpeechOutput(index={self.index}, "
                f"data=bytes({len(self.data)})")


class TextToSpeechRequestOutput(RequestOutput):
    """
    The output Data of a Text-to-Speech Request to the model.

    Args:
        request_id: The unique ID of the request.
        input_text: The input text string of the request.
        output: The output of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
    """

    def __init__(
        self,
        request_id: str,
        input_text: str,
        output: GeneratedSpeechOutput,
        finish_reason: Optional[str] = None,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            finish_reason=finish_reason,
            metrics=metrics,
        )
        self.input_text = input_text
        self.output: output
