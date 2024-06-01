from typing import Optional, List

from harmonyspeech.common.sequence import RequestMetrics


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
        finished: bool,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        self.request_id = request_id
        self.finished: finished
        self.metrics: metrics


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
        finish_reason: Optional[str] = None,
    ) -> None:
        self.index = index
        self.data = data
        self.finish_reason = finish_reason

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"GeneratedSpeechOutput(index={self.index}, "
                f"data=bytes({len(self.data)}), "
                f"finish_reason={self.finish_reason})")


class TextToSpeechRequestOutput(RequestOutput):
    """
    The output Data of a Text-to-Speech Request to the model.

    Args:
        request_id: The unique ID of the request.
        input: The input string of the request.
        output: The output of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
    """

    def __init__(
        self,
        request_id: str,
        input: str,
        output: GeneratedSpeechOutput,
        finished: bool,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            finished=finished,
            metrics=metrics,
        )
        self.input = input
        self.output: output