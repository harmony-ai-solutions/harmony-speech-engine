import enum
from dataclasses import dataclass
from typing import Union, Optional

from harmonyspeech.endpoints.openai.protocol import BaseRequest


class RequestStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        if status == RequestStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == RequestStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == RequestStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Args:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_render_time: The time when the actual processing started.
        last_render_time: The time when the processing ended.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
    """

    arrival_time: float
    first_scheduled_time: Optional[float]
    first_render_time: Optional[float]
    last_render_time: float
    time_in_queue: Optional[float]
    finished_time: Optional[float] = None


class EngineRequest:

    def __init__(
        self,
        request_id: str,
        request_data: BaseRequest,
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.request_data = request_data,
        self.metrics = RequestMetrics(
            arrival_time=arrival_time,
            first_scheduled_time=None,
            time_in_queue=None,
        )
        self.status = RequestStatus.WAITING

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)
