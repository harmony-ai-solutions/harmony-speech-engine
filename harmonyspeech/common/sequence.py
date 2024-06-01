import enum
from dataclasses import dataclass
from typing import Union, Optional


class SequenceStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


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

