import enum
import time
from dataclasses import dataclass
from typing import Union, Optional

from harmonyspeech.common.inputs import RequestInput
from harmonyspeech.common.metrics import RequestMetrics
from harmonyspeech.common.outputs import RequestOutput


class RequestStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_running(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.RUNNING,
        ]

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
            finish_reason = "skipped"
        else:
            finish_reason = None
        return finish_reason


class EngineRequest:
    def __init__(
        self,
        request_id: str,
        request_data: RequestInput,
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.request_data = request_data
        self.metrics = RequestMetrics(
            arrival_time=arrival_time,
            first_scheduled_time=None,
            time_in_queue=None,
        )
        self.status = RequestStatus.WAITING

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def set_running(self):
        if not RequestStatus.is_running(self.status) and not RequestStatus.is_finished(self.status):
            self.status = RequestStatus.RUNNING
            self.metrics.first_scheduled_time = time.monotonic()
            self.metrics.time_in_queue = self.metrics.first_scheduled_time - self.metrics.arrival_time


class ExecutorResult:
    def __init__(
        self,
        request_id: str,
        result_data: RequestOutput
    ):
        self.request_id = request_id
        self.request_data = result_data
