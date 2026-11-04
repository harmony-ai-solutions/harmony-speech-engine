import enum
import time

from loguru import logger

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
    FINISHED_FORWARDED = enum.auto()

    @staticmethod
    def is_waiting(status: "RequestStatus") -> bool:
        return status in [RequestStatus.WAITING]

    @staticmethod
    def is_running(status: "RequestStatus") -> bool:
        return status in [RequestStatus.RUNNING]

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_IGNORED,
            RequestStatus.FINISHED_FORWARDED,
        ]

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> str | None:
        if status == RequestStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == RequestStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == RequestStatus.FINISHED_IGNORED:
            finish_reason = "skipped"
        elif status == RequestStatus.FINISHED_FORWARDED:
            finish_reason = "forwarded"
        else:
            finish_reason = None
        return finish_reason


class EngineRequest:
    def __init__(
        self,
        request_id: str,
        request_data: RequestInput,
        arrival_time: float,
        deadline: float | None = None,
        artifacts: dict[str, object] | None = None,
    ) -> None:
        self.request_id = request_id
        self.request_data = request_data
        self.metrics = RequestMetrics(arrival_time=arrival_time, first_scheduled_time=None, time_in_queue=None)
        self.status = RequestStatus.WAITING
        # Monotonic timestamp after which the engine resolves this request as a
        # timeout error (None = no deadline).
        self.deadline = deadline
        # Typed in-process pipeline artifacts, keyed by request-field name
        # (``input_audio``, ``input_embedding``, ``input_vad_data``,
        # ``input_mel_spectrogram``, ...). They take precedence over the wire-encoded
        # request fields (Phase 2-2) and are released deterministically when the
        # request finishes (``clear_artifacts``).
        self.artifacts: dict[str, object] = artifacts if artifacts is not None else {}

    def get_artifact(self, field: str):
        """Return the artifact cached for ``field``, or ``None``."""
        artifacts = getattr(self, "artifacts", None)
        if not artifacts:
            return None
        return artifacts.get(field)

    def set_artifact(self, field: str, artifact) -> None:
        """Cache ``artifact`` under ``field`` (creating the store if needed)."""
        if getattr(self, "artifacts", None) is None:
            self.artifacts = {}
        self.artifacts[field] = artifact

    def clear_artifacts(self) -> None:
        """Drop every artifact reference so VRAM/host memory is released now.

        Called when the request reaches a ``FINISHED_*`` state. The dict itself
        is dropped; artifacts shared with a forwarded request live in that
        request's own (copied) store, so clearing here never destroys data a
        later pipeline step still needs.
        """
        artifacts = getattr(self, "artifacts", None)
        if not artifacts:
            self.artifacts = None
            return
        fields = ", ".join(sorted(artifacts))
        artifacts.clear()
        self.artifacts = None
        logger.debug(f"Request {self.request_id}: released artifacts [{fields}]")

    def is_waiting(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def is_running(self) -> bool:
        return RequestStatus.is_running(self.status)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def set_running(self):
        if not RequestStatus.is_running(self.status) and not RequestStatus.is_finished(self.status):
            self.status = RequestStatus.RUNNING
            self.metrics.first_scheduled_time = time.monotonic()
            self.metrics.time_in_queue = self.metrics.first_scheduled_time - self.metrics.arrival_time


class ExecutorResult:
    def __init__(self, request_id: str, input_data: RequestInput, result_data: RequestOutput):
        self.request_id = request_id
        self.input_data = input_data
        self.result_data = result_data
