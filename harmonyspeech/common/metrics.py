from dataclasses import dataclass
from typing import Optional


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
