from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Args:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
    """

    arrival_time: float
    first_scheduled_time: Optional[float]
    time_in_queue: Optional[float]
    finished_time: Optional[float] = None
