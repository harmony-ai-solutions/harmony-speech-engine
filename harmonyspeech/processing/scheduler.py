import enum
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.sequence import (RequestStatus, RequestMetrics, EngineRequest)


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.
    Slots are being defined
    """
    request_per_model_budget: Dict[str, int]
    _num_per_model_requests: Dict[str, int]

    def can_schedule(self, model: str, num_requests: int):
        assert num_requests != 0
        current_requests = 0 if model not in self._num_per_model_requests else self._num_per_model_requests[model]
        return current_requests + num_requests <= self.request_per_model_budget[model]

    def remaining_request_budget(self, model: str):
        return self.request_per_model_budget[model] - self.num_per_model_requests[model]

    def add_num_requests(self, model: str, num_requests: int):
        current_requests = 0 if model not in self._num_per_model_requests else self._num_per_model_requests[model]
        self._num_per_model_requests[model] = current_requests + num_requests

    def subtract_num_requests(self, model: str, num_requests: int):
        current_requests = 0 if model not in self._num_per_model_requests else self._num_per_model_requests[model]
        assert current_requests >= num_requests
        self._num_per_model_requests[model] = current_requests - num_requests

    @property
    def num_per_model_requests(self):
        return self._num_per_model_requests


@dataclass
class ScheduledEngineRequest:
    # Reference to a request which is scheduled
    request: EngineRequest


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_requests: Iterable[ScheduledEngineRequest]
    # Sequence groups that are going to be ignored.
    ignored_requests: List[EngineRequest]

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return not self.scheduled_requests


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.
    """
    # Selected sequences that are running
    requests: List[EngineRequest]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            requests=[],
        )


@dataclass
class SchedulerWaitingOutputs:
    """The requests that are scheduled from a waiting queue.
    """
    # Selected requests for processing
    requests: List[EngineRequest]
    # Ignored Requests
    ignored_requests: List[EngineRequest]

    @classmethod
    def create_empty(cls) -> "SchedulerWaitingOutputs":
        return SchedulerWaitingOutputs(
            requests=[],
            ignored_requests=[],
        )


class Scheduler:

    def __init__(
        self,
        model_configs: Optional[List[ModelConfig]]
    ) -> None:
        self.model_configs = model_configs

        # Requests in the WAITING state.
        self.waiting: Deque[EngineRequest] = deque()
        # Requests in the RUNNING state.
        self.running: Deque[EngineRequest] = deque()

        # Time at previous scheduling step
        self.prev_time = 0.0

    def add_request(self, request: EngineRequest) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(request)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running]:
            aborted_requests: List[EngineRequest] = []
            for request in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity .
                    break
                if request.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_requests.append(request)
                    request_ids.remove(request.request_id)
            for aborted_request in aborted_requests:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_request)
                if aborted_request.is_finished():
                    continue
                aborted_request.status = RequestStatus.FINISHED_ABORTED

    def has_unfinished_requests(self) -> bool:
        return self.waiting or self.running

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def _schedule_waiting_requests(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
    ) -> Tuple[deque, SchedulerWaitingOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        ignored_requests: List[EngineRequest] = []
        requests: List[EngineRequest] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_sequences = deque()
        while waiting_queue:
            request = waiting_queue[0]

            # Abort if budget is exceeded
            if budget.remaining_request_budget(request.request_data.model) <= 0:
                break

            # Can schedule this request.
            waiting_queue.popleft()
            requests.append(request)
            budget.add_num_requests(request.request_data.model, 1)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        # if len(requests) > 0:
        #     self.prev_prompt = True

        return waiting_queue, SchedulerWaitingOutputs(
            requests=requests,
            ignored_requests=ignored_requests)

    def get_scheduling_budget(self) -> SchedulingBudget:
        """ Calculate the overall scheduling Budget for requests per Model """
        budget_count = {}
        per_model_requests = {}
        for model_cfg in self.model_configs:
            budget_count[model_cfg.model] = model_cfg.max_batch_size
            per_model_requests[model_cfg.model] = 0

        return SchedulingBudget(
            request_per_model_budget=budget_count,
            _num_per_model_requests=per_model_requests
        )

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to opimimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Calculate the overall scheduling budget
        budget = self.get_scheduling_budget()
        # Include running requests to the budget.
        for request in self.running:
            budget.add_num_requests(request.request_data.model, 1)

        # Schedule waiting requests if there is capacity
        remaining_waiting, ready = self._schedule_waiting_requests(self.waiting, budget)

        # Update waiting requests.
        self.waiting = remaining_waiting
        # Update running requests.
        self.running.extend([r for r in ready.requests])

        return SchedulerOutputs(
            scheduled_requests=[ScheduledEngineRequest(request=r) for r in self.running],
            ignored_requests=[]
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        return self._schedule_default()

    def schedule(self) -> SchedulerOutputs:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running and self.waiting.
        scheduler_outputs = self._schedule()
        self.prev_time = time.time()
        return scheduler_outputs

    def free_finished_requests(self) -> None:
        self.running = deque(request for request in self.running if not request.is_finished())
