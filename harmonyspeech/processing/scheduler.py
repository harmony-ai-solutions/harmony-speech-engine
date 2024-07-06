import enum
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.request import (RequestStatus, RequestMetrics, EngineRequest)


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

    def current_num_requests(self, model: str):
        return 0 if model not in self._num_per_model_requests else self._num_per_model_requests[model]

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
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled requests per model.
    scheduled_requests_per_model: Dict[str, List[EngineRequest]]
    # Sequence groups that are going to be ignored.
    ignored_requests: List[EngineRequest]

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return not self.scheduled_requests_per_model


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

    def update_request_status(self, request_id: Union[str, Iterable[str]], new_status: RequestStatus) -> None:
        """Updates status for requests with the given ID.

        Check if the requests with the given ID are present in any of the state queues.
        If present (and new status is finished), remove the sequence group from the state queue.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
            new_status: New Status to set for the requests
        """
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running]:
            updated_requests: List[EngineRequest] = []
            for request in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity .
                    break
                if request.request_id in request_ids:
                    # Appending aborted group into pending list.
                    updated_requests.append(request)
                    request_ids.remove(request.request_id)

            for request in updated_requests:
                # Skip if already finished; but remove from queue
                if request.is_finished():
                    state_queue.remove(request)
                    continue
                # Update state
                request.status = new_status
                if request.is_finished():
                    request.metrics.finished_time = time.time()
                    # Remove from queue since it is finished now
                    state_queue.remove(request)

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

        new_scheduled_for_models = []
        leftover_waiting_requests = deque()
        while waiting_queue:
            # Get Request and model name
            request = waiting_queue[0]
            model_name = request.request_input.model
            # Remove from queue
            waiting_queue.popleft()

            # Do not schedule if there are already running requests for that model currently
            if budget.current_num_requests(model_name) > 0 and model_name not in new_scheduled_for_models:
                leftover_waiting_requests.append(request)
                continue
            # Ignore if budget for this model is exceeded
            if budget.remaining_request_budget(model_name) <= 0:
                leftover_waiting_requests.append(request)
                continue

            # Can schedule this request.
            requests.append(request)
            budget.add_num_requests(request.request_input.model, 1)
            # Keep track of model name in the new scheduled list
            # This way we know it's safe to add new requests for this model
            if model_name not in new_scheduled_for_models:
                new_scheduled_for_models.append(model_name)

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
        """Schedule queued requests."""
        # Calculate the overall scheduling budget
        budget = self.get_scheduling_budget()
        # Include running requests to the budget.
        for request in self.running:
            budget.add_num_requests(request.request_data.model, 1)

        # Schedule waiting requests if there is capacity
        # Only requests for models not currently processing a batch will be scheduled
        remaining_waiting, ready = self._schedule_waiting_requests(self.waiting, budget)

        # Update waiting requests.
        self.waiting = remaining_waiting

        # Build mapping of ready requests on a per model dictionary
        per_model_scheduled = {}
        for r in ready.requests:
            # Set Request to running state
            r.set_running()
            # Fill into dict
            if r.request_data.model not in per_model_scheduled.keys():
                per_model_scheduled[r.request_data.model] = [r]
            else:
                per_model_scheduled[r.request_data.model].append(r)

        # Update running requests.
        self.running.extend(ready.requests)

        return SchedulerOutputs(
            scheduled_requests_per_model=per_model_scheduled,
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
