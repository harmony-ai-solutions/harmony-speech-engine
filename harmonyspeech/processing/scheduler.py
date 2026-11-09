import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from loguru import logger

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.request import EngineRequest, RequestStatus

#: faster-whisper cuts audio into <=30 s clips, so that is also the boundary
#: between the two fairness lanes (Phase 10-1). The duration is already known at
#: admission: the artifact bus decoded the audio exactly once in
#: ``add_request``/``from_wire_audio``, so lane selection costs no extra decode.
WHISPER_SHORT_LANE_MAX_SECONDS = 30.0

#: Credit ratio used to drain the two Whisper lanes: three short requests for
#: every long one. When one lane is empty the other is served without idling, so
#: a burst of long audio can never starve short requests beyond this ratio (and
#: vice versa).
WHISPER_SHORT_CREDIT = 3
WHISPER_LONG_CREDIT = 1

#: faster-whisper's windowing scale (one 30 s clip). A request's admission cost
#: is the number of such windows its audio can produce (Phase 10-2): the memory
#: governor bounds *windows* in flight, not requests.
WHISPER_WINDOW_SECONDS = 30.0

#: Model types that decode autoregressively across engine steps (Phase 9-4).
#: Only these families get capacity-based admission and running-set carry-over;
#: every other family keeps the historical boolean rule byte-for-byte. The set
#: mirrors the Phase 9 AR-runner members (English + Turbo Chatterbox, see
#: ``docs/batching.md``); multilingual/embedding/VC stay per-request loops and
#: are deliberately excluded (multilingual is the documented 9-3 skip). 9-6
#: moves the shipped config entries into the ``ar_decode`` execution class; this
#: runtime set stays the single source of truth for admission.
AR_INFERENCE_MODEL_TYPES = frozenset({"ChatterboxTTS", "ChatterboxTurboTTS", "ChatterboxMultilingualTTS"})


def _artifact_duration(request) -> float | None:
    """Admission-time audio duration in seconds, or ``None`` when unknown."""
    artifacts = getattr(request, "artifacts", None) or {}
    artifact = artifacts.get("input_audio") if hasattr(artifacts, "get") else None
    try:
        return float(artifact.duration_s)
    except (TypeError, ValueError, AttributeError):
        return None


def whisper_lane(request) -> str:
    """Classify a request into the ``"short"`` (<=30 s) or ``"long"`` lane.

    Reads the admission-time audio artifact (``request.artifacts["input_audio"]``)
    — no decode happens here. A request whose duration cannot be determined is
    treated as long: it is the conservative choice (it never gets the short
    lane's priority), and a missing artifact is an input-plumbing bug rather than
    a reason to guess.
    """
    seconds = _artifact_duration(request)
    if seconds is None:
        return "long"
    return "short" if seconds <= WHISPER_SHORT_LANE_MAX_SECONDS else "long"


def whisper_window_count(request) -> int:
    """Estimated 30 s windows a request will form (Phase 10-2 admission cost).

    Uses the admission-time audio duration: ``ceil(seconds / 30)``. VAD can only
    remove audio, so this is an **upper bound** on the windows the collector
    will actually form — conservative for a memory bound, and cheap (no decode,
    no VAD at admission). Unknown/zero durations cost one window.
    """
    seconds = _artifact_duration(request)
    if seconds is None or seconds <= 0:
        return 1
    return max(1, int(math.ceil(seconds / WHISPER_WINDOW_SECONDS)))


class LaneCredits:
    """Credit-bucket drain policy for the two Whisper lanes (Phase 10-1).

    Picks from the short lane until its credit is spent, then one long-lane
    request, refilling both buckets when they are empty. This is the "3 short :
    1 long" ratio: within any round of four picks a long request is guaranteed a
    slot as soon as one is waiting, while short audio still gets the bulk of the
    capacity. When a lane is empty the other lane is served and its bucket
    refills on demand, so capacity is never left idle.

    Credits persist across engine steps for a model (a scheduler owns one
    instance per Whisper model), so the ratio holds over time and not just
    inside one step.
    """

    def __init__(self, short_credit: int = WHISPER_SHORT_CREDIT, long_credit: int = WHISPER_LONG_CREDIT) -> None:
        self.short_credit = max(1, int(short_credit))
        self.long_credit = max(1, int(long_credit))
        self._short = self.short_credit
        self._long = self.long_credit

    def reset(self) -> None:
        self._short = self.short_credit
        self._long = self.long_credit

    def pick(self, short_available: bool, long_available: bool) -> str | None:
        """Return the lane to serve next, or ``None`` when both are empty."""
        if not short_available and not long_available:
            return None
        if short_available and long_available:
            if self._short <= 0 and self._long <= 0:
                self.reset()
            if self._short > 0:
                self._short -= 1
                return "short"
            self._long -= 1
            return "long"
        if short_available:
            if self._short <= 0:
                self._short = self.short_credit
            self._short -= 1
            return "short"
        if self._long <= 0:
            self._long = self.long_credit
        self._long -= 1
        return "long"


def plan_lane_admission(
    candidates,
    capacity: int,
    credits: LaneCredits,
    *,
    window_costs: dict[int, int] | None = None,
    window_budget: int | None = None,
) -> tuple[list, list]:
    """Pick up to ``capacity`` requests, draining the two lanes by credit.

    ``candidates`` is an arrival-ordered ``[(request, lane)]`` list. Returns
    ``(selected, leftover)``: both preserve arrival order within their lane, so
    FIFO fairness inside a lane is untouched and only the inter-lane ratio is
    new.

    Phase 10-2 adds an optional **window budget**: ``window_costs`` maps
    ``id(request)`` to its 30 s window count, and ``window_budget`` caps the sum
    of admitted windows this step (the memory governor's allowance). A request
    that would exceed the remaining budget is left for a later step while the
    planner keeps scanning the other lane, so a large request cannot block a
    small one behind it. ``capacity`` remains the request-count bound
    (``max_batch_size``).
    """
    capacity = max(0, int(capacity or 0))
    if capacity == 0:
        return [], [request for request, _lane in candidates]

    costs = window_costs or {}
    budget = None if window_budget is None else max(0, int(window_budget))

    def cost(request) -> int:
        return max(1, int(costs.get(id(request), 1)))

    short = [request for request, lane in candidates if lane == "short"]
    longs = [request for request, lane in candidates if lane != "short"]
    selected: list = []
    used_windows = 0
    short_index = long_index = 0
    while len(selected) < capacity and (short_index < len(short) or long_index < len(longs)):
        choice = credits.pick(short_index < len(short), long_index < len(longs))
        if choice == "short":
            request = short[short_index]
            short_index += 1
        else:
            request = longs[long_index]
            long_index += 1
        request_windows = cost(request)
        if budget is not None and used_windows + request_windows > budget:
            # Does not fit this step; leave it for the next one and keep
            # scanning so the other lane (and smaller requests) can proceed.
            continue
        selected.append(request)
        used_windows += request_windows

    selected_ids = {id(request) for request in selected}
    leftover = [request for request, _lane in candidates if id(request) not in selected_ids]
    return selected, leftover


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.
    Slots are being defined
    """

    request_per_model_budget: dict[str, int]
    _num_per_model_requests: dict[str, int]

    def can_schedule(self, model: str, num_requests: int):
        assert num_requests != 0
        current_requests = 0 if model not in self._num_per_model_requests else self._num_per_model_requests[model]
        return current_requests + num_requests <= self.request_per_model_budget[model]

    def current_num_requests(self, model: str):
        return 0 if model not in self._num_per_model_requests else self._num_per_model_requests[model]

    def remaining_request_budget(self, model: str):
        if model not in self.request_per_model_budget:
            return 0
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
    scheduled_requests_per_model: dict[str, list[EngineRequest]]
    # Sequence groups that are going to be ignored.
    ignored_requests: list[EngineRequest]

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return not self.scheduled_requests_per_model


@dataclass
class SchedulerWaitingOutputs:
    """The requests that are scheduled from a waiting queue."""

    # Selected requests for processing
    requests: list[EngineRequest]
    # Ignored Requests
    ignored_requests: list[EngineRequest]

    @classmethod
    def create_empty(cls) -> "SchedulerWaitingOutputs":
        return SchedulerWaitingOutputs(requests=[], ignored_requests=[])


class Scheduler:
    def __init__(
        self,
        model_configs: list[ModelConfig] | None,
        whisper_admission=None,
        ar_admission=None,
    ) -> None:
        self.model_configs = model_configs

        # Requests in the WAITING state.
        self.waiting: deque[EngineRequest] = deque()
        # Requests in the RUNNING state.
        self.running: deque[EngineRequest] = deque()

        # Time at previous scheduling step
        self.prev_time = 0.0

        # Phase 10-1: per-model lane credits + the engine's memory governor.
        # `whisper_admission(active_windows) -> dict` is
        # `HarmonySpeechEngine.get_whisper_admission` (Phase 10-3); None keeps
        # the scheduler standalone (tests, tooling) with capacity-only admission.
        self.whisper_admission = whisper_admission
        # Phase 9-4: capacity-based admission input for AR families. The engine
        # injects `get_ar_cache_capacity`, which returns the 9-2 pool's
        # per-family capacity row (`slots`, `slots_active`, `slots_free`,
        # `active_rows`). None (or a disabled/unregistered arena) keeps the
        # historical boolean rule, so a deployment without AR models — or
        # before 9-3 registers an arena — behaves exactly as before.
        self.ar_admission = ar_admission
        self._model_types = {
            getattr(config, "name", None): getattr(config, "model_type", None)
            for config in (model_configs or [])
        }
        self._whisper_credits: dict[str, LaneCredits] = {}

    def add_request(self, request: EngineRequest) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(request)

    def update_request_status(self, request_id: str | Iterable[str], new_status: RequestStatus) -> None:
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
            updated_requests: list[EngineRequest] = []
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
                    # Deterministic artifact release (Phase 2-1): forwarded
                    # requests already carry a copy of the dict they need.
                    clear_artifacts = getattr(request, "clear_artifacts", None)
                    if callable(clear_artifacts):
                        clear_artifacts()
                    # Multi-step decode state is request-scoped too (Phase 9-4):
                    # drop the unfinished flag/progress on every terminal path.
                    clear_progress = getattr(request, "clear_progress", None)
                    if callable(clear_progress):
                        clear_progress()
                    # Remove from queue since it is finished now
                    state_queue.remove(request)

    def has_unfinished_requests(self) -> bool:
        return self.waiting or self.running

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def _schedule_waiting_requests(
        self, waiting_queue: deque, budget: SchedulingBudget
    ) -> tuple[deque, SchedulerWaitingOutputs]:
        """Schedule waiting requests, preserving the historical per-model rule.

        Three admission regimes share this method (Phase 9-4 / 10-1):

        * **AR families are capacity-based** (Phase 9-4): an unfinished row keeps
          decoding in ``running``, so admission continues while the shared pool
          has a free slot and the per-step width allows another row. Slots come
          from the injected ``ar_admission`` hook (9-2's pool capacity); without
          a live capacity signal the boolean rule applies instead.
        * **Every other non-Whisper model keeps the boolean rule**: a model that
          still has running requests is skipped this step, and it admits in
          arrival order up to its ``max_batch_size``. This is exactly the
          pre-Phase-10 behaviour.
        * **Whisper models are capacity-based**: they keep admitting while the
          budget allows, and the step is filled from two fairness lanes
          (``<=30 s`` vs ``>30 s``) drained at the credit ratio. The final width
          composes three bounds — the configured capacity
          (``max_batch_size``), the memory governor's allowed in-flight windows
          (Phase 10-3 ``get_whisper_admission``) minus the windows already
          running (Phase 10-2: a request costs ``ceil(seconds / 30)`` windows,
          not one), and the lane credits that decide *which* requests fill it.
          The executor's own worker semaphore (``num_workers``) is the hard
          in-flight bound below all of this.

        Args:
            waiting_queue: The queue that contains prefill requests. The given
                argument is NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.

        Returns:
            A tuple of the remaining waiting queue and the scheduled requests.
        """
        ignored_requests: list[EngineRequest] = []
        requests: list[EngineRequest] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])

        # Group in arrival order so per-model FIFO order (and the old boolean
        # rule) is untouched; only the inter-lane selection is new.
        by_model: dict[str, list[EngineRequest]] = {}
        for request in waiting_queue:
            model_name = request.request_data.model
            # Defence in depth (Phase 0-4): a request whose model is not part of
            # the scheduling budget at all can never be scheduled. Surface it as
            # an ignored request instead of requeueing it forever.
            if model_name not in budget.request_per_model_budget:
                logger.warning(
                    f"Request {request.request_id} targets unknown model '{model_name}' — "
                    f"marking it ignored instead of waiting forever."
                )
                ignored_requests.append(request)
                continue
            by_model.setdefault(model_name, []).append(request)

        leftover_waiting_requests: deque = deque()
        for model_name, model_requests in by_model.items():
            if self._is_ar_model(model_name):
                # Capacity-based admission for AR families (Phase 9-4). An
                # unfinished request keeps decoding in `running`, so the old
                # "model has running work -> admit nothing" boolean would starve
                # the family for the whole decode. Admit while the shared pool
                # has a free slot AND the per-step width (`max_batch_size`)
                # allows another row. Slots are a separate capacity signal,
                # never the admission width (the Phase 10 lesson).
                width = self._ar_admission_width(model_name, budget)
                for request in model_requests:
                    if width <= 0:
                        leftover_waiting_requests.append(request)
                        continue
                    requests.append(request)
                    budget.add_num_requests(model_name, 1)
                    width -= 1
                continue

            if not self._is_whisper_model(model_name):
                # Boolean admission: a model with running work waits a step.
                if budget.current_num_requests(model_name) > 0:
                    leftover_waiting_requests.extend(model_requests)
                    continue
                remaining = budget.remaining_request_budget(model_name)
                for request in model_requests:
                    if remaining <= 0:
                        leftover_waiting_requests.append(request)
                        continue
                    requests.append(request)
                    budget.add_num_requests(model_name, 1)
                    remaining -= 1
                continue

            # Capacity-based admission for Whisper models. The request cap is
            # `max_batch_size` minus running requests; the memory governor adds
            # a *window* budget (Phase 10-2: a request costs ceil(seconds/30)
            # windows, not one).
            remaining = budget.remaining_request_budget(model_name)
            window_budget = self._whisper_window_budget(model_name, model_requests)
            credits = self._whisper_credits.setdefault(model_name, LaneCredits())
            selected, rejected = plan_lane_admission(
                [(request, whisper_lane(request)) for request in model_requests],
                remaining,
                credits,
                window_costs={id(request): whisper_window_count(request) for request in model_requests},
                window_budget=window_budget,
            )
            for request in selected:
                requests.append(request)
                budget.add_num_requests(model_name, 1)
            leftover_waiting_requests.extend(rejected)

        # Phase 5-1: requests that did not fit this step stay in the waiting
        # queue. Returning the (popped) input deque here used to drop every
        # request above the model's `max_batch_size` on the floor — it never
        # ran, was never reported, and the caller hung until its deadline.
        return leftover_waiting_requests, SchedulerWaitingOutputs(
            requests=requests, ignored_requests=ignored_requests
        )

    # ------------------------------------------------------------------
    # AR multi-step decode admission (Phase 9-4)
    # ------------------------------------------------------------------
    def _is_ar_model(self, model_name: str | None) -> bool:
        """True when ``model_name`` is an AR family that decodes across steps."""
        return self._model_types.get(model_name) in AR_INFERENCE_MODEL_TYPES

    def _continues_across_steps(self, request: EngineRequest) -> bool:
        """True for a RUNNING AR row that returned a partial result last step.

        Only flagged requests are rescheduled: a request that has not yet
        produced a partial result is either new (it comes from ``waiting``) or
        already terminal (it was freed). This keeps the carry-over explicit and
        cannot accidentally re-run a stale non-AR request.
        """
        return (
            self._is_ar_model(getattr(request.request_data, "model", None))
            and not request.is_finished()
            and bool(getattr(request, "unfinished", False))
        )

    def _ar_capacity(self, model_name: str) -> dict | None:
        """Read the shared pool's capacity row for ``model_name``, or ``None``.

        ``None`` means "no capacity signal": the pool is disabled, the family
        has not registered an arena yet (9-3 registers it at model load), or the
        hook failed. Callers then keep the historical boolean rule, so a
        deployment without a live arena behaves exactly as before.

        The hook is called with the **scheduler's model name**
        (``ModelConfig.name`` / ``request_data.model``), so an AR runner must
        register its 9-2 arena under that key for capacity admission to apply;
        an arena under any other key is treated as "no signal".
        """
        hook = getattr(self, "ar_admission", None)
        if hook is None:
            return None
        try:
            decision = hook(model_name) or {}
        except KeyError:
            # No arena for this family yet (9-3). Not an error — fall back.
            return None
        except Exception as error:  # noqa: BLE001 - admission must never break scheduling
            logger.warning(
                f"AR capacity hook failed for '{model_name}' ({error}); using boolean admission"
            )
            return None
        if not decision.get("enabled"):
            return None
        try:
            slots = int(decision.get("slots") or 0)
            active = int(decision.get("active_rows", decision.get("slots_active", 0)) or 0)
            raw_free = decision.get("slots_free")
            slots_free = int(raw_free) if raw_free is not None else max(0, slots - active)
            rows_per_request = max(1, int(decision.get("rows_per_request", 1) or 1))
        except (TypeError, ValueError):  # pragma: no cover - defensive on odd hook output
            return None
        if slots <= 0:
            return None
        return {
            "slots": slots,
            "active_rows": active,
            "slots_free": max(0, slots_free),
            "rows_per_request": rows_per_request,
        }

    def _ar_admission_width(self, model_name: str, budget: SchedulingBudget) -> int:
        """Rows that may be newly admitted for an AR model this step.

        Composes the two independent signals (slots are not admission width):

        * the per-step request budget left after the continuing rows
          (``max_batch_size - running``), and
        * the pool's free **requests**: free slots divided by the contract's
          ``rows_per_request`` (2 for CFG, 1 for Turbo), so a 2-row request is
          never admitted when only one slot is free.

        With no capacity signal the historical boolean rule applies: a model
        that still has running work admits nothing, otherwise it fills its
        budget.
        """
        capacity = self._ar_capacity(model_name)
        if capacity is None:
            if budget.current_num_requests(model_name) > 0:
                return 0
            return budget.remaining_request_budget(model_name)
        rows_per_request = max(1, int(capacity.get("rows_per_request", 1) or 1))
        free_requests = int(capacity["slots_free"]) // rows_per_request
        return max(0, min(budget.remaining_request_budget(model_name), free_requests))

    # ------------------------------------------------------------------
    # Whisper lane admission (Phase 10-1)
    # ------------------------------------------------------------------
    def _is_whisper_model(self, model_name: str | None) -> bool:
        """True when ``model_name`` is a configured ``FasterWhisper`` entry."""
        return self._model_types.get(model_name) == "FasterWhisper"

    def _whisper_window_budget(self, model_name: str, candidates) -> int | None:
        """Window allowance for this step's new admissions, or ``None``.

        ``get_whisper_admission`` (Phase 10-3) returns the total number of
        Whisper windows that may be in flight. Phase 10-2 counts a request as
        the windows it will form (``ceil(seconds / 30)``), subtracts the windows
        already running, and returns the remaining allowance for new requests.

        A governor that is disabled (CPU, no resident Whisper, no measured cost)
        returns ``None`` so the caller falls back to plain capacity.

        **Progress floor.** When nothing is in flight and a request needs more
        windows than the current allowance (a long request against a small
        headroom), the budget is raised to that request's cost — otherwise its
        own size would starve it forever. This mirrors Phase 10-1's
        one-request-one-window behaviour for the degenerate case, while the
        governor's own ``min_windows`` floor still applies (``allowed == 0`` is
        a hard stop and is respected).
        """
        hook = getattr(self, "whisper_admission", None)
        if hook is None:
            return None

        running_windows = sum(
            whisper_window_count(request)
            for request in self.running
            if getattr(request.request_data, "model", None) == model_name
        )
        try:
            decision = hook(active_windows=running_windows) or {}
        except Exception as error:  # noqa: BLE001 - admission must never break scheduling
            logger.warning(f"Whisper admission governor failed ({error}); using capacity only")
            return None
        if not decision.get("enabled"):
            return None
        try:
            allowed = int(decision.get("allowed_windows") or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive on odd hook output
            return None

        headroom = max(0, allowed - running_windows)
        if running_windows == 0 and allowed > 0 and candidates:
            headroom = max(headroom, min(whisper_window_count(request) for request in candidates))
        return headroom

    def whisper_lane_depths(self) -> dict[str, dict[str, int]]:
        """Waiting-request depth per Whisper lane: ``{model: {"short", "long"}}``.

        Feeds the ``hse_whisper_lane_{short,long}_depth`` gauges. Only requests
        still waiting count: a running request is no longer queued behind
        anything.
        """
        depths: dict[str, dict[str, int]] = {}
        for request in self.waiting:
            model_name = getattr(request.request_data, "model", None)
            if not self._is_whisper_model(model_name):
                continue
            entry = depths.setdefault(model_name, {"short": 0, "long": 0})
            entry[whisper_lane(request)] += 1
        return depths

    def get_scheduling_budget(self) -> SchedulingBudget:
        """Calculate the overall scheduling Budget for requests per Model"""
        budget_count = {}
        per_model_requests = {}
        for model_cfg in self.model_configs:
            budget_count[model_cfg.name] = model_cfg.max_batch_size
            per_model_requests[model_cfg.name] = 0

        return SchedulingBudget(request_per_model_budget=budget_count, _num_per_model_requests=per_model_requests)

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        # Calculate the overall scheduling budget
        budget = self.get_scheduling_budget()
        # Include running requests to the budget.
        for request in self.running:
            budget.add_num_requests(request.request_data.model, 1)

        # Build mapping of ready requests on a per model dictionary.
        # Phase 9-4: an AR request that returned a partial result last step is
        # still RUNNING and must be scheduled again — one engine step is one
        # decode step, not one whole request. Non-AR families never have
        # unfinished running requests (every step completes them), so their
        # schedule is unchanged.
        per_model_scheduled: dict[str, list[EngineRequest]] = {}
        for request in self.running:
            if self._continues_across_steps(request):
                per_model_scheduled.setdefault(request.request_data.model, []).append(request)

        # Schedule waiting requests per each model's admission regime (boolean
        # for most families, lanes for Whisper, capacity for AR — see
        # `_schedule_waiting_requests`).
        remaining_waiting, ready = self._schedule_waiting_requests(self.waiting, budget)

        # Update waiting requests.
        self.waiting = remaining_waiting

        for r in ready.requests:
            # Set Request to running state
            r.set_running()
            # Fill into dict
            per_model_scheduled.setdefault(r.request_data.model, []).append(r)

        # Update running requests.
        self.running.extend(ready.requests)

        return SchedulerOutputs(
            scheduled_requests_per_model=per_model_scheduled, ignored_requests=ready.ignored_requests
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
