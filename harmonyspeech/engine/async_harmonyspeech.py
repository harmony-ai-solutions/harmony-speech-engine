import asyncio
import time
from typing import Type, Optional, Dict, Union, Callable, Tuple, List, Set, AsyncIterator
from loguru import logger

from harmonyspeech import HarmonySpeechEngine
from harmonyspeech.common.config import ModelConfig
from harmonyspeech.common.outputs import TextToSpeechRequestOutput, RequestOutput
from harmonyspeech.endpoints.openai.protocol import GenerationOptions, AudioOutputOptions, VoiceConversionRequest, \
    TextToSpeechRequest


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github. Include your full error "
           "log after killing the process with Ctrl+C.")

    exception = None
    try:
        task.result()
        # NOTE: This will be thrown if task exits normally (which it should not)
        raise AsyncEngineDeadError(msg)
    except asyncio.exceptions.CancelledError:
        pass
    except KeyboardInterrupt:
        raise
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            msg + " See stack trace above for the actual cause.") from e


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(request_id)
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(rid)

    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def process_exception(self,
                          request_id: str,
                          exception: Exception,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info(f"Finished request {request_id}.")
        self.abort_request(request_id)

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


class _AsyncHarmonySpeech(HarmonySpeechEngine):
    """Extension of HarmonySpeechEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            # Execute the model.
            output = await self.model_executor.execute_model_async(
                seq_group_metadata_list, scheduler_outputs.blocks_to_swap_in,
                scheduler_outputs.blocks_to_swap_out,
                scheduler_outputs.blocks_to_copy,
                scheduler_outputs.num_lookahead_slots)
        else:
            output = []

        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups)

        # Log stats.
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs))

        return request_outputs

    async def add_request_async(
        self,
        request_id: str,
        request_data: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = await self.encode_request_async(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        return self.add_request(
            request_id,
                                prompt=prompt,
                                prompt_token_ids=prompt_token_ids,
                                sampling_params=sampling_params,
                                arrival_time=arrival_time,
                                lora_request=lora_request,
                                multi_modal_data=multi_modal_data)

    async def check_health_async(self) -> None:
        self.model_executor.check_health()


class AsyncHarmonySpeech:


    _engine_class: Type[_AsyncHarmonySpeech] = _AsyncHarmonySpeech

    def __init__(
        self,
        *args,
        log_requests: bool = True,
        max_log_len: int = 0,
        start_engine_loop: bool = True,
        **kwargs
    ) -> None:
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker: Optional[RequestTracker] = None
        self._errored_with: Optional[BaseException] = None

    async def add_text_to_speech_request(
        self,
        request_id: str,
        request_data: TextToSpeechRequest,
        arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_input = request_data.input
            if self.max_log_len is not None:
                if shortened_input is not None:
                    shortened_input = shortened_input[:self.max_log_len]
            logger.info(
                f"Received request {request_id}: "
                f"input: {shortened_input!r}."
            )

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        if arrival_time is None:
            arrival_time = time.time()

        stream = self._request_tracker.add_request(
            request_id,
            request_data=request_data,
            arrival_time=arrival_time
        )

        return stream

    async def generate_text_to_speech(
        self,
        request_id: str,
        request_data: TextToSpeechRequest,
    ) -> AsyncIterator[TextToSpeechRequestOutput]:
        """
        Generate outputs for a TTS request.

        Args:


        Yields:
            The output `RequestOutput` objects from the HarmonySpeechEngine for the
            request.
        """
        # Preprocess the request.
        # This should not be used for logging, as it is monotonic time.
        arrival_time = time.monotonic()

        try:
            stream = await self.add_text_to_speech_request(
                request_id=request_id,
                request_data=request_data,
                arrival_time=arrival_time,
            )

            async for request_output in stream:
                yield request_output
        except asyncio.exceptions.CancelledError:
            logger.info(f"Request {request_id} cancelled.")
            self._abort(request_id)
            raise
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id, verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the HarmonySpeech engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    async def do_log_stats(self) -> None:
        if self.engine_use_ray:
            await self.engine.do_log_stats.remote()
        else:
            self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        if self.engine_use_ray:
            try:
                await self.engine.check_health.remote()
            except ray.exceptions.RayActorError as e:
                raise RuntimeError("Engine is dead.") from e
        else:
            await self.engine.check_health_async()
        logger.debug(f"Health check took {time.perf_counter()-t}s")