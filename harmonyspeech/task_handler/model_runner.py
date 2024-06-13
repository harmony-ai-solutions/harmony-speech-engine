import contextlib
import time
from enum import IntEnum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from harmonyspeech.attention import (
    AttentionMetadata,
    AttentionMetadataPerStage,
    get_attn_backend,
)
from harmonyspeech.common.config import (
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
)
from harmonyspeech.common.logger import get_loading_progress_bar
from harmonyspeech.common.sampling_params import SamplingParams, SamplingType
from harmonyspeech.common.sequence import (
    MultiModalData,
    SamplerOutput,
    SequenceData,
    SequenceGroupMetadata,
)
from harmonyspeech.common.utils import (
    CudaMemoryProfiler,
    async_tensor_h2d,
    is_hip,
    is_pin_memory_available,
    make_tensor_with_pad,
    maybe_expand_dim,
)
from harmonyspeech.distributed import (
    broadcast_tensor_dict,
    get_tensor_model_parallel_world_size,
    with_pynccl_for_all_reduce,
)
from harmonyspeech.distributed.device_communicators import (
    custom_all_reduce,
    pynccl_utils,
)
from harmonyspeech.modeling import SamplingMetadata
from harmonyspeech.modeling.loader import get_model
from harmonyspeech.modeling.sampling_metadata import PersistentMetadata

_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]

# How batches are constructed.
class BatchType(IntEnum):
    # Every batch is prefill.
    PREFILL = 0
    # Every batch is decode.
    DECODE = 1
    # Batch is a mixture of prefill and decode.
    MIXED = 2


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        is_driver_worker: bool = False,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.is_driver_worker = is_driver_worker

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME: This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        self.model = None
        self.block_size = None  # Set after initial profiling.
        self.lora_manager = None

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = None  # Set after initial profiling.
        self.pin_memory = is_pin_memory_available()

        self.attn_backend = get_attn_backend(
            self.model_config.dtype if model_config is not None else None)

    def load_model(self) -> None:
        with CudaMemoryProfiler() as m:
            self.model = get_model(
                self.model_config,
                self.device_config,
                parallel_config=self.parallel_config
            )

        self.model_memory_usage = m.consumed_memory
        tp = get_tensor_model_parallel_world_size()
        logger.info(
            "Model weights loaded. Memory usage: "
            f"{self.model_memory_usage / float(2**30):.2f} GiB x {tp} = "
            f"{self.model_memory_usage * tp / float(2**30):.2f} GiB")

     @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)


        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not sampling_metadata.perform_sampling:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output

    # @torch.inference_mode()
    # def profile_run(self) -> None:
    #     # Enable top-k sampling to reflect the accurate memory usage.
    #     sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
    #     max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
    #     max_num_seqs = self.scheduler_config.max_num_seqs
    #
    #     # This represents the maximum number of different requests
    #     # that will have unique loras, an therefore the max amount of memory
    #     # consumption create dummy lora request copies from the lora request
    #     # passed in, which contains a lora from the lora warmup path.
    #     dummy_lora_requests = []
    #     dummy_lora_requests_per_seq = []
    #     if self.lora_config:
    #         for idx in range(self.lora_config.max_loras):
    #             lora_id = idx + 1
    #             dummy_lora_request = LoRARequest(
    #                 lora_name=f"warmup_{lora_id}",
    #                 lora_int_id=lora_id,
    #                 lora_local_path="/not/a/real/path",
    #             )
    #             self.lora_manager.add_dummy_lora(dummy_lora_request,
    #                                              rank=LORA_WARMUP_RANK)
    #             dummy_lora_requests.append(dummy_lora_request)
    #         dummy_lora_requests_per_seq = [
    #             dummy_lora_requests[idx % len(dummy_lora_requests)]
    #             for idx in range(max_num_seqs)
    #         ]
    #
    #     # Profile memory usage with max_num_sequences sequences and the total
    #     # number of tokens equal to max_num_batched_tokens.
    #     seqs: List[SequenceGroupMetadata] = []
    #     # Additional GPU memory may be needed for vision encoding, which needs
    #     # to be accounted for when calculating the GPU blocks for
    #     # Aphrodite blocker manager.
    #     # To exercise the worst scenario for GPU memory consumption,
    #     # the number of seqs (batch_size) is chosen to maximize the number
    #     # of images processed.
    #     if self.vision_language_config:
    #         max_num_seqs = min(
    #             max_num_seqs,
    #             int(max_num_batched_tokens /
    #                 self.vision_language_config.image_feature_size))
    #     for group_id in range(max_num_seqs):
    #         seq_len = (max_num_batched_tokens // max_num_seqs +
    #                    (group_id < max_num_batched_tokens % max_num_seqs))
    #         seq_data, fake_multi_modal_input = _prepare_fake_inputs(
    #             seq_len, self.vision_language_config)
    #         seq = SequenceGroupMetadata(
    #             request_id=str(group_id),
    #             is_prompt=True,
    #             seq_data={group_id: seq_data},
    #             sampling_params=sampling_params,
    #             block_tables=None,
    #             persistent_data={},
    #             lora_request=dummy_lora_requests_per_seq[group_id]
    #             if dummy_lora_requests_per_seq else None,
    #             multi_modal_data=fake_multi_modal_input,
    #         )
    #         seqs.append(seq)
    #
    #     # Run the model with the dummy inputs.
    #     num_layers = self.model_config.get_num_layers(self.parallel_config)
    #     kv_caches = [None] * num_layers
    #     self.execute_model(seqs, kv_caches)
    #     torch.cuda.synchronize()
    #     return
    #
    # @torch.inference_mode()
    # def capture_model(self, kv_caches: List[torch.Tensor]) -> None:
    #     """Cuda graph capture a model.
    #
    #     Note that CUDA graph's performance gain is negligible if number
    #     of batched tokens are larger than 200. And since CUDA graph
    #     requires fixed sized tensors, supporting large/variable batch
    #     size requires high GPU memory overhead. Thus, Aphrodite only captures
    #     decoding requests. Mixed batch (chunked prefill + decoding) or
    #     prefill requests are not captured.
    #
    #     Since it is used for decoding-only, it assumes there's only 1 token
    #     per sequence in the batch.
    #     """
    #     # NOTE: This is a hack to ensure that the NCCL backend is never
    #     # deleted before the CUDA graphs.
    #     self.pynccl_backend = pynccl_utils.get_nccl_backend()
    #
    #     assert not self.model_config.enforce_eager
    #     logger.info("Capturing the model for CUDA graphs. This may lead to "
    #                 "unexpected consequences if the model is not static. To "
    #                 "run the model in eager mode, set 'enforce_eager=True' or "
    #                 "use '--enforce-eager' in the CLI.")
    #     logger.warning("CUDA graphs can take additional 1~3 GiB of memory "
    #                    "per GPU. If you are running out of memory, consider "
    #                    "decreasing `gpu_memory_utilization` or enforcing "
    #                    "eager mode.")
    #     start_time = time.perf_counter()
    #
    #     # Prepare dummy inputs. These will be reused for all batch sizes.
    #     max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
    #     input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
    #     input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
    #     slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
    #     slot_mapping.fill_(_PAD_SLOT_ID)
    #     context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
    #     block_tables = torch.from_numpy(self.graph_block_tables).cuda()
    #
    #     graph_batch_size = _get_graph_batch_size(
    #         self.scheduler_config.max_num_seqs)
    #     batch_size_capture_list = [
    #         bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
    #     ]
    #
    #     # NOTE: There are 3 backends for all-reduce: custom all-reduce
    #     # kernel, PyNCCL, and PyTorch NCCL. When using CUDA graph, we use
    #     # either custom all-reduce kernel or PyNCCL. When not using CUDA
    #     # graph, we use either custom all-reduce kernel or PyTorch NCCL.
    #     # We always prioritize using custom all-reduce kernel but fall back
    #     # to PyTorch or PyNCCL if it is disabled or not supported.
    #     # Initialize a new progress bar
    #     progress = get_loading_progress_bar()
    #     task = progress.add_task("[cyan]Capturing graph...",
    #                              total=len(batch_size_capture_list))
    #
    #     with progress, custom_all_reduce.capture():
    #         for batch_size in reversed(batch_size_capture_list):
    #             # Create dummy attn_metadata.
    #             decode_metadata = self.attn_backend.make_metadata(
    #                 is_prompt=False,
    #                 prompt_lens=None,
    #                 prompt_lens_tensor=None,
    #                 max_subquery_len=None,
    #                 max_context_len=self.max_context_len_to_capture,
    #                 max_prompt_len=None,
    #                 subquery_start_loc=None,
    #                 seq_start_loc=None,
    #                 context_lens=context_lens[:batch_size],
    #                 block_tables=block_tables[:batch_size],
    #                 use_cuda_graph=True,
    #             )
    #             attn_metadata = AttentionMetadata(
    #                 num_prefills=0,
    #                 num_prefill_tokens=0,
    #                 num_decode_tokens=batch_size,
    #                 slot_mapping=slot_mapping[:batch_size],
    #                 prefill_metadata=None,
    #                 decode_metadata=decode_metadata,
    #                 kv_cache_dtype=self.kv_cache_dtype,
    #             )
    #
    #             if self.lora_config:
    #                 lora_mapping = LoRAMapping(
    #                     [0] * batch_size,
    #                     [0] * batch_size,
    #                 )
    #                 self.set_active_loras(set(), lora_mapping)
    #
    #             graph_runner = CUDAGraphRunner(self.model)
    #             graph_runner.capture(
    #                 input_tokens[:batch_size],
    #                 input_positions[:batch_size],
    #                 kv_caches,
    #                 attn_metadata,
    #                 memory_pool=self.graph_memory_pool,
    #             )
    #             self.graph_memory_pool = graph_runner.graph.pool()
    #             self.graph_runners[batch_size] = graph_runner
    #             # Update the progress bar
    #             progress.update(task, advance=1)
    #     end_time = time.perf_counter()
    #     elapsed_time = end_time - start_time
    #     # This usually takes < 10 seconds.
    #     logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")
    #
    # def __del__(self) -> None:
    #     # Delete the CUDA graphs before deleting the pynccl communicator.
    #     # NOTE: This is necessary because otherwise deadlocks can
    #     # happen.
    #     # FIXME: This is a bit hacky. Find a more robust solution.
    #     # TODO: when we get enough user feedback that pynccl is
    #     # more stable than cupy, we can remove this
    #     self.graph_runners.clear()
    #     self.pynccl_backend = None


# class CUDAGraphRunner:
#
#     def __init__(self, model: nn.Module):
#         self.model = model
#         self.graph = None
#         self.input_buffers: Dict[str, torch.Tensor] = {}
#         self.output_buffers: Dict[str, torch.Tensor] = {}
#
#     def capture(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         kv_caches: List[torch.Tensor],
#         attn_metadata: AttentionMetadata,
#         memory_pool,
#         **kwargs,
#     ) -> None:
#         assert self.graph is None
#         # Run the model once without capturing the graph.
#         # This is to make sure that the captured graph does not include the
#         # kernel launches for initial benchmarking (e.g., Triton autotune).
#         with _maybe_pynccl():
#             self.model(
#                 input_ids,
#                 positions,
#                 kv_caches,
#                 attn_metadata,
#                 **kwargs,
#             )
#         torch.cuda.synchronize()
#
#         # Capture the graph.
#         # NOTE: Python 3.8 does not support multi-line with statements.
#         # https://stackoverflow.com/questions/31039022/python-multi-line-with-statement
#         self.graph = torch.cuda.CUDAGraph()
#         with torch.cuda.graph(self.graph, pool=memory_pool):  # noqa: SIM117
#             with _maybe_pynccl():
#                 hidden_states = self.model(
#                     input_ids,
#                     positions,
#                     kv_caches,
#                     attn_metadata,
#                     **kwargs,
#                 )
#         torch.cuda.synchronize()
#
#         # Save the input and output buffers.
#         self.input_buffers = {
#             "input_ids": input_ids,
#             "positions": positions,
#             "kv_caches": kv_caches,
#             "slot_mapping": attn_metadata.slot_mapping,
#             "context_lens": attn_metadata.decode_metadata.context_lens,
#             "block_tables": attn_metadata.decode_metadata.block_tables,
#         }
#         self.output_buffers = {"hidden_states": hidden_states}
#         return
#
#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         kv_caches: List[torch.Tensor],
#         attn_metadata: AttentionMetadata,
#         **kwargs,
#     ) -> torch.Tensor:
#         # KV caches are fixed tensors, so we don't need to copy them.
#         del kv_caches
#
#         # Copy the input tensors to the input buffers.
#         self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
#         self.input_buffers["positions"].copy_(positions, non_blocking=True)
#         self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
#                                                  non_blocking=True)
#         self.input_buffers["context_lens"].copy_(
#             attn_metadata.decode_metadata.context_lens, non_blocking=True)
#         self.input_buffers["block_tables"].copy_(
#             attn_metadata.decode_metadata.block_tables, non_blocking=True)
#         # Run the graph.
#         self.graph.replay()
#
#         # Return the output tensor.
#         return self.output_buffers["hidden_states"]
#
#     def __call__(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)


@contextlib.contextmanager
def _maybe_pynccl():
    if pynccl_utils.is_initialized(
    ) and not custom_all_reduce.is_initialized():
        with with_pynccl_for_all_reduce():
            yield
    else:
        yield


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


# def _prepare_fake_inputs(
#         seq_len: int, vision_language_config: Optional[VisionLanguageConfig]):
#     """Prepare fake inputs for profile run."""
#     if vision_language_config:
#         prompt_tokens = [
#             vision_language_config.image_token_id
#         ] * vision_language_config.image_feature_size + [0] * (
#             seq_len - vision_language_config.image_feature_size)
#         fake_image_input = MultiModalData(
#             type=MultiModalData.Type.IMAGE,
#             data=torch.zeros(vision_language_config.image_input_shape,
#                              dtype=torch.float16))
#     else:
#         prompt_tokens = [0] * seq_len
#         fake_image_input = None
#     return SequenceData(prompt_tokens), fake_image_input
