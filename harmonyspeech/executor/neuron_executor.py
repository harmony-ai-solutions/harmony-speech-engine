from typing import Dict, List, Set

from harmonyspeech.executor.executor_base import ExecutorBase
from harmonyspeech.common.sequence import SamplerOutput, SequenceGroupMetadata


class NeuronExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        # Instantiate the worker and load the model to the device.
        self._init_worker()

    def _init_worker(self):
        from harmonyspeech.task_handler.neuron_worker import NeuronWorker

        self.driver_worker = NeuronWorker(
            self.model_config,
            self.parallel_config,
            self.device_config,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]],
                      num_lookahead_slots: int) -> List[SamplerOutput]:
        assert (blocks_to_swap_in == {} and blocks_to_swap_out == {}
                and blocks_to_copy == {}), (
                    "Cache operations are not supported for Neuron backend.")
        assert num_lookahead_slots == 0, (
            "lookahead not supported for Neuron backend.")

        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list)
        return output

    def check_health(self) -> None:
        # NeuronExecutor will always be healthy as long as
        # it's running.
        return
