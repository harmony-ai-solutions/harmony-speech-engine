import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from loguru import logger
from prometheus_client import (REGISTRY, Counter, Gauge, Histogram, Info,
                               disable_created_metrics)

disable_created_metrics()

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.


# begin-metrics-definitions
class Metrics:

    def __init__(self, labelnames: List[str]):
        # Unregister any existing Aphrodite collectors
        for collector in list(REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "aphrodite" in collector._name:
                REGISTRY.unregister(collector)

        # Config Information
        self.info_cache_config = Info(
            name="aphrodite:cache_config",
            documentation="information of cache_config",
        )

        # System stats
        self.gauge_scheduler_running = Gauge(
            name="aphrodite:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames,
        )
        self.gauge_scheduler_swapped = Gauge(
            name="aphrodite:num_requests_swapped",
            documentation="Number of requests swapped to CPU.",
            labelnames=labelnames,
        )
        self.gauge_scheduler_waiting = Gauge(
            name="aphrodite:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames,
        )
        self.gauge_gpu_cache_usage = Gauge(
            name="aphrodite:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
        )
        self.gauge_cpu_cache_usage = Gauge(
            name="aphrodite:cpu_cache_usage_perc",
            documentation="CPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
        )

        # Raw stats from last model iteration
        self.counter_prompt_tokens = Counter(
            name="aphrodite:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
        )
        self.counter_generation_tokens = Counter(
            name="aphrodite:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
        )
        self.histogram_time_to_first_token = Histogram(
            name="aphrodite:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.001,
                0.005,
                0.01,
                0.02,
                0.04,
                0.06,
                0.08,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
            ],
        )
        self.histogram_time_per_output_token = Histogram(
            name="aphrodite:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
                2.5,
            ],
        )
        self.histogram_e2e_request_latency = Histogram(
            name="aphrodite:e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )

        # Legacy metrics
        self.gauge_avg_prompt_throughput = Gauge(
            name="aphrodite:avg_prompt_throughput_toks_per_s",
            documentation="Average prefill throughput in tokens/s.",
            labelnames=labelnames,
        )
        self.gauge_avg_generation_throughput = Gauge(
            name="aphrodite:avg_generation_throughput_toks_per_s",
            documentation="Average generation throughput in tokens/s.",
            labelnames=labelnames,
        )


# end-metrics-definitions


@dataclass
class Stats:
    """Created by AphroditeEngine for use by StatLogger."""

    now: float

    # System stats.
    num_running: int
    num_waiting: int
    # num_swapped: int
    # gpu_cache_usage: float
    # cpu_cache_usage: float

    # Raw stats from last model iteration.
    # num_prompt_tokens: int
    # num_generation_tokens: int
    # time_to_first_tokens: List[float]
    # time_per_output_tokens: List[float]
    # time_e2e_requests: List[float]
    #
    # spec_decode_metrics: Optional["SpecDecodeWorkerMetrics"] = None


class StatLogger:
    """StatLogger is used AphroditeEngine to log to Promethus and Stdout."""

    def __init__(self, local_interval: float, labels: Dict[str, str]) -> None:
        # Metadata for logging locally.
        self.last_local_log = time.monotonic()
        self.local_interval = local_interval

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

        # Prometheus metrics
        self.labels = labels
        self.metrics = Metrics(labelnames=list(labels.keys()))

    def info(self, type: str, obj: object) -> None:
        if type == "cache_config":
            self.metrics.info_cache_config.info(obj.metrics_info())

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        return float(np.sum(tracked_stats) / (now - self.last_local_log))

    def _local_interval_elapsed(self, now: float) -> bool:
        elapsed_time = now - self.last_local_log
        return elapsed_time > self.local_interval

    def log(self, stats: Stats) -> None:
        """Called by AphroditeEngine.
        Logs to Stdout every self.local_interval seconds."""

        # Log locally every local_interval seconds.
        if self._local_interval_elapsed(stats.now):
            # Compute summary metrics for tracked stats (and log them to
            # prometheus if applicable).

            # Log to stdout.
            logger.info(
                f"Running: {stats.num_running} reqs, "
                f"Pending: {stats.num_waiting} reqs"
            )

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now
