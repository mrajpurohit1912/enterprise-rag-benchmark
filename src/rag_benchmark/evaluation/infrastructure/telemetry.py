"""
Telemetry and SRE operational metrics evaluator for the enterprise RAG pipeline.
Measures latency percentiles (p50, p95, p99), execution averages, and query throughput.
"""
from typing import Any, List, Optional
import numpy as np

from rag_benchmark.evaluation.domain.base import EvaluationBase
from rag_benchmark.evaluation.domain.schemas import (
    EvaluationSample,
    TelemetryScorecard,
)


class TelemetryEvaluator(EvaluationBase):
    """
    Evaluates system operational latency, tail behavior, and throughput.
    Computes industry-standard SLA metrics (p50, p95, p99) across pipeline phases.
    """

    def __init__(self, wall_clock_duration_sec: Optional[float] = None) -> None:
        self.wall_clock_duration_sec = wall_clock_duration_sec

    def evaluate(
        self,
        predictions: List[Any],
        ground_truth: Optional[List[EvaluationSample]] = None,
        wall_clock_duration_sec: Optional[float] = None,
    ) -> TelemetryScorecard:
        """
        Calculates operational performance percentiles from latency records.

        Args:
            predictions: List of prediction models or dicts containing 'latency_ms'.
            ground_truth: Optional ground-truth samples (not strictly required for telemetry).
            wall_clock_duration_sec: Optional total elapsed benchmark execution time in seconds.

        Returns:
            Populated TelemetryScorecard.
        """
        if not predictions:
            return TelemetryScorecard()

        duration = (
            wall_clock_duration_sec
            if wall_clock_duration_sec is not None
            else self.wall_clock_duration_sec
        )

        latencies: List[float] = []
        for p in predictions:
            if hasattr(p, "latency_ms"):
                lat = float(p.latency_ms)
            elif isinstance(p, dict):
                lat = float(p.get("latency_ms", 0.0))
            else:
                lat = 0.0
            latencies.append(lat)

        if not latencies:
            return TelemetryScorecard()

        lat_array = np.array(latencies, dtype=float)
        total_queries = len(latencies)

        p50 = float(np.percentile(lat_array, 50))
        p95 = float(np.percentile(lat_array, 95))
        p99 = float(np.percentile(lat_array, 99))
        avg_total = float(np.mean(lat_array))

        # Calculate Throughput: queries per second
        if duration and duration > 0:
            qps = total_queries / duration
        else:
            # Fallback estimation using cumulative execution time
            total_time_sec = float(np.sum(lat_array)) / 1000.0
            qps = total_queries / total_time_sec if total_time_sec > 0 else 0.0

        return TelemetryScorecard(
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            p99_latency_ms=round(p99, 2),
            avg_total_latency_ms=round(avg_total, 2),
            throughput_qps=round(qps, 2),
            total_evaluated=total_queries,
        )

    def get_metric_names(self) -> List[str]:
        return [
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "avg_total_latency_ms",
            "throughput_qps",
        ]


# Backward-compatible alias
Telemetry = TelemetryEvaluator
