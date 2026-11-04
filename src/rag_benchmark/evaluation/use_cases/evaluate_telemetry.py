"""
Use case for evaluating operational telemetry, latency distributions, and throughput.
"""
from typing import Any, List, Optional

from rag_benchmark.evaluation.domain.schemas import TelemetryScorecard
from rag_benchmark.evaluation.infrastructure.factory import EvaluatorFactory


class EvaluateTelemetryUseCase:
    """
    Executes telemetry and operational SLA analysis over predictions and execution timings.
    Computes p50, p95, p99 latencies, averages, and QPS throughput.
    """

    def __init__(self, wall_clock_duration_sec: Optional[float] = None) -> None:
        self.wall_clock_duration_sec = wall_clock_duration_sec
        self.evaluator = EvaluatorFactory.create(
            "telemetry",
            wall_clock_duration_sec=wall_clock_duration_sec,
        )

    def execute(
        self,
        predictions: List[Any],
        wall_clock_duration_sec: Optional[float] = None,
    ) -> TelemetryScorecard:
        """
        Calculates operational performance percentiles from latency records.

        Args:
            predictions: List of prediction objects or dicts (containing latency_ms).
            wall_clock_duration_sec: Optional elapsed time for benchmark in seconds.

        Returns:
            TelemetryScorecard with percentiles and throughput.
        """
        duration = (
            wall_clock_duration_sec
            if wall_clock_duration_sec is not None
            else self.wall_clock_duration_sec
        )
        return self.evaluator.evaluate(
            predictions=predictions,
            wall_clock_duration_sec=duration,
        )
