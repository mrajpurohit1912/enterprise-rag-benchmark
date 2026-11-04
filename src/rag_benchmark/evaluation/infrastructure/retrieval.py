import math
from typing import Any, Dict, List

from rag_benchmark.evaluation.domain.base import EvaluationBase
from rag_benchmark.evaluation.domain.schemas import (
    EvaluationSample,
    RetrievalPrediction,
    RetrievalScorecard,
)


class RetrievalEvaluator(EvaluationBase):
    """
    Evaluates Information Retrieval and Re-Ranking quality.
    Computes deterministic ranking metrics (Hit Rate @ K, MRR, nDCG)
    comparing retrieved candidates against expected ground-truth document IDs.
    """

    def __init__(self, k_values: List[int] = None) -> None:
        self.k_values = k_values or [1, 3, 5, 10, 50]

    def evaluate(
        self,
        predictions: List[RetrievalPrediction],
        ground_truth: List[EvaluationSample],
    ) -> RetrievalScorecard:
        """
        Executes IR metrics over all query predictions against ground-truth samples.
        Returns an aggregated RetrievalScorecard.
        """
        if not predictions or not ground_truth:
            return RetrievalScorecard()

        # Map ground-truth samples by question_id for fast O(1) alignment
        truth_map: Dict[str, EvaluationSample] = {
            sample.question_id: sample for sample in ground_truth
        }

        total_queries = 0
        hits_at_k: Dict[int, float] = {k: 0.0 for k in self.k_values}
        mrr_sum = 0.0
        ndcg_sum = 0.0
        total_latency = 0.0

        for pred in predictions:
            sample = truth_map.get(pred.question_id)
            if not sample:
                continue

            total_queries += 1
            retrieved_ids = pred.retrieved_doc_ids
            expected_ids = sample.expected_doc_ids

            # Compute Hit Rate @ K
            for k in self.k_values:
                hits_at_k[k] += self.compute_hit_at_k(retrieved_ids, expected_ids, k=k)

            # Compute MRR
            mrr_sum += self.compute_reciprocal_rank(retrieved_ids, expected_ids)

            # Compute nDCG @ 10
            ndcg_sum += self.compute_ndcg_at_k(retrieved_ids, expected_ids, k=10)

            # Track retrieval latency
            total_latency += getattr(pred, "latency_ms", 0.0)

        if total_queries == 0:
            return RetrievalScorecard()

        return RetrievalScorecard(
            hit_rate_at_1=round(hits_at_k.get(1, 0.0) / total_queries, 4),
            hit_rate_at_3=round(hits_at_k.get(3, 0.0) / total_queries, 4),
            hit_rate_at_5=round(hits_at_k.get(5, 0.0) / total_queries, 4),
            hit_rate_at_10=round(hits_at_k.get(10, 0.0) / total_queries, 4),
            hit_rate_at_50=round(hits_at_k.get(50, 0.0) / total_queries, 4),
            mrr=round(mrr_sum / total_queries, 4),
            ndcg_at_10=round(ndcg_sum / total_queries, 4),
            avg_retrieval_latency_ms=round(total_latency / total_queries, 2),
            total_queries=total_queries,
        )

    def get_metric_names(self) -> List[str]:
        """Returns the list of metric names computed by this evaluator."""
        return [
            "hit_rate_at_1",
            "hit_rate_at_3",
            "hit_rate_at_5",
            "hit_rate_at_10",
            "hit_rate_at_50",
            "mrr",
            "ndcg_at_10",
            "avg_retrieval_latency_ms",
        ]

    @staticmethod
    def compute_hit_at_k(
        retrieved_ids: List[str],
        expected_ids: List[str],
        k: int,
    ) -> float:
        """Checks if at least one expected document appears in the top K."""
        if not retrieved_ids or not expected_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        return 1.0 if any(doc in top_k for doc in expected_ids) else 0.0

    @staticmethod
    def compute_reciprocal_rank(
        retrieved_ids: List[str],
        expected_ids: List[str],
    ) -> float:
        """Computes 1 / rank of the first relevant document."""
        if not retrieved_ids or not expected_ids:
            return 0.0
        expected_set = set(expected_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_ndcg_at_k(
        retrieved_ids: List[str],
        expected_ids: List[str],
        k: int = 10,
    ) -> float:
        """Computes Normalized Discounted Cumulative Gain at K."""
        if not retrieved_ids or not expected_ids:
            return 0.0

        expected_set = set(expected_ids)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in expected_set:
                dcg += 1.0 / math.log2(i + 1)

        idcg = sum(
            1.0 / math.log2(i + 1)
            for i in range(1, min(len(expected_ids), k) + 1)
        )
        return (dcg / idcg) if idcg > 0.0 else 0.0


# Backward-compatible alias
Retrieval = RetrievalEvaluator



