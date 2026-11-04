"""
Use case for evaluating retrieval and re-ranking accuracy.
"""
import time
from typing import Any, List, Optional, Tuple

from rag_benchmark.data.normalizer import normalize_text
from rag_benchmark.evaluation.domain.schemas import (
    EvaluationSample,
    RetrievalPrediction,
    RetrievalScorecard,
)
from rag_benchmark.evaluation.infrastructure.factory import EvaluatorFactory


class EvaluateRetrievalUseCase:
    """
    Executes search and re-ranking over benchmark samples without LLM generation.
    Computes deterministic ranking metrics (HitRate@K, MRR, nDCG) with low latency.
    """

    def __init__(self, rag_pipeline: Any) -> None:
        self.rag_pipeline = rag_pipeline
        self.evaluator = EvaluatorFactory.create("retrieval")

    def execute(
        self,
        samples: List[EvaluationSample],
        top_k: int = 50,
        top_n: int = 10,
    ) -> Tuple[RetrievalScorecard, List[RetrievalPrediction]]:
        """
        Runs retrieval evaluation across all provided benchmark samples.

        Args:
            samples: List of normalized evaluation test samples.
            top_k: Number of hybrid search candidates to retrieve from Qdrant.
            top_n: Number of candidates to evaluate after Cross-Encoder re-ranking.

        Returns:
            Tuple of (RetrievalScorecard, List[RetrievalPrediction])
        """
        predictions: List[RetrievalPrediction] = []

        for sample in samples:
            clean_query = normalize_text(sample.question)

            start_time = time.perf_counter()

            # 1. Hybrid Search (Dense + Sparse with RRF in Qdrant)
            candidates = self.rag_pipeline.retriever.search(
                query=clean_query,
                top_k=top_k,
            )

            # 2. Cross-Encoder Re-Ranking
            if candidates and hasattr(self.rag_pipeline, "reranker"):
                reranked = self.rag_pipeline.reranker.rerank(
                    query=clean_query,
                    candidates=candidates,
                    top_n=top_n,
                )
            else:
                reranked = candidates[:top_n]

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract unique document IDs preserving rank order
            retrieved_doc_ids: List[str] = []
            seen_docs = set()
            for cand in reranked:
                doc_id = cand.get("doc_id", "")
                if doc_id and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    retrieved_doc_ids.append(doc_id)

            scores = [float(cand.get("rerank_score", cand.get("score", 0.0))) for cand in reranked]

            predictions.append(
                RetrievalPrediction(
                    question_id=sample.question_id,
                    retrieved_doc_ids=retrieved_doc_ids,
                    retrieved_scores=scores,
                    latency_ms=round(elapsed_ms, 2),
                )
            )

        scorecard: RetrievalScorecard = self.evaluator.evaluate(
            predictions=predictions,
            ground_truth=samples,
        )

        return scorecard, predictions
