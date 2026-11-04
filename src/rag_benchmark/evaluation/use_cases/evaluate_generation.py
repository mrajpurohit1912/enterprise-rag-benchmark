"""
Use case for evaluating end-to-end grounded generation quality and citations.
"""
import time
from typing import Any, List, Optional, Tuple

from rag_benchmark.evaluation.domain.schemas import (
    EvaluationSample,
    GenerationPrediction,
    GenerationScorecard,
)
from rag_benchmark.evaluation.infrastructure.factory import EvaluatorFactory


class EvaluateGenerationUseCase:
    """
    Executes end-to-end grounded RAG answering over benchmark samples.
    Evaluates citation accuracy, hallucination/refusal behavior, and faithfulness.
    """

    def __init__(
        self,
        rag_pipeline: Any,
        judge_model: str = "gemini-2.0-flash",
        enable_llm_judge: bool = True,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.evaluator = EvaluatorFactory.create(
            "generation",
            judge_model=judge_model,
            enable_llm_judge=enable_llm_judge,
        )

    def execute(
        self,
        samples: List[EvaluationSample],
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[GenerationScorecard, List[GenerationPrediction]]:
        """
        Runs generation evaluation across all benchmark samples.

        Args:
            samples: List of normalized evaluation test samples.
            max_new_tokens: Optional token generation budget.

        Returns:
            Tuple of (GenerationScorecard, List[GenerationPrediction])
        """
        predictions: List[GenerationPrediction] = []

        for sample in samples:
            start_time = time.perf_counter()

            # Execute full pipeline query (retrieval + rerank + LLM generation)
            result = self.rag_pipeline.query(
                query_text=sample.question,
                max_new_tokens=max_new_tokens,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            answer = result.get("answer", "")
            cited_ids = result.get("cited_doc_ids", [])
            context = result.get("context", "")
            status = result.get("metadata", {}).get("status", "success")

            predictions.append(
                GenerationPrediction(
                    question_id=sample.question_id,
                    generated_answer=answer,
                    cited_doc_ids=cited_ids,
                    context_text=context,
                    status=status,
                    latency_ms=round(elapsed_ms, 2),
                )
            )

        scorecard: GenerationScorecard = self.evaluator.evaluate(
            predictions=predictions,
            ground_truth=samples,
        )

        return scorecard, predictions
