import json
import os
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv

from rag_benchmark.evaluation.domain.base import EvaluationBase
from rag_benchmark.evaluation.domain.schemas import (
    EvaluationSample,
    GenerationPrediction,
    GenerationScorecard,
)

# Load environment variables (supports .env.dev and .env)
load_dotenv(".env.dev")
load_dotenv(".env")


class GenerationEvaluator(EvaluationBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        judge_model: str = "gemini-2.0-flash",
        enable_llm_judge: bool = True,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("GOOGLE_GEMENI_API_KEY")
            or os.getenv("GOOGLE_GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        self.judge_model = judge_model
        self.enable_llm_judge = enable_llm_judge
        self._client = None

        if self.enable_llm_judge and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize Google GenAI Judge: {e}")

    def evaluate(
        self,
        predictions: List[GenerationPrediction],
        ground_truth: List[EvaluationSample],
    ) -> GenerationScorecard:
        if not predictions or not ground_truth:
            return GenerationScorecard()

        truth_map: Dict[str, EvaluationSample] = {
            s.question_id: s for s in ground_truth
        }

        total_queries = 0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        correct_refusals = 0
        refusal_eligible = 0
        faithfulness_scores: List[float] = []
        total_latency = 0.0

        for pred in predictions:
            sample = truth_map.get(pred.question_id)
            if not sample:
                continue

            total_queries += 1
            total_latency += getattr(pred, "latency_ms", 0.0)

            p, r, f1 = self.compute_citation_metrics(
                cited_ids=pred.cited_doc_ids,
                expected_ids=sample.expected_doc_ids,
            )

            total_precision += p
            total_recall += r
            total_f1 += f1

            is_refusal = "information not found" in pred.generated_answer.lower()
            if not sample.expected_doc_ids:
                refusal_eligible += 1
                if is_refusal:
                    correct_refusals += 1

            if self._client and pred.context_text and not is_refusal:
                faith_score = self.judge_faithfulness(
                    answer=pred.generated_answer,
                    context=pred.context_text,
                )
                if faith_score is not None:
                    faithfulness_scores.append(faith_score)

        avg_precision = total_precision / total_queries if total_queries else 0.0
        avg_recall = total_recall / total_queries if total_queries else 0.0
        avg_f1 = total_f1 / total_queries if total_queries else 0.0
        refusal_acc = (
            correct_refusals / refusal_eligible if refusal_eligible > 0 else 1.0
        )
        avg_sim = (
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else None
        )
        return GenerationScorecard(
            citation_precision=round(avg_precision, 4),
            citation_recall=round(avg_recall, 4),
            citation_f1=round(avg_f1, 4),
            refusal_accuracy=round(refusal_acc, 4),
            semantic_similarity=round(avg_sim, 4) if avg_sim is not None else None,
            avg_generation_latency_ms=round(total_latency / total_queries, 2)
            if total_queries
            else 0.0,
            total_queries=total_queries,
        )

    def judge_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> Optional[float]:
        if not self._client:
            return None

        prompt = f"""You are an enterprise AI evaluation judge.
Evaluate whether the following generated answer is strictly grounded in and supported by the provided context.

Context:
{context[:4000]}

Generated Answer:
{answer}

Respond ONLY with a JSON object in this exact format:
{{"faithfulness_score": <float between 0.0 and 1.0>, "reason": "<brief justification>"}}
"""
        try:
            response = self._client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
            )
            text = response.text.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return float(data.get("faithfulness_score", 1.0))
        except Exception as e:
            print(f"[Warning] LLM Judge call failed: {e}")
        return None

    def get_metric_names(self) -> List[str]:
        return [
            "citation_precision",
            "citation_recall",
            "citation_f1",
            "refusal_accuracy",
            "semantic_similarity",
            "avg_generation_latency_ms",
        ]

    @staticmethod
    def compute_citation_metrics(
        cited_ids: List[str],
        expected_ids: List[str],
    ) -> tuple[float, float, float]:
        if not expected_ids:
            return (1.0, 1.0, 1.0) if not cited_ids else (0.0, 1.0, 0.0)
        if not cited_ids:
            return 0.0, 0.0, 0.0

        cited_set = set(cited_ids)
        expected_set = set(expected_ids)
        tp = len(cited_set & expected_set)

        precision = tp / len(cited_set) if cited_set else 0.0
        recall = tp / len(expected_set) if expected_set else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1


# Backward-compatible alias
Generation = GenerationEvaluator

if __name__ == "__main__":
    evaluator = GenerationEvaluator()
    print(f"GenerationEvaluator initialized successfully (Judge client active: {evaluator._client is not None})")
    print(f"Supported metrics: {evaluator.get_metric_names()}")

