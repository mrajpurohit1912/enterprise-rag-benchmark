"""
Enterprise Evaluation Pipeline Orchestrator.
Coordinates dataset loading, retrieval evaluation, generation evaluation,
telemetry SLA analysis, and benchmark reporting.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from rag_benchmark.evaluation.domain.schemas import (
    EvaluationSample,
    FullEvaluationScorecard,
    GenerationScorecard,
    RetrievalScorecard,
    TelemetryScorecard,
)
from rag_benchmark.evaluation.use_cases.evaluate_generation import EvaluateGenerationUseCase
from rag_benchmark.evaluation.use_cases.evaluate_retrieval import EvaluateRetrievalUseCase
from rag_benchmark.evaluation.use_cases.evaluate_telemetry import EvaluateTelemetryUseCase

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Production-grade benchmark orchestrator for Enterprise RAG.
    Adheres to Clean Architecture: receives the System Under Test (SUT) as a dependency.
    """

    def __init__(
        self,
        rag_pipeline: Any,
        enable_llm_judge: bool = True,
        judge_model: str = "gemini-2.0-flash",
    ) -> None:
        """
        Initializes the Evaluation Pipeline with the target RAG system.

        Args:
            rag_pipeline: The instantiated enterprise RAG pipeline to be evaluated.
            enable_llm_judge: Whether to run Gemini LLM-as-a-judge for faithfulness.
            judge_model: Gemini model identifier for LLM-as-a-judge.
        """
        self.rag_pipeline = rag_pipeline
        self.enable_llm_judge = enable_llm_judge
        self.judge_model = judge_model

        # Initialize Use Cases
        self.retrieval_use_case = EvaluateRetrievalUseCase(rag_pipeline=self.rag_pipeline)
        self.generation_use_case = EvaluateGenerationUseCase(
            rag_pipeline=self.rag_pipeline,
            judge_model=self.judge_model,
            enable_llm_judge=self.enable_llm_judge,
        )
        self.telemetry_use_case = EvaluateTelemetryUseCase()

    @staticmethod
    def load_benchmark_samples(
        dataset_source: Optional[Union[str, List[EvaluationSample], List[Dict[str, Any]], Any]] = None,
        split: str = "test",
        sample_size: Optional[int] = None,
    ) -> List[EvaluationSample]:
        """
        Loads and normalizes benchmark evaluation samples from various sources:
        HuggingFace dataset names, loaded datasets, dict lists, or EvaluationSample lists.

        Args:
            dataset_source: Optional dataset name (str) or list/dataset object.
                            Defaults to 'ingalepratap/EnterpriseRAG-Bench' (questions config).
            split: Dataset split to evaluate (e.g. 'test').
            sample_size: Optional limit on the number of samples to evaluate.

        Returns:
            List of strongly-typed EvaluationSample instances.
        """
        if dataset_source is None or isinstance(dataset_source, str):
            repo_id = dataset_source 
            logger.info("Loading benchmark questions from HuggingFace dataset '%s' (split='%s')...", repo_id, split)
            from datasets import load_dataset

            hf_split = f"{split}[:{sample_size}]" if sample_size is not None else split
            raw_data = load_dataset(repo_id, "questions", split=hf_split)
            items = list(raw_data)
        elif isinstance(dataset_source, list):
            items = dataset_source[:sample_size] if sample_size is not None else dataset_source
        elif hasattr(dataset_source, "__iter__"):
            items = list(dataset_source)[:sample_size] if sample_size is not None else list(dataset_source)
        else:
            raise ValueError(f"Unsupported dataset source type: {type(dataset_source)}")

        samples: List[EvaluationSample] = []
        for item in items:
            if isinstance(item, EvaluationSample):
                samples.append(item)
            elif isinstance(item, dict):
                samples.append(EvaluationSample(**item))
            else:
                # Duck-type access
                samples.append(
                    EvaluationSample(
                        question_id=getattr(item, "question_id", str(len(samples))),
                        question=getattr(item, "question", ""),
                        expected_doc_ids=getattr(item, "expected_doc_ids", []),
                        gold_answer=getattr(item, "gold_answer", None),
                        answer_facts=getattr(item, "answer_facts", []),
                        source_types=getattr(item, "source_types", None),
                        question_type=getattr(item, "question_type", None),
                    )
                )

        logger.info("Successfully loaded %d evaluation samples.", len(samples))
        return samples

    def run(
        self,
        samples: Optional[Union[List[EvaluationSample], str]] = None,
        mode: str = "full",
        sample_size: Optional[int] = None,
        top_k: int = 50,
        top_n: int = 10,
        max_new_tokens: Optional[int] = None,
        output_report_path: Optional[str] = None,
    ) -> FullEvaluationScorecard:
        """
        Executes the benchmark evaluation across configured pillars.

        Args:
            samples: Preloaded EvaluationSample list or HuggingFace repo id (None loads default).
            mode: Evaluation mode: 'full', 'retrieval_only', or 'generation_only'.
            sample_size: Optional slice count for quick canary evaluations.
            top_k: Number of hybrid candidates to retrieve from Qdrant.
            top_n: Number of candidates to evaluate after Cross-Encoder re-ranking.
            max_new_tokens: Generation token budget.
            output_report_path: Optional file path to persist JSON scorecard report.

        Returns:
            FullEvaluationScorecard containing retrieval, generation, and telemetry metrics.
        """
        norm_samples = (
            samples
            if isinstance(samples, list) and samples and isinstance(samples[0], EvaluationSample)
            else self.load_benchmark_samples(samples, sample_size=sample_size)
        )

        if not norm_samples:
            raise ValueError("No evaluation samples provided or found.")

        logger.info(
            "Starting Enterprise RAG Benchmark | Mode: %s | Sample Count: %d | Top-K: %d | Top-N: %d",
            mode,
            len(norm_samples),
            top_k,
            top_n,
        )

        overall_start_time = time.perf_counter()

        retrieval_scorecard: Optional[RetrievalScorecard] = None
        generation_scorecard: Optional[GenerationScorecard] = None
        predictions_for_telemetry: List[Any] = []

        # 1. Retrieval & Re-Ranking Pillar
        if mode in ("full", "retrieval_only"):
            logger.info("--> [1/2] Executing Retrieval & Re-ranking Evaluation...")
            retrieval_scorecard, ret_predictions = self.retrieval_use_case.execute(
                samples=norm_samples,
                top_k=top_k,
                top_n=top_n,
            )
            predictions_for_telemetry = ret_predictions

        # 2. Generation & Grounding Pillar
        if mode in ("full", "generation_only"):
            logger.info("--> [2/2] Executing Grounded Generation & Judge Evaluation...")
            generation_scorecard, gen_predictions = self.generation_use_case.execute(
                samples=norm_samples,
                max_new_tokens=max_new_tokens,
            )
            # When evaluating generation, end-to-end latencies include retrieval + generation
            predictions_for_telemetry = gen_predictions

        total_wall_clock_sec = time.perf_counter() - overall_start_time

        # 3. Telemetry & SLA Pillar
        logger.info("--> Compiling Telemetry & Operational SLA Metrics...")
        telemetry_scorecard: TelemetryScorecard = self.telemetry_use_case.execute(
            predictions=predictions_for_telemetry,
            wall_clock_duration_sec=total_wall_clock_sec,
        )

        # Fallback empty retrieval scorecard if mode was generation_only
        if retrieval_scorecard is None:
            retrieval_scorecard = RetrievalScorecard()

        # Metadata extraction from RAG system
        metadata: Dict[str, Any] = {
            "mode": mode,
            "sample_count": len(norm_samples),
            "wall_clock_duration_sec": round(total_wall_clock_sec, 2),
            "top_k": top_k,
            "top_n": top_n,
            "llm_judge_active": self.enable_llm_judge,
            "judge_model": self.judge_model if self.enable_llm_judge else "None",
        }
        if hasattr(self.rag_pipeline, "config"):
            cfg = self.rag_pipeline.config
            metadata.update(
                {
                    "embedding_model": getattr(cfg, "embedding_model", "unknown"),
                    "reranker_model": getattr(cfg, "reranker_model", "unknown"),
                    "llm_model": getattr(cfg, "llm_model", "unknown"),
                    "collection_name": getattr(cfg, "collection_name", "unknown"),
                }
            )

        full_scorecard = FullEvaluationScorecard(
            retrieval=retrieval_scorecard,
            generation=generation_scorecard,
            telemetry=telemetry_scorecard,
            model_metadata=metadata,
        )

        # Print formatted report
        self.print_report(full_scorecard)

        # Persist report if requested
        if output_report_path:
            self.save_report(full_scorecard, output_report_path)

        return full_scorecard

    @staticmethod
    def print_report(scorecard: FullEvaluationScorecard) -> None:
        """
        Prints an enterprise-grade ASCII report summarising evaluation metrics.
        """
        ret = scorecard.retrieval
        gen = scorecard.generation
        tel = scorecard.telemetry
        meta = scorecard.model_metadata

        separator = "=" * 76
        sub_separator = "-" * 76

        print("\n" + separator)
        print("          ENTERPRISE RAG BENCHMARK EVALUATION SCORECARD")
        print(separator)
        print(f" Timestamp:  {scorecard.timestamp}")
        print(f" Samples:    {meta.get('sample_count', 0)} queries evaluated")
        print(f" Models:     Embedding={meta.get('embedding_model', 'N/A')} | Reranker={meta.get('reranker_model', 'N/A')}")
        print(f" LLM/Judge:  GenLLM={meta.get('llm_model', 'N/A')} | Judge={meta.get('judge_model', 'N/A')}")
        print(separator)

        # Retrieval Section
        print(" [1] RETRIEVAL & RE-RANKING METRICS")
        print(sub_separator)
        print(f"   * Hit Rate @ 1:              {ret.hit_rate_at_1 * 100:>6.2f} %")
        print(f"   * Hit Rate @ 3:              {ret.hit_rate_at_3 * 100:>6.2f} %")
        print(f"   * Hit Rate @ 5:              {ret.hit_rate_at_5 * 100:>6.2f} %")
        print(f"   * Hit Rate @ 10:             {ret.hit_rate_at_10 * 100:>6.2f} %")
        print(f"   * Hit Rate @ 50:             {ret.hit_rate_at_50 * 100:>6.2f} %")
        print(f"   * MRR (Mean Reciprocal Rank):{ret.mrr:>7.4f}")
        print(f"   * nDCG @ 10:                 {ret.ndcg_at_10:>7.4f}")
        print(f"   * Avg Retrieval Latency:     {ret.avg_retrieval_latency_ms:>7.2f} ms")

        # Generation Section (if present)
        if gen is not None:
            print("\n [2] GROUNDED GENERATION & CITATION METRICS")
            print(sub_separator)
            print(f"   * Citation Precision:        {gen.citation_precision * 100:>6.2f} %")
            print(f"   * Citation Recall:           {gen.citation_recall * 100:>6.2f} %")
            print(f"   * Citation F1 Score:         {gen.citation_f1 * 100:>6.2f} %")
            print(f"   * Refusal Accuracy:          {gen.refusal_accuracy * 100:>6.2f} %")
            if gen.semantic_similarity is not None:
                print(f"   * Judge Faithfulness:        {gen.semantic_similarity * 100:>6.2f} %")
            print(f"   * Avg Generation Latency:    {gen.avg_generation_latency_ms:>7.2f} ms")

        # Telemetry Section
        print("\n [3] OPERATIONAL SLA & TELEMETRY METRICS")
        print(sub_separator)
        print(f"   * Latency p50 (Median):      {tel.p50_latency_ms:>7.2f} ms")
        print(f"   * Latency p95:               {tel.p95_latency_ms:>7.2f} ms")
        print(f"   * Latency p99 (Tail SLA):    {tel.p99_latency_ms:>7.2f} ms")
        print(f"   * Latency Average:           {tel.avg_total_latency_ms:>7.2f} ms")
        print(f"   * System Throughput:         {tel.throughput_qps:>7.2f} QPS")
        print(f"   * Total Evaluated Requests:  {tel.total_evaluated:>7d}")
        print(separator + "\n")

    @staticmethod
    def save_report(scorecard: FullEvaluationScorecard, output_path: str) -> None:
        """
        Saves the scorecard report as a formatted JSON document.
        """
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(scorecard.model_dump_json(indent=2))

        logger.info("Evaluation scorecard successfully exported to '%s'.", output_path)
