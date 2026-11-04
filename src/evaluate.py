"""
CLI entry point for running the enterprise RAG evaluation benchmark.

Usage examples:
    # 1. Fast retrieval & ranking evaluation over 20 questions ($0 LLM cost)
    python -m src.evaluate --mode retrieval_only --sample-size 20

    # 2. Full end-to-end evaluation (retrieval + local generation + Gemini judge)
    python -m src.evaluate --mode full --sample-size 5 --output data/full_benchmark_report.json
"""
import argparse
import logging
import sys

from rag_benchmark.config import RAGConfig
from rag_benchmark.evaluation.pipeline import EvaluationPipeline
from rag_benchmark.pipeline import EnterpriseRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_benchmark.eval_cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enterprise RAG Multi-Pillar Benchmark Evaluation Runner"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="retrieval_only",
        choices=["full", "retrieval_only", "generation_only"],
        help="Evaluation pillar mode: 'retrieval_only' ($0 LLM cost), 'full', or 'generation_only'.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of benchmark test questions to evaluate (default: 10).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of hybrid candidates to retrieve from Qdrant (default: 50).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of candidates to evaluate after Cross-Encoder re-ranking (default: 10).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ingalepratap/EnterpriseRAG-Bench",
        help="HuggingFace benchmark dataset repository name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: 'test').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation_scorecard.json",
        help="File path to save the JSON evaluation report.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable Gemini LLM-as-a-judge for faithfulness evaluation.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini judge model identifier (default: 'gemini-2.0-flash').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Initializing Enterprise RAG Pipeline (SUT)...")
    config = RAGConfig()
    rag_pipeline = EnterpriseRAGPipeline(config=config)

    logger.info("Initializing Evaluation Orchestrator...")
    eval_pipeline = EvaluationPipeline(
        rag_pipeline=rag_pipeline,
        enable_llm_judge=not args.no_judge,
        judge_model=args.judge_model,
    )

    try:
        scorecard = eval_pipeline.run(
            samples=args.dataset,
            mode=args.mode,
            sample_size=args.sample_size,
            top_k=args.top_k,
            top_n=args.top_n,
            output_report_path=args.output,
        )
        logger.info("Evaluation completed successfully.")
        return scorecard
    finally:
        rag_pipeline.close()


if __name__ == "__main__":
    main()
