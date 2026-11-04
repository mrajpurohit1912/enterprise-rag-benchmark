"""
Unit and integration tests for the enterprise EvaluationPipeline.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from rag_benchmark.evaluation.domain.schemas import (
    EvaluationSample,
    FullEvaluationScorecard,
)
from rag_benchmark.evaluation.pipeline import EvaluationPipeline


class DummyRAGPipeline:
    """Mock RAG Pipeline for deterministic unit testing of the Evaluation Pipeline."""

    def __init__(self):
        self.config = MagicMock()
        self.config.embedding_model = "test-embed"
        self.config.reranker_model = "test-reranker"
        self.config.llm_model = "test-llm"
        self.config.collection_name = "test-collection"

        self.retriever = MagicMock()
        # Mock retrieval returning documents with doc_ids
        self.retriever.search.return_value = [
            {"doc_id": "doc_alpha", "score": 0.85, "text": "Alpha text"},
            {"doc_id": "doc_beta", "score": 0.70, "text": "Beta text"},
            {"doc_id": "doc_gamma", "score": 0.50, "text": "Gamma text"},
        ]

        self.reranker = MagicMock()
        self.reranker.rerank.return_value = [
            {"doc_id": "doc_alpha", "rerank_score": 2.5, "text": "Alpha text"},
            {"doc_id": "doc_beta", "rerank_score": 1.1, "text": "Beta text"},
            {"doc_id": "doc_gamma", "rerank_score": -0.5, "text": "Gamma text"},
        ]

    def query(self, query_text: str, max_new_tokens: int = None):
        return {
            "query": query_text,
            "answer": "According to document alpha, the parameter is 42.",
            "cited_doc_ids": ["doc_alpha"],
            "context": "Context from doc alpha",
            "metadata": {"status": "success"},
        }


class TestEvaluationPipeline(unittest.TestCase):

    def setUp(self):
        self.dummy_pipeline = DummyRAGPipeline()
        self.pipeline = EvaluationPipeline(
            rag_pipeline=self.dummy_pipeline,
            enable_llm_judge=False,  # Disable Gemini judge in offline unit tests
        )
        self.samples = [
            EvaluationSample(
                question_id="q1",
                question="What is the alpha parameter?",
                expected_doc_ids=["doc_alpha"],
                gold_answer="The parameter is 42.",
            ),
            EvaluationSample(
                question_id="q2",
                question="What is the beta parameter?",
                expected_doc_ids=["doc_beta"],
                gold_answer="The parameter is 99.",
            ),
        ]

    def test_load_benchmark_samples_from_dicts(self):
        raw_items = [
            {
                "question_id": "q1",
                "question": "test q",
                "expected_doc_ids": ["d1"],
                "source_types": ["github"],
            }
        ]
        samples = EvaluationPipeline.load_benchmark_samples(raw_items)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].question_id, "q1")
        self.assertEqual(samples[0].source_types, ["github"])

    def test_run_retrieval_only_mode(self):
        scorecard = self.pipeline.run(
            samples=self.samples,
            mode="retrieval_only",
            top_k=10,
            top_n=3,
        )

        self.assertIsInstance(scorecard, FullEvaluationScorecard)
        self.assertIsNotNone(scorecard.retrieval)
        self.assertIsNone(scorecard.generation)
        self.assertIsNotNone(scorecard.telemetry)

        # q1 expects doc_alpha (rank 1), q2 expects doc_beta (rank 2)
        # HitRate@1: q1 hits -> 0.5
        # HitRate@3: both hit -> 1.0
        self.assertAlmostEqual(scorecard.retrieval.hit_rate_at_1, 0.5, places=2)
        self.assertAlmostEqual(scorecard.retrieval.hit_rate_at_3, 1.0, places=2)
        self.assertGreater(scorecard.retrieval.mrr, 0.0)
        self.assertEqual(scorecard.telemetry.total_evaluated, 2)
        self.assertGreater(scorecard.telemetry.throughput_qps, 0.0)

    def test_run_full_mode_and_export_report(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            scorecard = self.pipeline.run(
                samples=self.samples,
                mode="full",
                top_k=10,
                top_n=3,
                output_report_path=tmp_path,
            )

            self.assertIsNotNone(scorecard.retrieval)
            self.assertIsNotNone(scorecard.generation)
            self.assertIsNotNone(scorecard.telemetry)

            # Both samples returned citation 'doc_alpha'
            # q1 expects ['doc_alpha'] -> precision=1, recall=1
            # q2 expects ['doc_beta'] -> precision=0, recall=0
            # Mean citation precision = 0.5
            self.assertAlmostEqual(scorecard.generation.citation_precision, 0.5, places=2)

            # Verify saved report
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("p50_latency_ms", content)
                self.assertIn("hit_rate_at_1", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
