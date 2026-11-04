"""
Domain schemas for the enterprise RAG evaluation framework.
Provides strongly-typed contracts for benchmark samples, predictions, and scorecards.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """Normalized evaluation test case from benchmark dataset."""
    question_id: str
    question: str
    expected_doc_ids: List[str] = Field(default_factory=list)
    gold_answer: Optional[str] = None
    answer_facts: Optional[List[str]] = Field(default_factory=list)
    source_types: Optional[Union[str, List[str]]] = None
    question_type: Optional[str] = None


class RetrievalPrediction(BaseModel):
    """Output from the retrieval and ranking phase for a single query."""
    question_id: str
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    retrieved_scores: List[float] = Field(default_factory=list)
    latency_ms: float = 0.0


class GenerationPrediction(BaseModel):
    """Output from the LLM generation phase for a single query."""
    question_id: str
    generated_answer: str
    cited_doc_ids: List[str] = Field(default_factory=list)
    context_text: str = ""
    status: str = "success"
    latency_ms: float = 0.0


class RetrievalScorecard(BaseModel):
    """Metrics assessing retrieval and re-ranking quality."""
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    hit_rate_at_50: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    total_queries: int = 0


class GenerationScorecard(BaseModel):
    """Metrics assessing LLM answer accuracy and grounding."""
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0
    refusal_accuracy: float = 0.0
    semantic_similarity: Optional[float] = None
    avg_generation_latency_ms: float = 0.0
    total_queries: int = 0


class TelemetryScorecard(BaseModel):
    """SRE and operational performance metrics."""
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    total_evaluated: int = 0


class FullEvaluationScorecard(BaseModel):
    """Comprehensive multi-pillar evaluation report."""
    retrieval: RetrievalScorecard
    generation: Optional[GenerationScorecard] = None
    telemetry: TelemetryScorecard
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
