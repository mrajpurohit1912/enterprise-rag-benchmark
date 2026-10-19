"""
Cross-Encoder Re-Ranking module for high-precision retrieval filtering.
"""
from typing import Dict, List
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks candidate documents using full joint cross-attention.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_n: int = 3,
    ) -> List[Dict]:
        """
        Evaluates (Query, Title + ChildText) pairs and reorders by relevance logits.
        """
        if not candidates:
            return []

        pairs = [
            [query, f"{cand.get('title', '')}\n{cand.get('child_text', '')}"]
            for cand in candidates
        ]

        scores = self.model.predict(pairs)

        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        for final_rank, cand in enumerate(reranked[:top_n], start=1):
            cand["final_rank"] = final_rank

        return reranked[:top_n]
