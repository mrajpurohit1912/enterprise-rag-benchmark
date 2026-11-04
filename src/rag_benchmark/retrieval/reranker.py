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
        max_passage_chars: int = 1500,
    ) -> List[Dict]:
        """
        Evaluates (Query, Title + Context) pairs using Cross-Encoder joint attention.
        Enriches candidate text with parent context to capture broader semantic nuances.
        """
        if not candidates:
            return []

        pairs = []
        for cand in candidates:
            title = cand.get("title", "").strip()
            # Prefer parent_text to give the CrossEncoder full section context; fallback to child_text
            passage = cand.get("parent_text") or cand.get("child_text", "")
            passage_snippet = passage[:max_passage_chars].strip()

            text = f"Title: {title}\n{passage_snippet}" if title else passage_snippet
            pairs.append([query, text])

        scores = self.model.predict(pairs)

        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        for final_rank, cand in enumerate(reranked[:top_n], start=1):
            cand["final_rank"] = final_rank

        return reranked[:top_n]
