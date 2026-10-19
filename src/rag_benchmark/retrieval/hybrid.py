"""
Hybrid Search Retriever combining Qdrant dense vectors with BM25 sparse search and RRF.
"""
from typing import Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag_benchmark.storage.vector_store import QdrantVectorStore
from rag_benchmark.models import ChildChunk


class HybridRetriever:
    """
    Coordinates Dense search (Qdrant) and Sparse search (BM25),
    fusing results with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunks: Optional[List[ChildChunk]] = None,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.rrf_k = rrf_k
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks_lookup: Dict[str, ChildChunk] = {}

        if chunks:
            self.build_bm25_index(chunks)

    def build_bm25_index(self, chunks: List[ChildChunk]) -> None:
        """
        Builds the in-memory BM25 inverted index for exact keyword matching.
        """
        self.chunks_lookup = {c.child_id: c for c in chunks}
        tokenized_corpus = [c.child_text.lower().split() for c in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 50,
        candidate_pool: int = 50,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Executes parallel Dense + BM25 search and merges candidates using Reciprocal Rank Fusion (RRF).
        """
        # 1. Dense Search via Qdrant
        query_vector = self.embedding_model.encode(
            query, normalize_embeddings=True
        ).tolist()
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_pool,
            source_type=source_type,
        )

        # 2. Sparse Search via BM25
        bm25_ranked_ids: List[str] = []
        if self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_pool]

            chunk_keys = list(self.chunks_lookup.keys())
            bm25_ranked_ids = [chunk_keys[i] for i in top_bm25_indices]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, Dict] = {}

        # Fuse Dense ranks
        for rank, item in enumerate(dense_results, start=1):
            c_id = item["child_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (self.rrf_k + rank))
            candidate_map[c_id] = item

        # Fuse BM25 ranks
        for rank, c_id in enumerate(bm25_ranked_ids, start=1):
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if c_id not in candidate_map and c_id in self.chunks_lookup:
                chunk = self.chunks_lookup[c_id]
                candidate_map[c_id] = {
                    "score": 0.0,
                    "child_id": chunk.child_id,
                    "parent_id": chunk.parent_id,
                    "doc_id": chunk.doc_id,
                    "source_type": chunk.source_type,
                    "title": chunk.title,
                    "child_text": chunk.child_text,
                    "parent_text": "",
                }

        # Sort by RRF score descending
        fused_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )[:top_k]

        fused_results = []
        for rank, c_id in enumerate(fused_ids, start=1):
            entry = candidate_map[c_id]
            entry["rank"] = rank
            entry["rrf_score"] = rrf_scores[c_id]
            fused_results.append(entry)

        return fused_results
