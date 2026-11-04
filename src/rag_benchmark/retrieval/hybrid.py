"""
Hybrid Search Retriever combining Qdrant dense vectors with Qdrant sparse vectors and native RRF.
"""
from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer

from rag_benchmark.storage.vector_store import QdrantVectorStore


class HybridRetriever:
    """
    Coordinates Dense search and Sparse search directly in Qdrant,
    fusing results natively with on-disk Reciprocal Rank Fusion (RRF).
    Eliminates in-memory corpus indexing for constant O(1) RAM usage.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunks: Optional[List[Any]] = None,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.rrf_k = rrf_k

    def build_bm25_index(self, chunks: Optional[List[Any]] = None) -> None:
        """
        No-op compatibility method. Sparse indexing is now performed directly
        on-disk within Qdrant during upsertion, eliminating in-memory indexing.
        """
        pass

    def search(
        self,
        query: str,
        top_k: int = 50,
        candidate_pool: int = 50,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Executes native hybrid search (dense embeddings + sparse keywords) in Qdrant
        with on-disk Reciprocal Rank Fusion (RRF).
        """
        k = top_k or candidate_pool

        # 1. Encode query into dense vector (only single string in RAM)
        query_dense = self.embedding_model.encode(
            query, normalize_embeddings=True
        ).tolist()

        # 2. Execute native hybrid search directly inside Qdrant
        fused_results = self.vector_store.search_hybrid(
            query_dense=query_dense,
            query_text=query,
            top_k=k,
            source_type=source_type,
        )

        return fused_results

