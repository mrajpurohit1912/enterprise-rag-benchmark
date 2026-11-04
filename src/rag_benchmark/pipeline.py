"""
Enterprise End-to-End RAG Pipeline.
Coordinates: Ingestion -> Normalization -> Deduplication -> Hierarchical Chunking ->
Qdrant Vector Store + BM25 -> Hybrid Retrieval (RRF) -> Cross-Encoder Reranking ->
Local Grounded LLM Generation.
"""
from typing import Any, Dict, List, Optional, Union
from collections.abc import Iterable,Iterator

from rag_benchmark.chunking.hierarchical import HierarchicalChunker
from rag_benchmark.config import RAGConfig
from rag_benchmark.data.deduplicator import ExactDeduplicator
from rag_benchmark.data.normalizer import normalize_text
from rag_benchmark.generation.generator import LocalRAGGenerator
from rag_benchmark.models import ChildChunk, CleanedDocuments, ParentChunk, RawDocuments
from rag_benchmark.retrieval.hybrid import HybridRetriever
from rag_benchmark.retrieval.reranker import CrossEncoderReranker
from rag_benchmark.storage.vector_store import QdrantVectorStore


class EnterpriseRAGPipeline:
    """
    Production-grade enterprise RAG pipeline integrating dense & sparse hybrid search,
    parent-child context hydration, cross-encoder reranking, and local citation-grounded LLM inference.
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or RAGConfig()

        # Ingestion & Preprocessing
        self.deduplicator = ExactDeduplicator()
        self.chunker = HierarchicalChunker(
            parent_chunk_size=self.config.parent_chunk_size,
            parent_chunk_overlap=self.config.parent_chunk_overlap,
            child_chunk_size=self.config.child_chunk_size,
            child_chunk_overlap=self.config.child_chunk_overlap,
        )

        # Storage & Retrieval
        self.vector_store = QdrantVectorStore(
            storage_path=self.config.storage_path,
            collection_name=self.config.collection_name,
            vector_size=self.config.vector_dim,
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model_name=self.config.embedding_model,
            rrf_k=self.config.rrf_k,
        )
        self.reranker = CrossEncoderReranker(
            model_name=self.config.reranker_model,
        )

        # Lazy-loaded Generator (avoids GPU VRAM allocation until query time)
        self._generator: Optional[LocalRAGGenerator] = None

    @property
    def generator(self) -> LocalRAGGenerator:
        """Lazy loads the generative LLM pipeline onto compute device."""
        if self._generator is None:
            self._generator = LocalRAGGenerator(
                model_name=self.config.generator_model,
            )
        return self._generator

    def iter_cleaned_documents(
        self,
        raw_docs: Iterable[dict],
    ) -> Iterator[CleanedDocuments]:
        """
        Indexes an incoming batch of documents into the enterprise RAG system:
        1. Normalizes text (Unicode NFKC, whitespace, linebreaks).
        2. Deduplicates exact matches via SHA-256 hash checking.
        3. Generates Parent (3000 chars) and Child (800 chars) chunks.
        4. Computes dense embeddings with SentenceTransformers.
        5. Persists child points with hydrated parent payloads in Qdrant.
        6. Builds in-memory BM25 index for hybrid retrieval.

        Args:
            raw_docs: List of raw document dicts or RawDocuments Pydantic models.
            batch_size: Batch size for vector store upsertion.

        Returns:
            Indexing statistics summary.
        """

        for doc in raw_docs:
            if isinstance(doc, dict):
                doc_id = doc.get("doc_id", "")
                source_type = doc.get("source_type", "")
                title = doc.get("title", "")
                content = doc.get("content", "")
            else:
                doc_id = doc.doc_id
                source_type = doc.source_type
                title = doc.title
                content = doc.content

            norm_title = normalize_text(title)
            norm_content = normalize_text(content)

            # Skip exact duplicates
            if self.deduplicator.is_duplicate(doc_id, norm_content):
                continue

            raw_model = RawDocuments(
                doc_id=doc_id,
                source_type=source_type,
                title=title,
                content=content,
            )
            yield CleanedDocuments(
                    raw_documents=raw_model,
                    doc_id=doc_id,
                    source_type=source_type,
                    title=norm_title,
                    content=norm_content,
                    cleaned_title=norm_title,
                    cleaned_content=norm_content,
                )
            
    def _batch_iterator(
        self,
        iterable: Iterable[Any],
        batch_size: int = 256,
    ) -> Iterator[List[Any]]:
        """Yields bounded slices from an iterable to preserve constant O(1) memory."""
        batch: List[Any] = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def index_documents(
        self,
        raw_docs: Iterable[dict],
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """
        Stream-indexes incoming documents in bounded micro-batches:
        1. Normalizes and deduplicates via generator streaming.
        2. Chunks documents into hierarchical Parent/Child chunks per batch.
        3. Computes dense embeddings and on-disk sparse representations.
        4. Persists child points with hydrated parent payloads directly in Qdrant.
        5. Automatically collects garbage per batch, maintaining flat memory usage.

        Args:
            raw_docs: Iterable of raw document dicts or RawDocuments Pydantic models.
            batch_size: Micro-batch size for streaming ingestion and upsertion.

        Returns:
            Indexing statistics summary.
        """
        import gc

        total_received = 0
        total_indexed = 0
        total_parents = 0
        total_children = 0
        total_upserted = 0

        if hasattr(raw_docs, "__len__"):
            try:
                total_received = len(raw_docs)
            except TypeError:
                pass

        # Stream documents in batches of `batch_size`
        cleaned_stream = self.iter_cleaned_documents(raw_docs)
        for doc_batch in self._batch_iterator(cleaned_stream, batch_size=batch_size):
            total_indexed += len(doc_batch)

            # 1. Hierarchical chunking for this batch only
            batch_parents, batch_children = self.chunker.chunk_documents(doc_batch)
            if not batch_children:
                continue

            total_parents += len(batch_parents)
            total_children += len(batch_children)

            # 2. Dense Embeddings for Child Chunks in this batch
            child_texts = [c.child_text for c in batch_children]
            embeddings = self.retriever.embedding_model.encode(
                child_texts,
                batch_size=min(len(child_texts), 64),
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for chunk, emb in zip(batch_children, embeddings):
                chunk.dense_vector = emb.tolist()

            # 3. Upsert to Qdrant (dense vectors, sparse vectors, and parent payload stored on disk)
            upserted = self.vector_store.upsert_chunks(
                chunks=batch_children,
                parent_store=batch_parents,
                batch_size=batch_size,
            )
            total_upserted += upserted

            # 4. Immediate Garbage Collection of batch structures to maintain constant RAM
            del doc_batch, batch_parents, batch_children, child_texts, embeddings
            gc.collect()

        return {
            "documents_received": total_received or total_indexed,
            "documents_indexed": total_indexed,
            "duplicates_skipped": len(self.deduplicator.duplicate_ids),
            "parent_chunks": total_parents,
            "child_chunks": total_children,
            "points_upserted": total_upserted,
            "total_qdrant_points": self.vector_store.count(),
        }

    def query(
        self,
        query_text: str,
        top_k_retrieval: Optional[int] = None,
        top_n_rerank: Optional[int] = None,
        source_type: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end grounded RAG answering:
        1. Normalizes user query.
        2. Executes Native Hybrid Search in Qdrant (Dense + Sparse + RRF).
        3. Hydrates parent document text directly from on-disk payload.
        4. Reranks candidate passages using joint Cross-Encoder.
        5. Synthesizes final grounded answer with strict inline citations.

        Args:
            query_text: Natural language user question.
            top_k_retrieval: Number of candidates from hybrid search (defaults to config).
            top_n_rerank: Top passages fed to LLM after reranking (defaults to config).
            source_type: Optional metadata filter (e.g. 'slack', 'confluence', 'jira').
            max_new_tokens: Maximum tokens for LLM generation.

        Returns:
            Dictionary with query, answer, cited_doc_ids, context, retrieved_chunks, and metadata.
        """
        clean_query = normalize_text(query_text)
        k_retrieve = top_k_retrieval or self.config.hybrid_top_k
        n_rerank = top_n_rerank or self.config.rerank_top_n
        new_tokens = max_new_tokens or self.config.max_new_tokens

        # 1. Native Hybrid Retrieval (Qdrant Dense + Sparse fused with RRF on disk)
        candidates = self.retriever.search(
            query=clean_query,
            top_k=k_retrieve,
            source_type=source_type,
        )

        if not candidates:
            return {
                "query": query_text,
                "answer": "Information not found in internal documents.",
                "cited_doc_ids": [],
                "context": "",
                "retrieved_chunks": [],
                "metadata": {"status": "no_candidates_found"},
            }

        # 2. Cross-Encoder Reranking
        reranked_docs = self.reranker.rerank(
            query=clean_query,
            candidates=candidates,
            top_n=n_rerank,
        )

        # 3. Confidence Gatekeeper: Early refusal if top candidate is irrelevant
        best_score = reranked_docs[0].get("rerank_score", -999.0) if reranked_docs else -999.0
        if (
            self.config.rerank_score_threshold is not None
            and best_score < self.config.rerank_score_threshold
        ):
            return {
                "query": query_text,
                "answer": "Information not found in internal documents.",
                "cited_doc_ids": [],
                "context": "",
                "retrieved_chunks": reranked_docs,
                "metadata": {
                    "source_type_filter": source_type,
                    "status": "low_confidence_refusal",
                    "top_rerank_score": best_score,
                    "threshold": self.config.rerank_score_threshold,
                    "candidates_retrieved": len(candidates),
                    "docs_reranked": len(reranked_docs),
                },
            }

        # 4. Grounded Local Generation
        gen_result = self.generator.generate(
            query=clean_query,
            context_docs=reranked_docs,
            max_new_tokens=new_tokens,
            use_lost_in_middle=self.config.use_lost_in_middle,
        )

        return {
            "query": query_text,
            "answer": gen_result["answer"],
            "cited_doc_ids": gen_result["cited_doc_ids"],
            "context": gen_result["context"],
            "retrieved_chunks": reranked_docs,
            "metadata": {
                "source_type_filter": source_type,
                "candidates_retrieved": len(candidates),
                "docs_reranked": len(reranked_docs),
                "model_name": gen_result["model_name"],
            },
        }

    def close(self) -> None:
        """Closes any underlying storage connections."""
        self.vector_store.close()

    def __enter__(self) -> "EnterpriseRAGPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
