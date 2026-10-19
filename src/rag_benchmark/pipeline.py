"""
Enterprise End-to-End RAG Pipeline.
Coordinates: Ingestion -> Normalization -> Deduplication -> Hierarchical Chunking ->
Qdrant Vector Store + BM25 -> Hybrid Retrieval (RRF) -> Cross-Encoder Reranking ->
Local Grounded LLM Generation.
"""
from typing import Any, Dict, List, Optional, Union

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

        # In-memory parent and child registries
        self.parent_store: Dict[str, ParentChunk] = {}
        self.child_chunks: List[ChildChunk] = []

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

    def index_documents(
        self,
        raw_docs: List[Union[Dict, RawDocuments]],
        batch_size: int = 500,
    ) -> Dict[str, Any]:
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
        cleaned_docs: List[CleanedDocuments] = []

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
            cleaned_docs.append(
                CleanedDocuments(
                    raw_documents=raw_model,
                    doc_id=doc_id,
                    source_type=source_type,
                    title=norm_title,
                    content=norm_content,
                    cleaned_title=norm_title,
                    cleaned_content=norm_content,
                )
            )

        if not cleaned_docs:
            return {
                "documents_received": len(raw_docs),
                "documents_indexed": 0,
                "duplicates_skipped": len(self.deduplicator.duplicate_ids),
                "parent_chunks": 0,
                "child_chunks": 0,
            }

        # 2. Hierarchical Chunking
        new_parents, new_children = self.chunker.chunk_documents(cleaned_docs)
        self.parent_store.update(new_parents)
        self.child_chunks.extend(new_children)

        # 3. Dense Embeddings for Child Chunks
        child_texts = [c.child_text for c in new_children]
        embeddings = self.retriever.embedding_model.encode(
            child_texts,
            normalize_embeddings=True,
            show_progress_bar=len(new_children) > 100,
        )

        for chunk, emb in zip(new_children, embeddings):
            chunk.dense_vector = emb.tolist()

        # 4. Upsert to Qdrant
        total_upserted = self.vector_store.upsert_chunks(
            chunks=new_children,
            parent_store=self.parent_store,
            batch_size=batch_size,
        )

        # 5. Build / Update BM25 Sparse Index
        self.retriever.build_bm25_index(self.child_chunks)

        return {
            "documents_received": len(raw_docs),
            "documents_indexed": len(cleaned_docs),
            "duplicates_skipped": len(self.deduplicator.duplicate_ids),
            "parent_chunks": len(new_parents),
            "child_chunks": len(new_children),
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
        2. Executes Hybrid Search (Qdrant Dense + BM25 Sparse + RRF).
        3. Hydrates parent document text for full context window.
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

        # 1. Hybrid Retrieval (Dense + BM25 fused with RRF)
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

        # 2. Hydrate Parent Text if missing (e.g. from BM25 fallback)
        for cand in candidates:
            if not cand.get("parent_text"):
                p_id = cand.get("parent_id")
                if p_id and p_id in self.parent_store:
                    cand["parent_text"] = self.parent_store[p_id].parent_text

        # 3. Cross-Encoder Reranking
        reranked_docs = self.reranker.rerank(
            query=clean_query,
            candidates=candidates,
            top_n=n_rerank,
        )

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
