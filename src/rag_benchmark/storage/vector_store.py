import collections
import math
import re
import uuid
import zlib
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from rag_benchmark.models import ChildChunk, ParentChunk


def compute_sparse_vector(text: str) -> SparseVector:
    """
    Computes deterministic on-disk term frequency sparse vector.
    Maps word tokens to positive 31-bit integer indices with sublinear term weighting.
    """
    if not text:
        return SparseVector(indices=[], values=[])
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return SparseVector(indices=[], values=[])

    counts = collections.Counter(tokens)
    indices: List[int] = []
    values: List[float] = []

    for token, count in counts.items():
        idx = zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF
        val = float(1.0 + math.log(count))
        indices.append(idx)
        values.append(val)

    return SparseVector(indices=indices, values=values)


class QdrantVectorStore:
    def __init__(
        self,
        storage_path: str = "data/qdrant_db",
        collection_name: str = "enterprise_rag_chunks",
        vector_size: int = 384,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.client = QdrantClient(path=storage_path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        on_disk=True,
                    )
                },
                hnsw_config=HnswConfigDiff(
                    on_disk=True,
                ),
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=True),
                    )
                },
            )

    def upsert_chunks(
        self,
        chunks: List[ChildChunk],
        parent_store: Optional[Dict[str, ParentChunk]] = None,
        batch_size: int = 500,
    ) -> int:
        points: List[PointStruct] = []

        for chunk in chunks:
            if not chunk.dense_vector:
                continue

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.child_id))

            parent_text = ""
            if parent_store and chunk.parent_id in parent_store:
                parent_text = parent_store[chunk.parent_id].parent_text

            payload = {
                "child_id": chunk.child_id,
                "parent_id": chunk.parent_id,
                "doc_id": chunk.doc_id,
                "source_type": chunk.source_type,
                "title": chunk.title,
                "child_text": chunk.child_text,
                "parent_text": parent_text,  # Instant hydration payload!
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": chunk.dense_vector,
                        "sparse": compute_sparse_vector(chunk.child_text),
                    },
                    payload=payload,
                )
            )

        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_upserted += len(batch)

        return total_upserted

    def search_hybrid(
        self,
        query_dense: List[float],
        query_text: str,
        top_k: int = 50,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Executes native Qdrant Hybrid search combining dense vectors and sparse keywords
        with Reciprocal Rank Fusion (RRF) on disk.
        """
        query_filter = None
        if source_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type),
                    )
                ]
            )

        query_sparse = compute_sparse_vector(query_text)

        prefetches = [
            Prefetch(
                query=query_dense,
                using="dense",
                filter=query_filter,
                limit=top_k,
            )
        ]
        if query_sparse.indices:
            prefetches.append(
                Prefetch(
                    query=query_sparse,
                    using="sparse",
                    filter=query_filter,
                    limit=top_k,
                )
            )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetches,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        results = []
        for rank, hit in enumerate(search_result.points, start=1):
            payload = hit.payload or {}
            results.append({
                "rank": rank,
                "score": hit.score,
                "child_id": payload.get("child_id"),
                "parent_id": payload.get("parent_id"),
                "doc_id": payload.get("doc_id"),
                "source_type": payload.get("source_type"),
                "title": payload.get("title"),
                "child_text": payload.get("child_text"),
                "parent_text": payload.get("parent_text"),
            })

        return results

    def search(
        self,
        query_vector: List[float],
        top_k: int = 50,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Executes HNSW dense vector search with optional payload filtering.
        """
        query_filter = None
        if source_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type),
                    )
                ]
            )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in search_result.points:
            payload = hit.payload or {}
            results.append({
                "score": hit.score,
                "child_id": payload.get("child_id"),
                "parent_id": payload.get("parent_id"),
                "doc_id": payload.get("doc_id"),
                "source_type": payload.get("source_type"),
                "title": payload.get("title"),
                "child_text": payload.get("child_text"),
                "parent_text": payload.get("parent_text"),
            })

        return results

    def count(self) -> int:
        """Returns the total number of indexed points."""
        return self.client.count(collection_name=self.collection_name).count

    def close(self) -> None:
        """Closes the underlying Qdrant client connection."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
