import uuid
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag_benchmark.models import ChildChunk, ParentChunk


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
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
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
                    vector=chunk.dense_vector,
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

    def search(
        self,
        query_vector: List[float],
        top_k: int = 50,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Executes HNSW dense vector search with optional payload filtering.
        """
        # Optional metadata pre-filtering (e.g. source_type == 'confluence')
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

        # Qdrant query_points search
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
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