"""
Hierarchical Parent-Child chunking implementation for enterprise corpora.
"""
from typing import Dict, List, Tuple, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_benchmark.models import CleanedDocuments, RawDocuments, ParentChunk, ChildChunk

class HierarchicalChunker:
    """
    Splits documents into large Parent chunks (for LLM context hydration)
    and granular Child chunks (for high-precision vector and BM25 search).
    """

    def __init__(
        self,
        parent_chunk_size: int = 3000,
        parent_chunk_overlap: int = 300,
        child_chunk_size: int = 800,
        child_chunk_overlap: int = 100,
    ) -> None:
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n## ", "\n\n### ", "\n\n", "\n", ". ", " "]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def chunk_document(
        self, doc: Union[CleanedDocuments, RawDocuments, dict]
    ) -> Tuple[Dict[str, ParentChunk], List[ChildChunk]]:
        """
        Splits a single document into parents and children.
        Supports both Pydantic models and dictionary representations.
        
        Returns:
            parent_store: Dict mapping parent_id -> ParentChunk
            child_records: List of ChildChunk models
        """
        if isinstance(doc, dict):
            doc_id = doc.get("doc_id", "")
            source_type = doc.get("source_type", "")
            title = doc.get("cleaned_title") or doc.get("normalized_title") or doc.get("title", "")
            content = doc.get("cleaned_content") or doc.get("normalized_content") or doc.get("content", "")
        else:
            doc_id = doc.doc_id
            source_type = doc.source_type
            title = getattr(doc, "cleaned_title", None) or getattr(doc, "title", "")
            content = getattr(doc, "cleaned_content", None) or getattr(doc, "content", "")

        parent_store: Dict[str, ParentChunk] = {}
        child_records: List[ChildChunk] = []

        if not content or not content.strip():
            return parent_store, child_records

        # 1. Split into large parent chunks (sections)
        parent_texts = self.parent_splitter.split_text(content)

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = f"{doc_id}_p{p_idx}"
            
            parent_chunk = ParentChunk(
                parent_id=parent_id,
                doc_id=doc_id,
                source_type=source_type,
                title=title,
                parent_text=parent_text,
            )
            parent_store[parent_id] = parent_chunk

            # 2. Split each parent into granular child chunks (paragraphs)
            child_texts = self.child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_texts):
                child_id = f"{parent_id}_c{c_idx}"
                
                child_records.append(
                    ChildChunk(
                        child_id=child_id,
                        parent_id=parent_id,
                        doc_id=doc_id,
                        source_type=source_type,
                        title=title,
                        child_text=child_text,
                    )
                )

        return parent_store, child_records

    def chunk_documents(
        self, docs: List[Union[CleanedDocuments, RawDocuments, dict]]
    ) -> Tuple[Dict[str, ParentChunk], List[ChildChunk]]:
        """
        Batch chunks multiple documents and aggregates all parent and child chunks.
        """
        all_parents: Dict[str, ParentChunk] = {}
        all_children: List[ChildChunk] = []

        for doc in docs:
            doc_parents, doc_children = self.chunk_document(doc)
            all_parents.update(doc_parents)
            all_children.extend(doc_children)

        return all_parents, all_children
