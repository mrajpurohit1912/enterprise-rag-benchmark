"""
Prompt engineering and context formatting templates for enterprise RAG generation.
"""
import re
from typing import Dict, List, Optional, Tuple


SYSTEM_PROMPT = """You are an internal enterprise AI assistant for Redwood Inference.
Your task is to answer the user's question based strictly and exclusively on the provided internal documents.

STRICT OPERATIONAL RULES:
1. CITATIONS: Every single factual statement, step, or recommendation you make MUST include an inline citation with its Document ID, formatted as [Doc ID: <doc_id>] (e.g., [Doc ID: dsid_xxx]).
2. ZERO HALLUCINATION / REFUSAL: If the provided documents do not contain sufficient evidence to answer the question, state: "Information not found in internal documents." Do NOT extrapolate, speculate, or use outside knowledge.
3. CONFLICTS: If two documents provide contradictory procedures or numbers, explicitly state both viewpoints and cite their respective Document IDs.
4. TONE: Professional, concise, and structured (use bullet points and bold headers for operational procedures).
"""


def lost_in_the_middle_reorder(documents: List[Dict]) -> List[Dict]:
    """
    Reorders documents according to the 'Lost in the Middle' heuristic:
    Places the most relevant documents at the very beginning and very end
    of the context window, where LLM attention weights are highest.
    """
    if len(documents) <= 2:
        return documents

    result: List[Dict] = [None] * len(documents)  # type: ignore
    head = 0
    tail = len(documents) - 1
    for i, doc in enumerate(documents):
        if i % 2 == 0:
            result[head] = doc
            head += 1
        else:
            result[tail] = doc
            tail -= 1

    return result


def format_context_for_llm(
    reranked_docs: List[Dict],
    deduplicate_by_parent: bool = True,
    use_lost_in_middle: bool = False,
    max_context_chars: Optional[int] = None,
) -> Tuple[str, Dict[int, str]]:
    """
    Assembles retrieved and hydrated documents into a structured context block
    with explicit Document IDs for citations.

    Returns:
        Tuple of (formatted_context_string, doc_index_to_id_mapping)
    """
    if not reranked_docs:
        return "No relevant internal documents found.", {}

    # 1. Deduplicate if multiple children belong to the same parent
    docs_to_format: List[Dict] = []
    seen_keys = set()

    for doc in reranked_docs:
        key = (
            doc.get("parent_id")
            if deduplicate_by_parent and doc.get("parent_id")
            else doc.get("child_id") or doc.get("doc_id")
        )
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        docs_to_format.append(doc)

    # 2. Optional Lost-in-the-Middle reordering
    if use_lost_in_middle and len(docs_to_format) > 2:
        docs_to_format = lost_in_the_middle_reorder(docs_to_format)

    # 3. Assemble document blocks and mapping
    context_blocks = []
    doc_index_map: Dict[int, str] = {}
    total_chars = 0

    for i, doc in enumerate(docs_to_format, start=1):
        content = (
            doc.get("parent_text")
            or doc.get("hydrated_parent_text")
            or doc.get("child_text")
            or ""
        ).strip()

        doc_id = doc.get("doc_id", "UNKNOWN")
        source = str(doc.get("source_type", "UNKNOWN")).upper()
        title = doc.get("title", "Untitled Document")

        block = (
            f"--- DOCUMENT {i} ---\n"
            f"Doc ID: {doc_id}\n"
            f"Source: {source}\n"
            f"Title: {title}\n"
            f"Content:\n{content}\n"
        )

        if max_context_chars and (total_chars + len(block) > max_context_chars):
            break

        context_blocks.append(block)
        doc_index_map[i] = doc_id
        total_chars += len(block)

    return "\n".join(context_blocks), doc_index_map


def build_messages(
    query: str,
    context: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """
    Formats system prompt, context documents, and query into standard ChatML / OpenAI message format.
    Reinforces citation instructions at the immediate point of inference.
    """
    user_content = (
        f"CONTEXT DOCUMENTS:\n{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"INSTRUCTION: Answer the question using ONLY the provided internal documents above. "
        f"Always provide inline citations with the exact Document ID like [Doc ID: <doc_id>]."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_rag_prompt(
    query: str,
    context: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Builds a single string prompt suitable for raw text-completion models.
    """
    return f"""{system_prompt}

CONTEXT DOCUMENTS:
{context}

USER QUESTION:
{query}

INSTRUCTION: Answer strictly based on the context above. Include inline citations like [Doc ID: <doc_id>].

ANSWER:"""


def extract_citations(
    text: str,
    doc_index_map: Optional[Dict[int, str]] = None,
) -> List[str]:
    """
    Extracts cited document IDs from the generated text.
    Handles:
    - Explicit citations: [Doc ID: dsid_xxx] or [dsid_xxx]
    - Direct corpus doc_id mentions: dsid_xxx
    - Ordinal references (e.g. 'Document 1', 'Document 2') mapped via doc_index_map
    """
    cited: List[str] = []

    # 1. Match patterns like [Doc ID: dsid_xxx] or [Doc ID: xxx]
    explicit_matches = re.findall(r"\[Doc ID:\s*([a-zA-Z0-9_\-]+)\]", text, re.IGNORECASE)
    cited.extend(explicit_matches)

    # 2. Match direct corpus IDs like dsid_xxx
    dsid_matches = re.findall(r"\b(dsid_[a-zA-Z0-9_\-]+)\b", text)
    cited.extend(dsid_matches)

    # 3. If doc_index_map provided, detect ordinal references like 'Document 1', 'Document 2'
    if doc_index_map:
        ordinal_matches = re.findall(r"\bDocument\s+(\d+)\b", text, re.IGNORECASE)
        for idx_str in ordinal_matches:
            idx = int(idx_str)
            if idx in doc_index_map:
                cited.append(doc_index_map[idx])

    # Deduplicate while preserving occurrence order
    return list(dict.fromkeys(cited))
