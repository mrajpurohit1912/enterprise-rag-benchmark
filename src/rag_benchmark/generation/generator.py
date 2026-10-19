"""
Local GPU/CPU LLM generator for grounded RAG answer synthesis.
"""
from typing import Any, Dict, List, Optional
import torch
from transformers import pipeline

from rag_benchmark.generation.prompt import (
    SYSTEM_PROMPT,
    build_messages,
    extract_citations,
    format_context_for_llm,
)


class LocalRAGGenerator:
    """
    Manages local LLM inference for RAG synthesis using Hugging Face transformers.
    Supports GPU acceleration (CUDA) with fp16/bf16 precision and deterministic decoding.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt

        # Determine compute device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Determine precision
        if torch_dtype is None:
            self.torch_dtype = (
                torch.float16 if self.device == "cuda" else torch.float32
            )
        else:
            self.torch_dtype = torch_dtype

        # Initialize Hugging Face pipeline using 'dtype'
        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            device=self.device,
            dtype=self.torch_dtype,
        )

    def generate(
        self,
        query: str,
        context_docs: List[Dict],
        max_new_tokens: int = 450,
        do_sample: bool = False,
        temperature: float = 0.0,
        use_lost_in_middle: bool = False,
        max_context_chars: Optional[int] = 8000,
    ) -> Dict[str, Any]:
        """
        Generates a grounded, citation-backed answer for the given query using retrieved context documents.

        Args:
            query: The user's question.
            context_docs: Ranked candidate document chunks with parent_text / child_text.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to sample (set False for deterministic greedy decoding).
            temperature: Sampling temperature (ignored if do_sample is False).
            use_lost_in_middle: Whether to place top evidence at start and end of prompt.
            max_context_chars: Maximum character budget for the injected context.

        Returns:
            Dictionary containing:
                - answer: Synthesized text response
                - cited_doc_ids: List of document IDs cited by the model
                - context: Assembled context string fed to LLM
                - retrieved_docs: Original retrieved docs passed in
                - model_name: Name of the generator model
        """
        formatted_context, doc_index_map = format_context_for_llm(
            reranked_docs=context_docs,
            deduplicate_by_parent=True,
            use_lost_in_middle=use_lost_in_middle,
            max_context_chars=max_context_chars,
        )

        messages = build_messages(
            query=query,
            context=formatted_context,
            system_prompt=self.system_prompt,
        )

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        output = self.generator(messages, **gen_kwargs)
        generated_answer = output[0]["generated_text"][-1]["content"].strip()

        cited_ids = extract_citations(generated_answer, doc_index_map=doc_index_map)

        return {
            "query": query,
            "answer": generated_answer,
            "cited_doc_ids": cited_ids,
            "context": formatted_context,
            "retrieved_docs": context_docs,
            "model_name": self.model_name,
        }
