"""
RAG answer generation for the Medical RAG application.

Save as:
    rag/generator.py

This module is adapted directly from the evaluation notebook's:
- RAG_SYSTEM
- generate_rag_answer(...)
- generate_decomposed_rag_answer(...)

The only generation-backend change is:
    vllm_chat(...) -> OllamaGenerator.chat(...)
"""

from __future__ import annotations

from typing import Any, Optional

from models.ollama_generator import OllamaGenerator
from rag.retrieval_pipeline import RetrievalPipeline


RAG_SYSTEM = r"""
You are an evidence-grounded medical question-answering assistant.

Rules:
1. Base the medical answer on the supplied retrieved evidence.
2. Cite supporting retrieved chunks inline using their IDs exactly, e.g. [C1], [C3].
3. Never invent a chunk ID.
4. If the evidence is insufficient, say so explicitly instead of guessing.
5. Use professional clinical language, but do not claim to be a physician.
6. Be clear about uncertainty.
7. Do not reveal private health information belonging to another user/person.
8. Minimize repetition of the user's own identifying information unless clinically needed.
9. If the evidence indicates urgent/emergency care is warranted, say so plainly.
"""


class RAGGenerator:
    """
    Generate final evidence-grounded answers with the user-selected Ollama model.

    Parameters
    ----------
    generator:
        Initialized OllamaGenerator.
    retrieval_pipeline:
        Initialized RetrievalPipeline.
    max_tokens:
        Maximum generated tokens for the final answer.
    """

    def __init__(
        self,
        generator: OllamaGenerator,
        retrieval_pipeline: RetrievalPipeline,
        max_tokens: int = 1536,
    ) -> None:
        self.generator = generator
        self.retrieval_pipeline = retrieval_pipeline
        self.max_tokens = int(max_tokens)

    # ------------------------------------------------------------------
    # Normal Medical QA
    # ------------------------------------------------------------------

    def generate_rag_answer(
        self,
        query: str,
        final_k: int,
        subqueries: Optional[list[str]] = None,
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """
        Normal Medical QA path.

        This preserves the evaluation notebook's retrieval behavior:
        each supplied subquery uses the same final_k, then the merged evidence
        is globally reranked against the original query.
        """

        query = str(query).strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        subqueries = subqueries or [query]

        plans = [
            {
                "subquery": str(sq).strip(),
                "difficulty": None,
                "retrieval_k": int(final_k),
                "short_reason": "Normal-QA retrieval budget.",
            }
            for sq in subqueries
            if str(sq).strip()
        ]

        if not plans:
            plans = [
                {
                    "subquery": query,
                    "difficulty": None,
                    "retrieval_k": int(final_k),
                    "short_reason": "Normal-QA retrieval budget.",
                }
            ]

        chunks, retrieval_latency, retrieval_audit = (
            self.retrieval_pipeline.retrieve_planned_subqueries(
                original_query=query,
                subquery_plans=plans,
            )
        )

        context = self.retrieval_pipeline.build_context(chunks)

        messages = self._build_messages(
            user_content=(
                f"User question:\n{query}\n\n"
                f"Retrieved evidence:\n{context}\n\n"
                "Answer the user's question."
            ),
            chat_history=chat_history,
        )

        answer, gen_meta = self.generator.chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "retrieved_context": context,
            "retrieval_latency_s": float(retrieval_latency),
            "retrieval_audit": retrieval_audit,
            "generation_meta": gen_meta,
        }

    # ------------------------------------------------------------------
    # Decomposed Medical QA
    # ------------------------------------------------------------------

    def generate_decomposed_rag_answer(
        self,
        query: str,
        subquery_plans: list[dict[str, Any]],
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """
        Clarified AMBIGUOUS / CONTRADICTION and OLD_RELEVANT path.

        Every decomposed subquery independently reuses the same adaptive
        Medical-QA retrieval policy:
            easy -> 3-4 chunks
            mid  -> 5-8 chunks
            hard -> 9-15 chunks

        Evidence is merged, deduplicated, globally reranked, then ONE final
        answer is generated for the original user query.
        """

        query = str(query).strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if not subquery_plans:
            raise ValueError("subquery_plans cannot be empty.")

        chunks, retrieval_latency, retrieval_audit = (
            self.retrieval_pipeline.retrieve_planned_subqueries(
                original_query=query,
                subquery_plans=subquery_plans,
            )
        )

        context = self.retrieval_pipeline.build_context(chunks)

        plan_text = "\n".join(
            (
                f"- {p['subquery']} | "
                f"difficulty={p.get('difficulty')} | "
                f"top_k={p['retrieval_k']}"
            )
            for p in subquery_plans
        )

        messages = self._build_messages(
            user_content=(
                f"Original user question:\n{query}\n\n"
                f"Decomposed retrieval plan:\n{plan_text}\n\n"
                f"Merged and reranked evidence:\n{context}\n\n"
                "Synthesize one answer to the ORIGINAL user question."
            ),
            chat_history=chat_history,
        )

        answer, gen_meta = self.generator.chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "retrieved_context": context,
            "retrieval_latency_s": float(retrieval_latency),
            "retrieval_audit": retrieval_audit,
            "generation_meta": gen_meta,
        }

    # ------------------------------------------------------------------
    # Chat-memory integration
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        user_content: str,
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> list[dict[str, str]]:
        """
        Build Ollama chat messages while keeping the original RAG system prompt.

        Chat history is appended before the current evidence-grounded request.
        Only user/assistant roles are accepted.
        """

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": RAG_SYSTEM,
            }
        ]

        for item in chat_history or []:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()

            if role not in {"user", "assistant"}:
                continue

            if not content:
                continue

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        return messages


# ----------------------------------------------------------------------
# Function-style wrappers
# ----------------------------------------------------------------------

def generate_rag_answer(
    query: str,
    final_k: int,
    generator: OllamaGenerator,
    retrieval_pipeline: RetrievalPipeline,
    subqueries: Optional[list[str]] = None,
    chat_history: Optional[list[dict[str, str]]] = None,
    max_tokens: int = 1536,
) -> dict[str, Any]:
    """
    Function-style wrapper matching the evaluation notebook's organization.
    """

    rag_generator = RAGGenerator(
        generator=generator,
        retrieval_pipeline=retrieval_pipeline,
        max_tokens=max_tokens,
    )

    return rag_generator.generate_rag_answer(
        query=query,
        final_k=final_k,
        subqueries=subqueries,
        chat_history=chat_history,
    )


def generate_decomposed_rag_answer(
    query: str,
    subquery_plans: list[dict[str, Any]],
    generator: OllamaGenerator,
    retrieval_pipeline: RetrievalPipeline,
    chat_history: Optional[list[dict[str, str]]] = None,
    max_tokens: int = 1536,
) -> dict[str, Any]:
    """
    Function-style wrapper matching the evaluation notebook's organization.
    """

    rag_generator = RAGGenerator(
        generator=generator,
        retrieval_pipeline=retrieval_pipeline,
        max_tokens=max_tokens,
    )

    return rag_generator.generate_decomposed_rag_answer(
        query=query,
        subquery_plans=subquery_plans,
        chat_history=chat_history,
    )
