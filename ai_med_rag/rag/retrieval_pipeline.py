"""
Retrieval orchestration for the Medical RAG application.

Save as:
    rag/retrieval_pipeline.py

This module is extracted from the evaluation notebook's
`_retrieve_planned_subqueries(...)` logic.

Responsibilities:
- Run each decomposed subquery with its own adaptive retrieval_k.
- Merge retrieved chunks.
- Deduplicate by FAISS vector ID.
- Preserve which subqueries matched each chunk.
- Globally rerank the merged evidence against the ORIGINAL user question.
- Cap multi-subquery context growth at 15 chunks.

It does NOT generate the final answer. Generation will remain separate.
"""

from __future__ import annotations

from typing import Any

from retrieval.medcpt_retriever import MedCPTRetriever


DECOMPOSED_FINAL_CONTEXT_CAP = 15


class RetrievalPipeline:
    """
    Orchestrates single-query and decomposed-query retrieval.

    Parameters
    ----------
    retriever:
        Initialized MedCPTRetriever.
    final_context_cap:
        Maximum number of final chunks after merging multiple subqueries.
        The evaluation notebook uses 15.
    """

    def __init__(
        self,
        retriever: MedCPTRetriever,
        final_context_cap: int = DECOMPOSED_FINAL_CONTEXT_CAP,
    ) -> None:
        self.retriever = retriever
        self.final_context_cap = int(final_context_cap)

        if self.final_context_cap < 1:
            raise ValueError("final_context_cap must be at least 1.")

    # ------------------------------------------------------------------
    # Main decomposed-query retrieval
    # ------------------------------------------------------------------

    def retrieve_planned_subqueries(
        self,
        original_query: str,
        subquery_plans: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        float,
        list[dict[str, Any]],
    ]:
        """
        Retrieve EACH decomposed subquery using its own adaptive k.

        Example plan:
            {
                "subquery": "...",
                "difficulty": "mid",
                "retrieval_k": 7,
                "short_reason": "..."
            }

        Returns
        -------
        final_chunks:
            Deduplicated and globally reranked evidence chunks.

        total_latency:
            Sum of per-subquery retrieval latency.

        retrieval_audit:
            Per-subquery retrieval details.
        """

        original_query = str(original_query).strip()

        if not original_query:
            raise ValueError("Original query cannot be empty.")

        if not subquery_plans:
            raise ValueError(
                "subquery_plans cannot be empty."
            )

        all_chunks: dict[int, dict[str, Any]] = {}
        total_latency = 0.0
        retrieval_audit: list[dict[str, Any]] = []

        # --------------------------------------------------------------
        # Retrieve each subquery independently
        # --------------------------------------------------------------
        for plan in subquery_plans:
            subquery = str(
                plan.get("subquery") or ""
            ).strip()

            if not subquery:
                continue

            try:
                retrieval_k = int(
                    plan.get("retrieval_k")
                )
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid retrieval_k for subquery: {subquery}"
                )

            if retrieval_k < 1:
                raise ValueError(
                    f"retrieval_k must be >= 1 for subquery: {subquery}"
                )

            difficulty = plan.get("difficulty")

            chunks, latency = self.retriever.retrieve(
                query=subquery,
                final_k=retrieval_k,
            )

            total_latency += float(latency)

            retrieval_audit.append(
                {
                    "subquery": subquery,
                    "difficulty": difficulty,
                    "retrieval_k": retrieval_k,
                    "returned_chunks": len(chunks),
                }
            )

            # ----------------------------------------------------------
            # Merge and deduplicate by FAISS vector ID
            # ----------------------------------------------------------
            for chunk in chunks:
                faiss_id = int(chunk["faiss_id"])
                previous = all_chunks.get(faiss_id)

                if (
                    previous is None
                    or float(chunk["rerank_score"])
                    > float(previous["rerank_score"])
                ):
                    kept = dict(chunk)

                    # Preserve existing matches if this vector was already seen.
                    old_subqueries = []
                    old_difficulties = []

                    if previous is not None:
                        old_subqueries = list(
                            previous.get(
                                "matched_subqueries",
                                [],
                            )
                        )
                        old_difficulties = list(
                            previous.get(
                                "matched_subquery_difficulties",
                                [],
                            )
                        )

                    kept["matched_subqueries"] = (
                        old_subqueries + [subquery]
                    )

                    kept[
                        "matched_subquery_difficulties"
                    ] = (
                        old_difficulties + [difficulty]
                    )

                    all_chunks[faiss_id] = kept

                else:
                    previous.setdefault(
                        "matched_subqueries",
                        [],
                    ).append(subquery)

                    previous.setdefault(
                        "matched_subquery_difficulties",
                        [],
                    ).append(difficulty)

        merged = list(all_chunks.values())

        if not merged:
            return (
                [],
                total_latency,
                retrieval_audit,
            )

        # --------------------------------------------------------------
        # Determine final context size
        # --------------------------------------------------------------
        if len(subquery_plans) == 1:
            # Preserve the single subquery's requested retrieval depth.
            final_k = int(
                subquery_plans[0]["retrieval_k"]
            )

        else:
            # Preserve the evaluation notebook policy:
            #
            # requested total evidence budget
            #        ↓
            # sum(retrieval_k)
            #        ↓
            # capped at 15
            final_k = min(
                self.final_context_cap,
                sum(
                    int(plan["retrieval_k"])
                    for plan in subquery_plans
                ),
                len(merged),
            )

        # --------------------------------------------------------------
        # Global reranking against ORIGINAL user question
        # --------------------------------------------------------------
        #
        # `rerank_candidates` only needs the FAISS IDs plus the retrieval
        # scores/ranks. It reloads the actual chunk text from metadata.
        #
        fused_like = [
            {
                "faiss_id": chunk["faiss_id"],
                "rrf_score": chunk.get(
                    "rrf_score",
                    0.0,
                ),
                "dense_rank": chunk.get(
                    "dense_rank"
                ),
                "bm25_rank": chunk.get(
                    "bm25_rank"
                ),
                "dense_score": chunk.get(
                    "dense_score"
                ),
                "bm25_score": chunk.get(
                    "bm25_score"
                ),
            }
            for chunk in merged
        ]

        final_chunks = self.retriever.rerank_candidates(
            query=original_query,
            candidates=fused_like,
            top_k=final_k,
        )

        # --------------------------------------------------------------
        # Restore decomposition audit metadata after global reranking
        # --------------------------------------------------------------
        merged_lookup = {
            int(chunk["faiss_id"]): chunk
            for chunk in merged
        }

        for chunk in final_chunks:
            old = merged_lookup.get(
                int(chunk["faiss_id"]),
                {},
            )

            chunk["matched_subqueries"] = list(
                old.get(
                    "matched_subqueries",
                    [],
                )
            )

            chunk[
                "matched_subquery_difficulties"
            ] = list(
                old.get(
                    "matched_subquery_difficulties",
                    [],
                )
            )

        return (
            final_chunks,
            total_latency,
            retrieval_audit,
        )

    # ------------------------------------------------------------------
    # Normal medical QA path
    # ------------------------------------------------------------------

    def retrieve_normal_query(
        self,
        query: str,
        final_k: int,
    ) -> tuple[
        list[dict[str, Any]],
        float,
        list[dict[str, Any]],
    ]:
        """
        Retrieve a normal medical question.

        The evaluation notebook converts the normal query into one retrieval
        plan and then sends it through the same shared retrieval function.
        """

        plan = {
            "subquery": str(query).strip(),
            "difficulty": None,
            "retrieval_k": int(final_k),
            "short_reason": (
                "Normal-QA retrieval budget."
            ),
        }

        return self.retrieve_planned_subqueries(
            original_query=query,
            subquery_plans=[plan],
        )

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def build_context(
        self,
        chunks: list[dict[str, Any]],
    ) -> str:
        """
        Convert final chunks into the citation-ready [C1], [C2], ... format.
        """

        return self.retriever.chunks_to_context(
            chunks
        )


# ----------------------------------------------------------------------
# Function-style wrapper
# ----------------------------------------------------------------------

def retrieve_planned_subqueries(
    original_query: str,
    subquery_plans: list[dict[str, Any]],
    retriever: MedCPTRetriever,
    final_context_cap: int = DECOMPOSED_FINAL_CONTEXT_CAP,
):
    """
    Function-style wrapper matching the notebook's original organization.
    """

    pipeline = RetrievalPipeline(
        retriever=retriever,
        final_context_cap=final_context_cap,
    )

    return pipeline.retrieve_planned_subqueries(
        original_query=original_query,
        subquery_plans=subquery_plans,
    )
