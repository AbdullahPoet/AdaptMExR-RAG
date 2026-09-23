"""
Decomposition Agent for the Medical RAG application.

Save as:
    agents/decomposition.py

This file is adapted directly from the uploaded evaluation notebook's
shared decomposition and subquery-difficulty logic.

It replaces `vllm_chat(...)` with `OllamaGenerator` while preserving:
- minimum 1-3 retrieval subqueries, maximum 4
- clinically relevant history preservation
- irrelevant identifier/distractor removal
- Easy / Mid / Hard subquery difficulty
- adaptive retrieval_k:
    easy: 3-4
    mid: 5-8
    hard: 9-15
"""

from __future__ import annotations

from typing import Any

from agents.query_analyzer import clamp_k, extract_json_object
from models.ollama_generator import OllamaGenerator


DECOMPOSE_SYSTEM = r"""
You decompose a medical question for evidence retrieval.
Do NOT answer it.

Create the MINIMUM number of self-contained retrieval subqueries needed to answer
the user's question. Usually 1-3; use at most 4.
Preserve clinically relevant history and remove irrelevant identifiers/distractors.

Return ONLY JSON:
{
  "subqueries": ["...", "..."]
}
"""


SUBQUERY_DIFFICULTY_SYSTEM = r"""
You are a medical retrieval-difficulty classifier.
Do NOT answer the question.

Classify ONE decomposed medical retrieval subquery as:
easy = direct single-fact/simple medical QA
mid = moderate synthesis/comparison or several facts
hard = complex/multi-step/high-risk clinical reasoning

Choose retrieval_k inside:
easy: 3-4
mid: 5-8
hard: 9-15

Return ONLY JSON:
{
  "difficulty": "easy|mid|hard",
  "retrieval_k": integer,
  "short_reason": "one concise audit note"
}
"""


class DecompositionAgent:
    """
    Ollama-backed medical query decomposition agent.

    A single instance uses the user-selected Ollama model for both:
    1. subquery decomposition
    2. per-subquery retrieval-depth classification
    """

    def __init__(self, generator: OllamaGenerator) -> None:
        self.generator = generator

    def decompose(
        self,
        query: str,
        relationship_summary: str = "",
        clarification_context: str = "",
    ) -> tuple[list[str], dict[str, Any]]:
        """
        Decompose one medical query into the minimum set of retrieval subqueries.

        Returns
        -------
        subqueries:
            1-4 self-contained medical retrieval queries.
        meta:
            Ollama generation metadata.
        """

        query = str(query).strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        raw, meta = self.generator.chat(
            messages=[
                {
                    "role": "system",
                    "content": DECOMPOSE_SYSTEM,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Relationship summary:\n"
                        f"{relationship_summary}\n\n"
                        f"Additional clarification:\n"
                        f"{clarification_context}"
                    ),
                },
            ],
            max_tokens=350,
            temperature=0.0,
        )

        obj = extract_json_object(raw)

        subqueries = [
            str(item).strip()
            for item in (obj.get("subqueries") or [])
            if str(item).strip()
        ]

        # Preserve the notebook fallback:
        # if decomposition fails, retrieve using the original query.
        if not subqueries:
            subqueries = [query]

        return subqueries[:4], meta

    def classify_subquery(
        self,
        subquery: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Classify one decomposed retrieval query and choose adaptive top-k.
        """

        subquery = str(subquery).strip()

        if not subquery:
            raise ValueError("Subquery cannot be empty.")

        raw, meta = self.generator.chat(
            messages=[
                {
                    "role": "system",
                    "content": SUBQUERY_DIFFICULTY_SYSTEM,
                },
                {
                    "role": "user",
                    "content": subquery,
                },
            ],
            max_tokens=450,
            temperature=0.0,
        )

        obj = extract_json_object(raw)

        difficulty = str(
            obj.get("difficulty", "mid")
        ).lower().strip()

        if difficulty not in {
            "easy",
            "mid",
            "hard",
        }:
            difficulty = "mid"

        retrieval_k = clamp_k(
            difficulty,
            obj.get("retrieval_k"),
        )

        plan = {
            "subquery": subquery,
            "difficulty": difficulty,
            "retrieval_k": retrieval_k,
            "short_reason": str(
                obj.get("short_reason") or ""
            ).strip(),
        }

        return plan, meta

    def build_plans(
        self,
        subqueries: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Build a retrieval plan for every decomposed subquery.

        This intentionally preserves the notebook behavior of classifying
        each subquery independently.
        """

        plans: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []

        for subquery in subqueries:
            plan, meta = self.classify_subquery(
                subquery
            )

            plans.append(plan)
            metas.append(meta)

        return plans, metas


# ----------------------------------------------------------------------
# Function-style wrappers matching the evaluation notebook
# ----------------------------------------------------------------------

def decompose_query(
    query: str,
    generator: OllamaGenerator,
    relationship_summary: str = "",
    clarification_context: str = "",
) -> tuple[list[str], dict[str, Any]]:
    """
    Notebook-compatible wrapper around DecompositionAgent.decompose().
    """

    agent = DecompositionAgent(generator)

    return agent.decompose(
        query=query,
        relationship_summary=relationship_summary,
        clarification_context=clarification_context,
    )


def classify_subquery_difficulty(
    subquery: str,
    generator: OllamaGenerator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Notebook-compatible wrapper around classify_subquery().
    """

    agent = DecompositionAgent(generator)

    return agent.classify_subquery(
        subquery
    )


def build_subquery_plans(
    subqueries: list[str],
    generator: OllamaGenerator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Notebook-compatible wrapper around build_plans().
    """

    agent = DecompositionAgent(generator)

    return agent.build_plans(
        subqueries
    )
