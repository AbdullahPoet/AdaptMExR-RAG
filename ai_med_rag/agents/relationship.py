"""
Relationship Agent for the Medical RAG application.

Save as:
    agents/relationship.py

This is adapted directly from the evaluation notebook's
`relationship_analysis(...)` function, replacing `vllm_chat(...)`
with `OllamaGenerator`.

Purpose:
- Use the current medical problem plus relevant history/clarification.
- Identify clinically plausible relationships that should guide retrieval.
- Avoid diagnosing or assuming causation.
"""

from __future__ import annotations

from typing import Any

from agents.query_analyzer import extract_json_object
from models.ollama_generator import OllamaGenerator


RELATIONSHIP_SYSTEM = r"""
You are a medical query-relationship analyser.
Do not diagnose and do not give a final answer.

Given the user's current problem plus relevant history/clarification, identify only
the clinically plausible relationship(s) that should guide retrieval.
Do not assume causation.

Return ONLY JSON:
{
  "relationship_summary": "2-4 concise sentences",
  "retrieval_focus": ["short concept 1", "short concept 2", "..."]
}
"""


class RelationshipAgent:
    """
    Analyze clinically relevant relationships before decomposition/retrieval.
    """

    def __init__(self, generator: OllamaGenerator) -> None:
        self.generator = generator

    def analyze(
        self,
        query: str,
        clarification_context: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Return:
            (
                {
                    "relationship_summary": str,
                    "retrieval_focus": list[str],
                },
                generation_metadata,
            )
        """

        query = str(query).strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        raw, meta = self.generator.chat(
            messages=[
                {
                    "role": "system",
                    "content": RELATIONSHIP_SYSTEM,
                },
                {
                    "role": "user",
                    "content": (
                        f"Original query:\n{query}\n\n"
                        f"Clarification/history context:\n"
                        f"{clarification_context}"
                    ),
                },
            ],
            max_tokens=700,
            temperature=0.0,
        )

        obj = extract_json_object(raw)

        retrieval_focus = obj.get("retrieval_focus") or []

        if not isinstance(retrieval_focus, list):
            retrieval_focus = [retrieval_focus]

        retrieval_focus = [
            str(item).strip()
            for item in retrieval_focus
            if str(item).strip()
        ]

        result = {
            "relationship_summary": str(
                obj.get("relationship_summary") or ""
            ).strip(),
            "retrieval_focus": retrieval_focus,
        }

        return result, meta


def relationship_analysis(
    query: str,
    generator: OllamaGenerator,
    clarification_context: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Function-style wrapper matching the original evaluation notebook.

    Example:
        result, meta = relationship_analysis(
            query="I have chest discomfort and a history of GERD.",
            generator=generator,
            clarification_context="Symptoms started after eating.",
        )
    """
    agent = RelationshipAgent(generator)

    return agent.analyze(
        query=query,
        clarification_context=clarification_context,
    )
