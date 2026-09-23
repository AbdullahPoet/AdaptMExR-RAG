"""
Query Analyser Agent for the Medical RAG application.

Save as:
    agents/query_analyzer.py

This is adapted directly from the evaluation notebook's
"Query Analyser Agent", replacing `vllm_chat(...)` with `OllamaGenerator`.

Responsibilities:
- Route a medical query into one primary category.
- Detect ambiguity / contradiction / emergency / privacy / recency needs.
- Estimate NORMAL-query difficulty.
- Choose adaptive retrieval_k.
- Remove unrelated stale context from the retrieval query.
- Return a clarification question when needed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from models.ollama_generator import OllamaGenerator


QUERY_CATEGORIES = [
    "NORMAL",
    "AMBIGUOUS",
    "CONTRADICTION",
    "OLD_RELEVANT",
    "UPDATED_NEED",
    "PRIVACY_LEAKAGE",
    "SEVERE_ER",
]

K_RANGES = {
    "easy": (3, 4),
    "mid": (5, 8),
    "hard": (9, 15),
}

DEFAULT_K = {
    "easy": 4,
    "mid": 8,
    "hard": 12,
}


ANALYSER_SYSTEM = r"""
You are the routing classifier for a medical RAG system.
Do NOT answer the medical question.

Classify the user's query into exactly ONE primary category:

NORMAL:
Ordinary medical QA answerable from the local medical knowledge base.
Also use NORMAL when the query contains old information that is clearly
unrelated to the current question. Put that distractor in ignored_stale_context.

AMBIGUOUS:
The query is clinically ambiguous and a meaningful answer requires clarification.

CONTRADICTION:
The user's statements materially conflict with each other and clarification is needed.

OLD_RELEVANT:
The user gives older medical history that could materially affect interpretation,
diagnosis, treatment, medication safety, or risk for the current problem.

UPDATED_NEED:
The user explicitly asks for current/recent/latest information, a current guideline,
recent outbreak, current recommendation, or another fact that requires live web evidence.

PRIVACY_LEAKAGE:
The user is asking to obtain or reveal ANOTHER person's private medical/health information.
Do NOT label a query Privacy Leakage merely because the user includes their OWN PHI,
sexual history, ED, STI history, intercourse date, or other sensitive information.

SEVERE_ER:
The prompt itself contains signs that may require immediate emergency assessment,
such as ongoing uncontrolled bleeding, stroke-like deficits, severe breathing difficulty,
possible anaphylaxis, severe chest pain, major trauma, severe overdose, seizure with
persistent impairment, or another clearly time-critical emergency.

PRIORITY when multiple categories overlap:
1) SEVERE_ER if the current user/patient has an emergency.
2) PRIVACY_LEAKAGE if the main request is to reveal another person's private health data.
3) UPDATED_NEED.
4) CONTRADICTION.
5) AMBIGUOUS.
6) OLD_RELEVANT.
7) NORMAL.

For NORMAL, also estimate difficulty:
easy = direct single-fact/simple medical QA
mid = moderate synthesis/comparison or several facts
hard = complex/multi-part/multi-step clinical reasoning

Choose retrieval_k inside:
easy: 3-4
mid: 5-8
hard: 9-15

Return ONLY JSON with:
{
  "category": "...",
  "difficulty": "easy|mid|hard|null",
  "retrieval_k": integer_or_null,
  "clean_retrieval_query": "...",
  "ignored_stale_context": ["..."],
  "needs_clarification": true_or_false,
  "clarification_question": "... or empty",
  "short_reason": "one short audit-friendly reason, no chain-of-thought"
}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Robustly extract one JSON object from an LLM response.

    Ollama models occasionally wrap JSON in markdown fences or add small
    amounts of surrounding text. This helper keeps the application resilient
    without changing the agent's requested output format.
    """
    if not text:
        return {}

    cleaned = str(text).strip()

    # Remove common markdown code fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced-looking JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = cleaned[start:end + 1]

    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def clamp_k(difficulty: str | None, proposed: Any) -> int:
    """Keep retrieval_k inside the notebook's configured difficulty range."""
    difficulty = difficulty if difficulty in K_RANGES else "mid"
    lo, hi = K_RANGES[difficulty]

    try:
        k = int(proposed)
    except (TypeError, ValueError):
        k = DEFAULT_K[difficulty]

    return max(lo, min(hi, k))


class QueryAnalyzer:
    """
    Ollama-backed query-routing agent.

    Parameters
    ----------
    generator:
        Initialized OllamaGenerator using the model selected by the user.
    """

    def __init__(self, generator: OllamaGenerator) -> None:
        self.generator = generator

    def analyze(
        self,
        query: str,
        clarification_context: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Analyze a user query and return:
            (routing_result, generation_metadata)
        """
        query = str(query).strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        content = query

        if clarification_context.strip():
            content += (
                "\n\nAdditional clarification supplied by the user:\n"
                + clarification_context.strip()
            )

        raw, meta = self.generator.chat(
            messages=[
                {
                    "role": "system",
                    "content": ANALYSER_SYSTEM,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            max_tokens=1024,
            temperature=0.0,
        )

        obj = extract_json_object(raw)

        category = str(
            obj.get("category", "NORMAL")
        ).upper().strip()

        if category not in QUERY_CATEGORIES:
            category = "NORMAL"

        difficulty = obj.get("difficulty")

        if isinstance(difficulty, str):
            difficulty = difficulty.lower().strip()

        if difficulty not in {"easy", "mid", "hard"}:
            difficulty = (
                "mid"
                if category == "NORMAL"
                else None
            )

        retrieval_k = (
            clamp_k(
                difficulty,
                obj.get("retrieval_k"),
            )
            if category == "NORMAL"
            else None
        )

        ignored_stale_context = obj.get(
            "ignored_stale_context"
        ) or []

        if not isinstance(
            ignored_stale_context,
            list,
        ):
            ignored_stale_context = [
                str(ignored_stale_context)
            ]

        needs_clarification = bool(
            obj.get(
                "needs_clarification",
                category in {
                    "AMBIGUOUS",
                    "CONTRADICTION",
                },
            )
        )

        clarification_question = str(
            obj.get(
                "clarification_question"
            ) or ""
        ).strip()

        # Defensive fallback so ambiguous routes never dead-end.
        if (
            category
            in {
                "AMBIGUOUS",
                "CONTRADICTION",
            }
            and not clarification_question
        ):
            clarification_question = (
                "Could you clarify the conflicting or uncertain medical "
                "detail so I can answer more accurately?"
            )

        result = {
            "category": category,
            "difficulty": difficulty,
            "retrieval_k": retrieval_k,
            "clean_retrieval_query": str(
                obj.get(
                    "clean_retrieval_query"
                )
                or query
            ).strip(),
            "ignored_stale_context": [
                str(item).strip()
                for item in ignored_stale_context
                if str(item).strip()
            ],
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            "short_reason": str(
                obj.get(
                    "short_reason"
                )
                or ""
            ).strip(),
        }

        return result, meta


# ----------------------------------------------------------------------
# Optional function-style wrapper
# ----------------------------------------------------------------------

def analyse_query_text(
    query: str,
    generator: OllamaGenerator,
    clarification_context: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Function wrapper matching the evaluation notebook's naming style.

    Example:
        result, meta = analyse_query_text(
            query="What causes migraine?",
            generator=generator,
        )
    """
    analyzer = QueryAnalyzer(generator)

    return analyzer.analyze(
        query=query,
        clarification_context=clarification_context,
    )
