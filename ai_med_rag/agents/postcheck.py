"""
Post-generation quality-control agent for the Medical RAG application.

Save as:
    agents/postcheck.py

This module is adapted directly from the evaluation notebook's:
- POSTCHECK_SYSTEM
- REPAIR_SYSTEM
- postcheck_answer(...)
- maybe_repair_answer(...)

The only backend change is:
    vllm_chat(...) -> OllamaGenerator.chat(...)
"""

from __future__ import annotations

from typing import Any

from agents.query_analyzer import extract_json_object
from models.ollama_generator import OllamaGenerator


AUTO_REPAIR = True
GROUNDEDNESS_REPAIR_THRESHOLD = 0.75
GENERATOR_MAX_TOKENS = 1500


POSTCHECK_SYSTEM = r"""
You are a strict quality-control evaluator for a medical RAG answer.

Evaluate:
1) professionalism_ok:
   Professional clinical communication; clear, respectful, appropriately cautious;
   does not pretend to be the user's physician.

2) third_party_privacy_leak:
   True only if the answer reveals or infers another person's private medical
   information inappropriately. The user's own health information is not
   automatically a third-party privacy leak.

3) groundedness:
   A number from 0.0 to 1.0 representing how well the answer's factual medical
   claims are supported by the supplied evidence.

4) cited_ids_valid:
   Every [C#] or [W#] citation appearing in the answer exists in the supplied
   valid ID list.

Return ONLY JSON:
{
  "professionalism_ok": true,
  "third_party_privacy_leak": false,
  "groundedness": 0.0,
  "cited_ids_valid": true,
  "unsupported_claims": ["..."],
  "quality_note": "short audit note"
}
"""


REPAIR_SYSTEM = r"""
Rewrite the medical answer so it:
- remains concise and professional,
- contains no third-party private health information,
- makes only evidence-supported medical claims,
- cites only the valid evidence IDs supplied,
- states insufficiency/uncertainty instead of guessing,
- preserves appropriate urgent-care advice if supported.

Return only the repaired answer, no commentary.
"""


class PostcheckAgent:
    """
    Run quality control on a generated medical RAG answer and optionally
    repair it once when it fails the configured checks.
    """

    def __init__(
        self,
        generator: OllamaGenerator,
        auto_repair: bool = AUTO_REPAIR,
        groundedness_threshold: float = GROUNDEDNESS_REPAIR_THRESHOLD,
        repair_max_tokens: int = GENERATOR_MAX_TOKENS,
    ) -> None:
        self.generator = generator
        self.auto_repair = bool(auto_repair)
        self.groundedness_threshold = float(groundedness_threshold)
        self.repair_max_tokens = int(repair_max_tokens)

    # ------------------------------------------------------------------
    # Quality check
    # ------------------------------------------------------------------

    def check(
        self,
        query: str,
        answer: str,
        evidence_context: str,
        valid_ids: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Evaluate professionalism, privacy, groundedness, and citation validity.
        """

        raw, meta = self.generator.chat(
            messages=[
                {
                    "role": "system",
                    "content": POSTCHECK_SYSTEM,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Answer:\n{answer}\n\n"
                        f"Valid evidence IDs: {valid_ids}\n\n"
                        f"Evidence:\n{evidence_context}"
                    ),
                },
            ],
            max_tokens=400,
            temperature=0.0,
        )

        obj = extract_json_object(raw)

        unsupported_claims = obj.get("unsupported_claims") or []

        if not isinstance(unsupported_claims, list):
            unsupported_claims = [unsupported_claims]

        try:
            groundedness = float(
                obj.get("groundedness", 0.0)
            )
        except (TypeError, ValueError):
            groundedness = 0.0

        groundedness = max(
            0.0,
            min(1.0, groundedness),
        )

        result = {
            "professionalism_ok": bool(
                obj.get("professionalism_ok", False)
            ),
            "third_party_privacy_leak": bool(
                obj.get("third_party_privacy_leak", False)
            ),
            "groundedness": groundedness,
            "cited_ids_valid": bool(
                obj.get("cited_ids_valid", False)
            ),
            "unsupported_claims": [
                str(item).strip()
                for item in unsupported_claims
                if str(item).strip()
            ],
            "quality_note": str(
                obj.get("quality_note") or ""
            ).strip(),
        }

        return result, meta

    # ------------------------------------------------------------------
    # Repair decision
    # ------------------------------------------------------------------

    def should_repair(
        self,
        check: dict[str, Any],
    ) -> bool:
        """
        Match the evaluation notebook's one-pass repair trigger.
        """

        return bool(
            not check.get("professionalism_ok", False)
            or check.get("third_party_privacy_leak", False)
            or not check.get("cited_ids_valid", False)
            or float(check.get("groundedness", 0.0))
            < self.groundedness_threshold
        )

    # ------------------------------------------------------------------
    # Optional one-pass repair
    # ------------------------------------------------------------------

    def maybe_repair(
        self,
        query: str,
        answer: str,
        evidence_context: str,
        valid_ids: list[str],
        check: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None, bool]:
        """
        Return:
            repaired_or_original_answer,
            repair_generation_metadata_or_none,
            repaired_flag
        """

        fail = self.should_repair(check)

        if not self.auto_repair or not fail:
            return answer, None, False

        repaired, meta = self.generator.chat(
            messages=[
                {
                    "role": "system",
                    "content": REPAIR_SYSTEM,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Original answer:\n{answer}\n\n"
                        f"Valid evidence IDs: {valid_ids}\n\n"
                        f"Evidence:\n{evidence_context}"
                    ),
                },
            ],
            max_tokens=self.repair_max_tokens,
            temperature=0.0,
        )

        return repaired, meta, True

    # ------------------------------------------------------------------
    # Full post-generation QC convenience method
    # ------------------------------------------------------------------

    def evaluate_and_repair(
        self,
        query: str,
        answer: str,
        evidence_context: str,
        valid_ids: list[str],
    ) -> dict[str, Any]:
        """
        Run the original notebook's postcheck + optional one-pass repair.

        This does not perform a second postcheck after repair because the
        evaluation notebook performs a single repair pass.
        """

        check, check_meta = self.check(
            query=query,
            answer=answer,
            evidence_context=evidence_context,
            valid_ids=valid_ids,
        )

        final_answer, repair_meta, repaired = self.maybe_repair(
            query=query,
            answer=answer,
            evidence_context=evidence_context,
            valid_ids=valid_ids,
            check=check,
        )

        return {
            "answer": final_answer,
            "pre_repair_answer": answer,
            "postcheck": check,
            "repaired": repaired,
            "postcheck_meta": check_meta,
            "repair_meta": repair_meta,
        }


# ----------------------------------------------------------------------
# Function-style wrappers matching the notebook
# ----------------------------------------------------------------------

def postcheck_answer(
    query: str,
    answer: str,
    evidence_context: str,
    valid_ids: list[str],
    generator: OllamaGenerator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Notebook-compatible wrapper for postcheck_answer(...).
    """

    agent = PostcheckAgent(generator)

    return agent.check(
        query=query,
        answer=answer,
        evidence_context=evidence_context,
        valid_ids=valid_ids,
    )


def maybe_repair_answer(
    query: str,
    answer: str,
    evidence_context: str,
    valid_ids: list[str],
    check: dict[str, Any],
    generator: OllamaGenerator,
    auto_repair: bool = AUTO_REPAIR,
    groundedness_threshold: float = GROUNDEDNESS_REPAIR_THRESHOLD,
    repair_max_tokens: int = GENERATOR_MAX_TOKENS,
) -> tuple[str, dict[str, Any] | None]:
    """
    Notebook-compatible wrapper for maybe_repair_answer(...).

    Returns the same two-item shape as the original notebook:
        answer, meta
    """

    agent = PostcheckAgent(
        generator=generator,
        auto_repair=auto_repair,
        groundedness_threshold=groundedness_threshold,
        repair_max_tokens=repair_max_tokens,
    )

    final_answer, meta, _ = agent.maybe_repair(
        query=query,
        answer=answer,
        evidence_context=evidence_context,
        valid_ids=valid_ids,
        check=check,
    )

    return final_answer, meta
