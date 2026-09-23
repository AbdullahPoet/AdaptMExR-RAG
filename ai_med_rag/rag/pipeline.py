"""
Central Medical RAG pipeline for the Streamlit application.

Save as:
    rag/pipeline.py

This revision connects Tavily to the UPDATED_NEED route.

Main flow:
    Query Analyzer
        ↓
    Route
        ├─ NORMAL
        │    → MedCPT retrieval → Ollama → postcheck
        │
        ├─ AMBIGUOUS / CONTRADICTION
        │    → clarification request
        │
        ├─ OLD_RELEVANT
        │    → relationship analysis
        │    → decomposition
        │    → MedCPT retrieval
        │    → Ollama
        │    → postcheck
        │
        ├─ UPDATED_NEED
        │    → Tavily search
        │    → Ollama synthesis with [W#] citations
        │    → postcheck
        │
        ├─ PRIVACY_LEAKAGE
        │    → fixed privacy response
        │
        └─ SEVERE_ER
             → emergency escalation

The model backend is Ollama.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from agents.decomposition import DecompositionAgent
from agents.postcheck import PostcheckAgent
from agents.query_analyzer import QueryAnalyzer
from agents.relationship import RelationshipAgent
from models.ollama_generator import OllamaGenerator
from rag.generator import RAGGenerator
from rag.retrieval_pipeline import RetrievalPipeline
from retrieval.medcpt_retriever import MedCPTRetriever
from web_search.tavily_search import TavilySearch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"


PRIVACY_BLOCK_RESPONSE = (
    "I don't have information regarding the query."
)

EMERGENCY_RESPONSE = (
    "This may require immediate medical attention. Please go to the nearest "
    "emergency department/hospital or call your local emergency services now. "
    "Do not delay care while waiting for an online answer."
)


UPDATED_NEED_SYSTEM = r"""
You are an evidence-grounded medical assistant answering a question that requires
current or recently updated information.

Use ONLY the supplied Tavily web evidence for current factual claims.

Rules:
1. Cite web evidence inline using the supplied IDs exactly, such as [W1] or [W2].
2. Never invent a web citation ID.
3. If the web evidence is incomplete, conflicting, or insufficient, say so.
4. Distinguish established guidance from preliminary, observational, or news-level evidence.
5. Use professional clinical language.
6. Do not claim to be the user's physician.
7. Do not reveal another person's private medical information.
8. If the available evidence supports urgent medical evaluation, state that clearly.
9. Keep the answer focused on the user's question.
"""


class MedicalRAGPipeline:
    """
    Main application pipeline.

    Parameters
    ----------
    ollama_model:
        Model selected by the user in Streamlit.
    ollama_base_url:
        Local Ollama API endpoint.
    tavily_api_key:
        User-supplied Tavily API key for UPDATED_NEED routes.
    vector_store_dir:
        Directory containing the existing MedCPT FAISS vector store.
    retrieval_device:
        Device for MedCPT models. CPU is the safest default because
        Ollama may already be using the GPU.
    """

    def __init__(
        self,
        ollama_model: str,
        ollama_base_url: str = "http://127.0.0.1:11434",
        tavily_api_key: str | None = None,
        vector_store_dir: str | Path = DEFAULT_VECTOR_DB_DIR,
        retrieval_device: str = "cpu",
    ) -> None:
        self.ollama_model = str(ollama_model).strip()
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.vector_store_dir = Path(vector_store_dir)
        self.tavily_api_key = (
            str(tavily_api_key).strip()
            if tavily_api_key
            else None
        )

        # Shared Ollama backend
        self.generator = OllamaGenerator(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
        )

        # Local medical retrieval stack
        self.retriever = MedCPTRetriever(
            vector_store_dir=self.vector_store_dir,
            device=retrieval_device,
        )

        self.retrieval_pipeline = RetrievalPipeline(
            retriever=self.retriever,
        )

        # Agents
        self.query_analyzer = QueryAnalyzer(self.generator)
        self.relationship_agent = RelationshipAgent(self.generator)
        self.decomposition_agent = DecompositionAgent(self.generator)

        self.rag_generator = RAGGenerator(
            generator=self.generator,
            retrieval_pipeline=self.retrieval_pipeline,
        )

        self.postcheck_agent = PostcheckAgent(
            generator=self.generator,
        )

        # Tavily is optional at construction time, but REQUIRED if
        # an UPDATED_NEED route is actually triggered.
        self.tavily = (
            TavilySearch(self.tavily_api_key)
            if self.tavily_api_key
            else None
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def ask(
        self,
        question: str,
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """
        Process one conversational medical question.

        Returns a dictionary containing at least:
            answer
            status
            category

        Extra audit fields are returned for debugging/research use.
        """

        question = str(question).strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        chat_history = chat_history or []

        effective_query, clarification_context = (
            self._resolve_conversational_query(
                question=question,
                chat_history=chat_history,
            )
        )

        analysis, analysis_meta = self.query_analyzer.analyze(
            query=effective_query,
            clarification_context=clarification_context,
        )

        result: dict[str, Any] = {
            "query": question,
            "effective_query": effective_query,
            "category": analysis["category"],
            "difficulty": analysis["difficulty"],
            "retrieval_k": analysis["retrieval_k"],
            "clean_retrieval_query": analysis["clean_retrieval_query"],
            "ignored_stale_context": analysis["ignored_stale_context"],
            "needs_clarification": analysis["needs_clarification"],
            "clarification_question": analysis["clarification_question"],
            "analyser_reason": analysis["short_reason"],
            "retrieved_chunks": [],
            "retrieved_context": "",
            "web_results": [],
            "web_context": "",
            "subqueries": [],
            "subquery_plans": [],
            "relationship_summary": "",
            "retrieval_focus": [],
            "agent_trace": [
                {
                    "node": "query_analyser",
                    **(analysis_meta or {}),
                }
            ],
        }

        category = analysis["category"]

        if category == "SEVERE_ER":
            result.update(
                {
                    "answer": EMERGENCY_RESPONSE,
                    "status": "emergency_escalation",
                }
            )
            result["agent_trace"].append(
                {"node": "severe_er_agent"}
            )
            return self._postcheck_fixed_route(result)

        if category == "PRIVACY_LEAKAGE":
            result.update(
                {
                    "answer": PRIVACY_BLOCK_RESPONSE,
                    "status": "blocked_third_party_privacy_request",
                }
            )
            result["agent_trace"].append(
                {"node": "privacy_agent"}
            )
            return self._postcheck_fixed_route(result)

        if category == "UPDATED_NEED":
            return self._answer_updated_need(
                result=result,
                effective_query=effective_query,
                chat_history=chat_history,
            )

        if category in {
            "AMBIGUOUS",
            "CONTRADICTION",
        }:
            clarification = (
                analysis["clarification_question"].strip()
                or (
                    "Your question contains ambiguity or conflicting "
                    "information that could change the medical answer. "
                    "Could you clarify the uncertain detail?"
                )
            )

            result.update(
                {
                    "answer": clarification,
                    "needs_clarification": True,
                    "clarification_question": clarification,
                    "status": "needs_clarification",
                }
            )

            result["agent_trace"].append(
                {"node": "clearness_agent"}
            )

            return self._postcheck_fixed_route(result)

        if category == "OLD_RELEVANT":
            return self._answer_old_relevant(
                result=result,
                effective_query=effective_query,
                clarification_context=clarification_context,
                chat_history=chat_history,
            )

        return self._answer_normal(
            result=result,
            effective_query=effective_query,
            analysis=analysis,
            chat_history=chat_history,
        )

    # ==================================================================
    # NORMAL
    # ==================================================================

    def _answer_normal(
        self,
        result: dict[str, Any],
        effective_query: str,
        analysis: dict[str, Any],
        chat_history: list[dict[str, str]],
    ) -> dict[str, Any]:

        difficulty = (
            analysis.get("difficulty")
            if analysis.get("difficulty") in {
                "easy",
                "mid",
                "hard",
            }
            else "mid"
        )

        retrieval_k = analysis.get("retrieval_k")

        if retrieval_k is None:
            retrieval_k = {
                "easy": 4,
                "mid": 8,
                "hard": 12,
            }[difficulty]

        retrieval_query = (
            analysis.get("clean_retrieval_query")
            or effective_query
        )

        rag = self.rag_generator.generate_rag_answer(
            query=retrieval_query,
            final_k=int(retrieval_k),
            subqueries=[retrieval_query],
            chat_history=chat_history,
        )

        result.update(
            {
                "answer": rag["answer"],
                "retrieved_chunks": rag["retrieved_chunks"],
                "retrieved_context": rag["retrieved_context"],
                "retrieval_audit": rag["retrieval_audit"],
                "status": "answered",
            }
        )

        result["agent_trace"].append(
            {
                "node": "qa_agent",
                **(rag.get("generation_meta") or {}),
                "retrieval_latency_s": rag.get(
                    "retrieval_latency_s",
                    0.0,
                ),
                "final_k": int(retrieval_k),
            }
        )

        return self._apply_local_postcheck(
            original_user_query=result["query"],
            result=result,
        )

    # ==================================================================
    # OLD_RELEVANT
    # ==================================================================

    def _answer_old_relevant(
        self,
        result: dict[str, Any],
        effective_query: str,
        clarification_context: str,
        chat_history: list[dict[str, str]],
    ) -> dict[str, Any]:

        relationship, rel_meta = (
            self.relationship_agent.analyze(
                query=effective_query,
                clarification_context=clarification_context,
            )
        )

        result["relationship_summary"] = (
            relationship["relationship_summary"]
        )
        result["retrieval_focus"] = (
            relationship["retrieval_focus"]
        )

        result["agent_trace"].append(
            {
                "node": "relationship_agent",
                **(rel_meta or {}),
            }
        )

        retrieval_query = (
            result.get("clean_retrieval_query")
            or effective_query
        )

        subqueries, decomp_meta = (
            self.decomposition_agent.decompose(
                query=retrieval_query,
                relationship_summary=relationship[
                    "relationship_summary"
                ],
                clarification_context=clarification_context,
            )
        )

        plans, difficulty_metas = (
            self.decomposition_agent.build_plans(
                subqueries=subqueries,
            )
        )

        result["subqueries"] = subqueries
        result["subquery_plans"] = plans

        result["agent_trace"].append(
            {
                "node": "decomposition_agent",
                **(decomp_meta or {}),
                "num_subqueries": len(subqueries),
            }
        )

        for plan, meta in zip(
            plans,
            difficulty_metas,
        ):
            result["agent_trace"].append(
                {
                    "node": "subquery_difficulty_agent",
                    **(meta or {}),
                    "subquery": plan["subquery"],
                    "difficulty": plan["difficulty"],
                    "retrieval_k": plan["retrieval_k"],
                }
            )

        rag = self.rag_generator.generate_decomposed_rag_answer(
            query=effective_query,
            subquery_plans=plans,
            chat_history=chat_history,
        )

        result.update(
            {
                "answer": rag["answer"],
                "retrieved_chunks": rag["retrieved_chunks"],
                "retrieved_context": rag["retrieved_context"],
                "retrieval_audit": rag["retrieval_audit"],
                "status": "answered",
            }
        )

        result["agent_trace"].append(
            {
                "node": "decomposed_rag_generator",
                **(rag.get("generation_meta") or {}),
                "retrieval_latency_s": rag.get(
                    "retrieval_latency_s",
                    0.0,
                ),
                "final_context_chunks": len(
                    rag["retrieved_chunks"]
                ),
            }
        )

        return self._apply_local_postcheck(
            original_user_query=result["query"],
            result=result,
        )

    # ==================================================================
    # UPDATED_NEED / Tavily
    # ==================================================================

    def _answer_updated_need(
        self,
        result: dict[str, Any],
        effective_query: str,
        chat_history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Search Tavily, then synthesize a current answer with [W#] citations.
        """

        if self.tavily is None:
            raise RuntimeError(
                "This query requires current information, but no Tavily "
                "API key is configured."
            )

        search_query = (
            result.get("clean_retrieval_query")
            or effective_query
        )

        web = self.tavily.search(
            query=search_query,
            max_results=5,
            search_depth="advanced",
        )

        web_results = web["results"]
        web_context = web["context"]

        if not web_results:
            result.update(
                {
                    "answer": (
                        "I could not retrieve current web evidence for this "
                        "question. Please try again later or rephrase the query."
                    ),
                    "status": "web_search_no_results",
                    "web_results": [],
                    "web_context": "",
                }
            )
            return self._postcheck_fixed_route(result)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": UPDATED_NEED_SYSTEM,
            }
        ]

        for item in chat_history:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()

            if role in {"user", "assistant"} and content:
                messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"User question:\n{effective_query}\n\n"
                    f"Current web evidence:\n{web_context}\n\n"
                    "Answer the user's question using the current evidence."
                ),
            }
        )

        answer, gen_meta = self.generator.chat(
            messages=messages,
            max_tokens=1536,
            temperature=0.0,
        )

        result.update(
            {
                "answer": answer,
                "status": "answered_with_current_web_evidence",
                "web_results": web_results,
                "web_context": web_context,
            }
        )

        result["agent_trace"].append(
            {
                "node": "tavily_search",
                "num_results": len(web_results),
            }
        )

        result["agent_trace"].append(
            {
                "node": "updated_need_generator",
                **(gen_meta or {}),
            }
        )

        return self._apply_web_postcheck(
            original_user_query=result["query"],
            result=result,
        )

    # ==================================================================
    # Postcheck helpers
    # ==================================================================

    def _apply_local_postcheck(
        self,
        original_user_query: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:

        chunks = result.get("retrieved_chunks") or []

        if not chunks:
            return self._postcheck_fixed_route(result)

        valid_ids = [
            chunk["context_id"]
            for chunk in chunks
        ]

        return self._run_postcheck(
            original_user_query=original_user_query,
            result=result,
            evidence_context=result.get(
                "retrieved_context",
                "",
            ),
            valid_ids=valid_ids,
        )

    def _apply_web_postcheck(
        self,
        original_user_query: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:

        web_results = result.get("web_results") or []

        if not web_results:
            return self._postcheck_fixed_route(result)

        valid_ids = [
            item["web_id"]
            for item in web_results
            if item.get("web_id")
        ]

        return self._run_postcheck(
            original_user_query=original_user_query,
            result=result,
            evidence_context=result.get(
                "web_context",
                "",
            ),
            valid_ids=valid_ids,
        )

    def _run_postcheck(
        self,
        original_user_query: str,
        result: dict[str, Any],
        evidence_context: str,
        valid_ids: list[str],
    ) -> dict[str, Any]:

        qc = self.postcheck_agent.evaluate_and_repair(
            query=original_user_query,
            answer=result.get("answer", ""),
            evidence_context=evidence_context,
            valid_ids=valid_ids,
        )

        result["pre_repair_answer"] = (
            qc["pre_repair_answer"]
            if qc["repaired"]
            else ""
        )

        result["answer"] = qc["answer"]
        result["postcheck"] = qc["postcheck"]
        result["repaired"] = qc["repaired"]

        result["agent_trace"].append(
            {
                "node": "postcheck_agent",
                **(qc.get("postcheck_meta") or {}),
            }
        )

        if qc["repaired"]:
            result["agent_trace"].append(
                {
                    "node": "repair_agent",
                    **(qc.get("repair_meta") or {}),
                }
            )

        return result

    @staticmethod
    def _postcheck_fixed_route(
        result: dict[str, Any],
    ) -> dict[str, Any]:

        result["postcheck"] = {
            "professionalism_ok": True,
            "third_party_privacy_leak": False,
            "groundedness": None,
            "cited_ids_valid": True,
            "unsupported_claims": [],
            "quality_note": (
                "Fixed safety/privacy/clarification route."
            ),
        }

        result["repaired"] = False
        return result

    # ==================================================================
    # Conversational clarification support
    # ==================================================================

    @staticmethod
    def _resolve_conversational_query(
        question: str,
        chat_history: list[dict[str, str]],
    ) -> tuple[str, str]:

        if len(chat_history) < 2:
            return question, ""

        last_assistant_index: int | None = None

        for index in range(
            len(chat_history) - 1,
            -1,
            -1,
        ):
            if chat_history[index].get("role") == "assistant":
                last_assistant_index = index
                break

        if last_assistant_index is None:
            return question, ""

        assistant_text = str(
            chat_history[last_assistant_index].get(
                "content",
                "",
            )
        ).strip()

        if not MedicalRAGPipeline._looks_like_clarification(
            assistant_text
        ):
            return question, ""

        previous_user_text = ""

        for index in range(
            last_assistant_index - 1,
            -1,
            -1,
        ):
            if chat_history[index].get("role") == "user":
                previous_user_text = str(
                    chat_history[index].get(
                        "content",
                        "",
                    )
                ).strip()
                break

        if not previous_user_text:
            return question, ""

        effective_query = (
            f"{previous_user_text}\n\n"
            f"User clarification: {question}"
        )

        return effective_query, question

    @staticmethod
    def _looks_like_clarification(
        assistant_text: str,
    ) -> bool:

        text = assistant_text.lower()

        markers = (
            "could you clarify",
            "can you clarify",
            "please clarify",
            "which ",
            "what exactly",
            "uncertain detail",
            "conflicting",
            "ambiguity",
        )

        return (
            "?" in assistant_text
            and any(marker in text for marker in markers)
        )

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def info(self) -> dict[str, Any]:
        return {
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
            "tavily_enabled": self.tavily is not None,
            "vector_store": self.retriever.info(),
        }
