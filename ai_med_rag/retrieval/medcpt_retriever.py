"""
MedCPT hybrid retriever for the Medical RAG application.

Save as:
    retrieval/medcpt_retriever.py

This module is extracted from the evaluation notebook's retrieval pipeline:

    MedCPT query encoder
        +
    FAISS dense retrieval
        +
    BM25 sparse retrieval
        +
    Reciprocal Rank Fusion (RRF)
        +
    MedCPT Cross-Encoder reranking

Expected vector-store files:
    vector_db/medquad_medcpt.faiss
    vector_db/medquad_metadata.pkl
    vector_db/medquad_vectorstore_config.json
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
RERANKER_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"

RRF_K = 60


class MedCPTRetriever:
    """
    Hybrid medical retriever using the same retrieval design as the
    evaluation notebook.

    Parameters
    ----------
    vector_store_dir:
        Directory containing the FAISS index, metadata pickle, and config JSON.
    device:
        Retrieval device. Defaults to CPU so Ollama can use the GPU independently.
        You may explicitly pass "cuda" if desired.
    """

    def __init__(
        self,
        vector_store_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        self.vector_store_dir = Path(vector_store_dir)
        self.device = device

        self.index_path = (
            self.vector_store_dir / "medquad_medcpt.faiss"
        )
        self.metadata_path = (
            self.vector_store_dir / "medquad_metadata.pkl"
        )
        self.config_path = (
            self.vector_store_dir / "medquad_vectorstore_config.json"
        )

        self._validate_files()
        self._load_vector_store()
        self._build_bm25()
        self._load_medcpt_models()
        self._validate_dimensions()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _validate_files(self) -> None:
        required = [
            self.index_path,
            self.metadata_path,
        ]

        missing = [str(path) for path in required if not path.exists()]

        if missing:
            raise FileNotFoundError(
                "Missing required vector-store file(s):\n"
                + "\n".join(missing)
            )

    def _load_vector_store(self) -> None:
        """Load the existing FAISS index and metadata."""
        self.faiss_index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "rb") as f:
            self.vector_metadata = pickle.load(f)

        if self.faiss_index.ntotal != len(self.vector_metadata):
            raise ValueError(
                "FAISS/metadata mismatch: "
                f"{self.faiss_index.ntotal} vectors vs "
                f"{len(self.vector_metadata)} metadata records."
            )

        self.vector_config: dict[str, Any] = {}

        if self.config_path.exists():
            try:
                self.vector_config = json.loads(
                    self.config_path.read_text(encoding="utf-8")
                )
            except Exception:
                self.vector_config = {}

    def _build_bm25(self) -> None:
        """Build sparse BM25 retrieval over the existing chunk metadata."""
        corpus = [
            self._lexical_tokens(item.get("text", ""))
            for item in self.vector_metadata
        ]

        self.bm25 = BM25Okapi(corpus)

    def _load_medcpt_models(self) -> None:
        """Load MedCPT query encoder and reranker."""
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            QUERY_MODEL_NAME
        )

        self.query_model = AutoModel.from_pretrained(
            QUERY_MODEL_NAME
        ).to(self.device)

        self.query_model.eval()

        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_NAME
        )

        self.rerank_model = (
            AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL_NAME
            ).to(self.device)
        )

        self.rerank_model.eval()

    def _validate_dimensions(self) -> None:
        hidden_size = int(self.query_model.config.hidden_size)

        if hidden_size != self.faiss_index.d:
            raise ValueError(
                "MedCPT query encoder dimension does not match FAISS index. "
                f"Query encoder: {hidden_size}; "
                f"FAISS index: {self.faiss_index.d}"
            )

    # ------------------------------------------------------------------
    # Tokenization / embeddings
    # ------------------------------------------------------------------

    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        return re.findall(
            r"[a-z0-9]+(?:[-'][a-z0-9]+)?",
            str(text).lower(),
        )

    @torch.inference_mode()
    def embed_query(self, question: str) -> np.ndarray:
        """
        Encode a query using MedCPT Query Encoder.

        This preserves the evaluation notebook's CLS-token embedding and
        L2 normalization behavior.
        """
        encoded = self.query_tokenizer(
            [str(question)],
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        output = self.query_model(**encoded)

        embedding = (
            output.last_hidden_state[:, 0, :]
            .float()
            .cpu()
            .numpy()
            .astype("float32")
        )

        faiss.normalize_L2(embedding)

        return embedding

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def _dense_candidates(
        self,
        query: str,
        n: int,
    ) -> list[tuple[int, int, float]]:
        qvec = self.embed_query(query)

        scores, indices = self.faiss_index.search(
            qvec,
            min(n, self.faiss_index.ntotal),
        )

        results = []

        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        ):
            if idx >= 0:
                results.append(
                    (
                        int(idx),
                        rank,
                        float(score),
                    )
                )

        return results

    def _bm25_candidates(
        self,
        query: str,
        n: int,
    ) -> list[tuple[int, int, float]]:
        scores = np.asarray(
            self.bm25.get_scores(
                self._lexical_tokens(query)
            ),
            dtype=float,
        )

        n = min(n, len(scores))

        if n == 0:
            return []

        top_indices = np.argpartition(
            -scores,
            n - 1,
        )[:n]

        top_indices = top_indices[
            np.argsort(-scores[top_indices])
        ]

        return [
            (
                int(idx),
                rank,
                float(scores[idx]),
            )
            for rank, idx in enumerate(
                top_indices,
                start=1,
            )
        ]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def rrf_fuse(
        dense: list[tuple[int, int, float]],
        sparse: list[tuple[int, int, float]],
        rrf_k: int = RRF_K,
    ) -> list[dict[str, Any]]:
        fused: dict[int, dict[str, Any]] = {}

        for source_name, rows in [
            ("dense", dense),
            ("bm25", sparse),
        ]:
            for idx, rank, raw_score in rows:
                item = fused.setdefault(
                    idx,
                    {
                        "faiss_id": idx,
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "bm25_rank": None,
                        "dense_score": None,
                        "bm25_score": None,
                    },
                )

                item["rrf_score"] += (
                    1.0 / (rrf_k + rank)
                )

                item[f"{source_name}_rank"] = rank
                item[f"{source_name}_score"] = raw_score

        return sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # MedCPT reranking
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def rerank_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pairs = []

        for candidate in candidates:
            metadata = self.vector_metadata[
                candidate["faiss_id"]
            ]

            pairs.append(
                (
                    str(query),
                    str(metadata.get("text", "")),
                )
            )

        scores: list[float] = []
        batch_size = 16

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]

            questions = [pair[0] for pair in batch]
            documents = [pair[1] for pair in batch]

            encoded = self.rerank_tokenizer(
                questions,
                documents,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            logits = self.rerank_model(
                **encoded
            ).logits

            if (
                logits.ndim == 2
                and logits.shape[1] == 1
            ):
                values = logits[:, 0]
            else:
                values = logits.squeeze()

            scores.extend(
                values
                .detach()
                .float()
                .cpu()
                .tolist()
            )

        enriched = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            metadata = dict(
                self.vector_metadata[
                    candidate["faiss_id"]
                ]
            )

            enriched.append(
                {
                    **candidate,
                    "rerank_score": float(score),
                    "row_id": metadata.get("row_id"),
                    "chunk_id": metadata.get("chunk_id"),
                    "question": metadata.get("question"),
                    "focus_area": metadata.get("focus_area"),
                    "question_type": metadata.get(
                        "question_type"
                    ),
                    "source": metadata.get("source"),
                    "text": metadata.get("text", ""),
                }
            )

        enriched.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        top = enriched[:top_k]

        for rank, item in enumerate(
            top,
            start=1,
        ):
            item["context_id"] = f"C{rank}"
            item["final_rank"] = rank

        return top

    # ------------------------------------------------------------------
    # Main retrieval method
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        final_k: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Run hybrid MedCPT + BM25 retrieval and MedCPT reranking.

        Returns
        -------
        chunks:
            Final reranked evidence chunks.
        latency_s:
            Retrieval latency in seconds.
        """
        if not query or not str(query).strip():
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        final_k = max(
            1,
            min(
                int(final_k),
                self.faiss_index.ntotal,
            ),
        )

        started = time.perf_counter()

        candidate_k = min(
            self.faiss_index.ntotal,
            max(
                24,
                final_k * 3,
            ),
        )

        dense = self._dense_candidates(
            query,
            candidate_k,
        )

        sparse = self._bm25_candidates(
            query,
            candidate_k,
        )

        fused = self.rrf_fuse(
            dense,
            sparse,
        )

        fused_pool = fused[
            :min(
                len(fused),
                max(
                    30,
                    final_k * 3,
                ),
            )
        ]

        top_chunks = self.rerank_candidates(
            query=query,
            candidates=fused_pool,
            top_k=final_k,
        )

        latency = (
            time.perf_counter()
            - started
        )

        return top_chunks, latency

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    @staticmethod
    def chunks_to_context(
        chunks: list[dict[str, Any]],
    ) -> str:
        """
        Convert retrieved chunks to the citation-ready context format used
        in the evaluation pipeline.
        """
        blocks = []

        for chunk in chunks:
            header = [
                f"[{chunk['context_id']}]",
                f"Chunk ID: {chunk.get('chunk_id')}",
                (
                    "Reranker score: "
                    f"{chunk.get('rerank_score', 0):.4f}"
                ),
            ]

            if chunk.get("source"):
                header.append(
                    f"Source: {chunk['source']}"
                )

            if chunk.get("focus_area"):
                header.append(
                    f"Focus area: {chunk['focus_area']}"
                )

            header.append("Evidence:")
            header.append(
                str(chunk.get("text", ""))
            )

            blocks.append(
                "\n".join(header)
            )

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Return useful diagnostics for the app/debugging."""
        return {
            "vector_store_dir": str(
                self.vector_store_dir
            ),
            "vectors": int(
                self.faiss_index.ntotal
            ),
            "dimension": int(
                self.faiss_index.d
            ),
            "metadata_records": int(
                len(self.vector_metadata)
            ),
            "query_model": QUERY_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "device": self.device,
            "vector_config": self.vector_config,
        }
