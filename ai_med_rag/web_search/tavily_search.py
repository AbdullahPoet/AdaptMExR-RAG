"""
Tavily web-search client for the Medical RAG application.

Save as:
    web_search/tavily_search.py

Purpose:
- Use Tavily only for queries routed as UPDATED_NEED.
- Return structured, citation-ready web evidence.
- Keep the Tavily API key in memory only.
"""

from __future__ import annotations

from typing import Any
import requests


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilySearch:
    """
    Small Tavily search wrapper for current/recent medical information.

    Parameters
    ----------
    api_key:
        User-supplied Tavily API key.
    timeout:
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 20,
    ) -> None:
        api_key = str(api_key).strip()

        if not api_key:
            raise ValueError("Tavily API key cannot be empty.")

        self.api_key = api_key
        self.timeout = int(timeout)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
    ) -> dict[str, Any]:
        """
        Run a Tavily search and return normalized evidence.

        Returns
        -------
        {
            "query": "...",
            "answer": "...",
            "results": [
                {
                    "web_id": "W1",
                    "title": "...",
                    "url": "...",
                    "content": "...",
                    "score": ...
                }
            ],
            "context": "... citation-ready text ..."
        }
        """

        query = str(query).strip()

        if not query:
            raise ValueError("Search query cannot be empty.")

        max_results = max(1, min(int(max_results), 10))

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
        }

        try:
            response = requests.post(
                TAVILY_SEARCH_URL,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Tavily. Check the internet connection."
            ) from exc

        except requests.Timeout as exc:
            raise RuntimeError(
                f"Tavily search timed out after {self.timeout} seconds."
            ) from exc

        except requests.HTTPError as exc:
            detail = ""

            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]

            raise RuntimeError(
                f"Tavily search failed with HTTP "
                f"{response.status_code}: {detail}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Tavily returned invalid JSON."
            ) from exc

        normalized_results: list[dict[str, Any]] = []

        for index, item in enumerate(
            data.get("results", []),
            start=1,
        ):
            normalized_results.append(
                {
                    "web_id": f"W{index}",
                    "title": str(
                        item.get("title") or ""
                    ).strip(),
                    "url": str(
                        item.get("url") or ""
                    ).strip(),
                    "content": str(
                        item.get("content") or ""
                    ).strip(),
                    "score": item.get("score"),
                }
            )

        context = self.results_to_context(
            normalized_results
        )

        return {
            "query": query,
            "answer": str(
                data.get("answer") or ""
            ).strip(),
            "results": normalized_results,
            "context": context,
        }

    @staticmethod
    def results_to_context(
        results: list[dict[str, Any]],
    ) -> str:
        """
        Convert Tavily results into citation-ready [W1], [W2], ... evidence.
        """

        blocks: list[str] = []

        for item in results:
            lines = [
                f"[{item['web_id']}]",
                f"Title: {item.get('title', '')}",
                f"URL: {item.get('url', '')}",
                "Evidence:",
                str(item.get("content", "")),
            ]

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    @staticmethod
    def valid_ids(
        results: list[dict[str, Any]],
    ) -> list[str]:
        """Return all valid web citation IDs."""
        return [
            str(item["web_id"])
            for item in results
            if item.get("web_id")
        ]
