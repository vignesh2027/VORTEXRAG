"""
VORTEXRAG LangChain Integration
================================
Drop-in LangChain retriever that wraps the full 7-layer VORTEXRAG pipeline.

Usage:
    from vortexrag.integrations.langchain_retriever import VortexRAGRetriever

    retriever = VortexRAGRetriever(domain="medical", top_k=5)
    retriever.add_documents(["text chunk 1", "text chunk 2", ...])

    # Use directly
    docs = retriever.invoke("What causes sepsis?")

    # Async
    docs = await retriever.ainvoke("What causes sepsis?")

    # Batch
    results = retriever.batch(["query 1", "query 2", "query 3"])

    # Or inside a LangChain chain (when langchain-core is installed)
    from langchain.chains import RetrievalQA
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = chain.invoke({"query": "What causes sepsis?"})
"""

from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os
from typing import List, Optional, Any

# Ensure the project root is on the path when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortexrag import VortexRAG, VortexRAGConfig

# ---------------------------------------------------------------------------
# Optional LangChain import — soft dependency
# ---------------------------------------------------------------------------
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Standalone retriever (works with or without langchain-core)
# ---------------------------------------------------------------------------

class _VortexRAGRetrieverBase:
    """
    VortexRAGRetriever base — works without langchain-core.

    Async and batch methods are available on all instances regardless of
    whether langchain-core is installed.

    Parameters
    ----------
    domain : str
        Domain preset: scientific, medical, biomedical, legal, cybersecurity,
        financial, code, educational, general, historical, customer, creative.
    top_k : int
        Maximum documents returned per query.
    config : VortexRAGConfig | None
        Full pipeline config override (ignores domain/verbose if set).
    verbose : bool
        Print pipeline progress.
    max_workers : int
        Thread pool size for batch/async retrieval.
    """

    def __init__(
        self,
        domain: str = "general",
        top_k: int = 5,
        config: Optional[VortexRAGConfig] = None,
        verbose: bool = False,
        max_workers: int = 4,
    ):
        if config is not None:
            self._config = config
        else:
            self._config = VortexRAGConfig(domain=domain)
            self._config.verbose = verbose

        self._top_k = top_k
        self._rag: Optional[VortexRAG] = None
        self._texts: list[str] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_documents(self, texts: list[str]) -> "_VortexRAGRetrieverBase":
        """
        Add raw text strings to the retriever index.

        Can be called multiple times — new texts are appended and the index
        is rebuilt automatically.

        Returns self for chaining:
            retriever = VortexRAGRetriever(domain="medical").add_documents(docs)
        """
        self._texts.extend(texts)
        self._rag = VortexRAG(corpus=self._texts, config=self._config)
        self._rag.index()
        return self

    def add_corpus_path(self, path: str) -> "_VortexRAGRetrieverBase":
        """Load all .txt / .md files from a directory path and index them."""
        self._rag = VortexRAG(corpus=path, config=self._config)
        self._rag.index()
        return self

    # ------------------------------------------------------------------
    # Sync retrieval
    # ------------------------------------------------------------------

    def get_relevant_documents(self, query: str) -> list:
        """
        Retrieve top-k documents for `query` via the VORTEXRAG pipeline.

        Returns LangChain Document objects when langchain-core is installed,
        otherwise plain dicts with `page_content` and `metadata` keys.
        """
        if self._rag is None:
            raise ValueError(
                "No documents indexed. Call add_documents() or add_corpus_path() first."
            )

        result = self._rag.query(query)
        chunks = result.context_window[: self._top_k]
        scores = result.phi_scores[: self._top_k]

        if _LANGCHAIN_AVAILABLE:
            return [
                Document(
                    page_content=chunk,
                    metadata={
                        "phi_score": round(scores[i], 4),
                        "rank": i + 1,
                        "domain": self._config.domain,
                        "delta_r": round(result.delta_r, 4),
                        "esr": round(result.esr, 4),
                    },
                )
                for i, chunk in enumerate(chunks)
            ]
        else:
            return [
                {
                    "page_content": chunk,
                    "metadata": {
                        "phi_score": round(scores[i], 4),
                        "rank": i + 1,
                        "domain": self._config.domain,
                    },
                }
                for i, chunk in enumerate(chunks)
            ]

    def invoke(self, query: str, **kwargs: Any) -> list:
        """Alias for get_relevant_documents."""
        return self.get_relevant_documents(query)

    # ------------------------------------------------------------------
    # Async and batch retrieval (always available, no extra deps)
    # ------------------------------------------------------------------

    async def aget_relevant_documents(self, query: str) -> list:
        """Non-blocking async retrieval via thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get_relevant_documents, query)

    async def ainvoke(self, query: str, **kwargs: Any) -> list:
        """Async alias for aget_relevant_documents."""
        return await self.aget_relevant_documents(query)

    def batch(self, queries: list) -> list:
        """Concurrent sync batch retrieval across multiple queries."""
        futures = [self._executor.submit(self.get_relevant_documents, q) for q in queries]
        return [f.result() for f in futures]

    async def abatch(self, queries: list) -> list:
        """Concurrent async batch retrieval via asyncio.gather."""
        return await asyncio.gather(*[self.aget_relevant_documents(q) for q in queries])


# ---------------------------------------------------------------------------
# LangChain-native subclass (only created when langchain-core is installed)
# ---------------------------------------------------------------------------

if _LANGCHAIN_AVAILABLE:
    class VortexRAGRetriever(_VortexRAGRetrieverBase, BaseRetriever):
        """
        LangChain BaseRetriever subclass backed by VORTEXRAG.

        Inherits all methods from _VortexRAGRetrieverBase including async/batch.

        Example::

            from vortexrag.integrations.langchain_retriever import VortexRAGRetriever

            retriever = VortexRAGRetriever(domain="medical", top_k=5)
            retriever.add_documents(corpus_texts)

            chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        """

        model_config = {"arbitrary_types_allowed": True}

        def model_post_init(self, __context: Any) -> None:
            if not hasattr(self, "_rag"):
                self._rag = None
            if not hasattr(self, "_texts"):
                self._texts = []
            if not hasattr(self, "_executor"):
                self._executor = ThreadPoolExecutor(max_workers=4)

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional["CallbackManagerForRetrieverRun"] = None,
        ) -> "List[Document]":
            return self.get_relevant_documents(query)  # type: ignore[return-value]

else:
    # Fallback: plain class, all methods including async/batch still available
    VortexRAGRetriever = _VortexRAGRetrieverBase  # type: ignore[misc,assignment]
