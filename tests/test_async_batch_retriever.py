"""
Tests for async and batch extensions of VortexRAGRetriever.
"""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.langchain_retriever import VortexRAGRetriever

SAMPLE_DOCS = [
    "Sepsis is a life-threatening condition caused by the body's extreme response to infection. "
    "It occurs when chemicals released into the bloodstream to fight infection trigger widespread inflammation.",

    "Treatment of sepsis requires immediate administration of antibiotics and intravenous fluids. "
    "Vasopressors may be needed to maintain blood pressure in cases of septic shock.",

    "The sequential organ failure assessment (SOFA) score is used to track a patient's status "
    "and predict outcomes in intensive care units.",

    "Machine learning models have been applied to predict sepsis onset from electronic health records, "
    "achieving AUROC scores above 0.85 on held-out test sets.",
]


@pytest.fixture
def retriever():
    r = VortexRAGRetriever(domain="medical", top_k=3)
    r.add_documents(SAMPLE_DOCS)
    return r


class TestAsyncRetrieval:
    def test_aget_relevant_documents_returns_list(self, retriever):
        results = asyncio.run(retriever.aget_relevant_documents("What causes sepsis?"))
        assert isinstance(results, list)
        assert len(results) > 0

    def test_ainvoke_alias(self, retriever):
        results = asyncio.run(retriever.ainvoke("sepsis treatment"))
        assert isinstance(results, list)

    def test_async_respects_top_k(self, retriever):
        results = asyncio.run(retriever.aget_relevant_documents("organ failure"))
        assert len(results) <= 3

    def test_async_result_has_page_content(self, retriever):
        results = asyncio.run(retriever.aget_relevant_documents("infection"))
        for doc in results:
            if isinstance(doc, dict):
                assert "page_content" in doc
            else:
                assert hasattr(doc, "page_content")

    def test_async_result_has_metadata(self, retriever):
        results = asyncio.run(retriever.aget_relevant_documents("antibiotics"))
        for doc in results:
            meta = doc.metadata if hasattr(doc, "metadata") else doc["metadata"]
            assert "phi_score" in meta
            assert "rank" in meta

    def test_async_raises_without_documents(self):
        r = VortexRAGRetriever(domain="medical", top_k=3)
        with pytest.raises(ValueError, match="No documents indexed"):
            asyncio.run(r.aget_relevant_documents("query"))

    def test_async_matches_sync_length(self, retriever):
        query = "What causes sepsis?"
        sync_results = retriever.get_relevant_documents(query)
        async_results = asyncio.run(retriever.aget_relevant_documents(query))
        assert len(sync_results) == len(async_results)

    def test_abatch_concurrent(self, retriever):
        queries = ["What causes sepsis?", "How is it treated?", "SOFA score"]
        results = asyncio.run(retriever.abatch(queries))
        assert len(results) == 3
        for doc_list in results:
            assert isinstance(doc_list, list)

    def test_abatch_empty_list(self, retriever):
        results = asyncio.run(retriever.abatch([]))
        assert results == []


class TestBatchRetrieval:
    def test_batch_returns_list_of_lists(self, retriever):
        results = retriever.batch(["What causes sepsis?", "sepsis treatment"])
        assert len(results) == 2
        for doc_list in results:
            assert isinstance(doc_list, list)

    def test_batch_order_preserved(self, retriever):
        queries = ["infection", "antibiotics", "organ failure"]
        results = retriever.batch(queries)
        assert len(results) == len(queries)