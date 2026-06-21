"""
Tests for the LangChain integration (VortexRAGRetriever).
Runs without langchain-core installed — falls back to plain dicts.
"""

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


class TestVortexRAGRetriever:
    def test_instantiation_no_langchain(self):
        r = VortexRAGRetriever(domain="general", top_k=2)
        assert r._top_k == 2

    def test_add_documents_indexes(self, retriever):
        assert retriever._rag is not None
        assert retriever._rag._indexed

    def test_get_relevant_documents_returns_list(self, retriever):
        results = retriever.get_relevant_documents("What causes sepsis?")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_results_respect_top_k(self, retriever):
        results = retriever.get_relevant_documents("sepsis treatment")
        assert len(results) <= 3

    def test_result_has_page_content(self, retriever):
        results = retriever.get_relevant_documents("What causes sepsis?")
        for doc in results:
            if isinstance(doc, dict):
                assert "page_content" in doc
                assert len(doc["page_content"]) > 0
            else:
                assert hasattr(doc, "page_content")
                assert len(doc.page_content) > 0

    def test_result_has_metadata(self, retriever):
        results = retriever.get_relevant_documents("sepsis score")
        for doc in results:
            meta = doc.metadata if hasattr(doc, "metadata") else doc["metadata"]
            assert "phi_score" in meta
            assert "rank" in meta
            assert meta["rank"] >= 1

    def test_invoke_alias(self, retriever):
        results = retriever.invoke("infection treatment")
        assert isinstance(results, list)

    def test_raises_without_documents(self):
        r = VortexRAGRetriever(domain="medical", top_k=3)
        with pytest.raises(ValueError, match="No documents indexed"):
            r.get_relevant_documents("query")

    def test_add_documents_chaining(self):
        r = (
            VortexRAGRetriever(domain="scientific", top_k=2)
            .add_documents(SAMPLE_DOCS[:2])
        )
        assert r._rag is not None

    def test_multiple_add_documents_appends(self):
        r = VortexRAGRetriever(domain="general", top_k=5)
        r.add_documents(SAMPLE_DOCS[:2])
        r.add_documents(SAMPLE_DOCS[2:])
        assert len(r._texts) == len(SAMPLE_DOCS)

    def test_domain_propagates(self):
        r = VortexRAGRetriever(domain="biomedical", top_k=3)
        r.add_documents(SAMPLE_DOCS)
        results = r.get_relevant_documents("organ failure")
        for doc in results:
            meta = doc.metadata if hasattr(doc, "metadata") else doc["metadata"]
            assert meta["domain"] == "biomedical"

    def test_different_domains_accepted(self):
        for domain in ["scientific", "legal", "biomedical", "financial", "code"]:
            r = VortexRAGRetriever(domain=domain, top_k=2)
            r.add_documents(SAMPLE_DOCS)
            results = r.get_relevant_documents("test query")
            assert isinstance(results, list)
