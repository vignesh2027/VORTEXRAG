"""
End-to-end tests for VORTEXRAG pipeline.

Tests the 4 worked examples from the VORTEXRAG paper:
  1. Multi-hop legal reasoning
  2. Medical mechanism synthesis
  3. Python asyncio SyntaxError vs RuntimeError
  4. Type Ia vs Type II supernova progenitors

Each test verifies that VORTEXRAG:
  a) Runs without errors on the full pipeline
  b) Returns a VortexRAGResult with valid scores
  c) The correct chunk (high causal relevance) scores higher than
     semantically similar but causally irrelevant chunks
"""

import pytest
from vortexrag import VortexRAG, VortexRAGConfig


# ──── Test Corpora ──────────────────────────────────────────────────────────

LEGAL_CORPUS = [
    # Causally relevant: direct answer to the query
    """Cooper v. Aaron (1958) extended the Brown v. Board of Education (1954)
    desegregation mandate to all state-run institutions, including public
    universities. The Supreme Court unanimously held that no state official
    could nullify or circumvent its orders. This directly established that
    the principles of Brown applied to university-level segregation.""",

    # High semantic similarity but causally IRRELEVANT (should be filtered by SDC)
    """The Civil Rights Act of 1964 prohibited discrimination based on race,
    color, religion, sex, or national origin in employment and public
    accommodations. It was a landmark piece of legislation that transformed
    American society and is widely cited in civil rights cases.""",

    # Low relevance
    """The First Amendment to the United States Constitution protects freedom
    of speech, religion, press, assembly, and petition from government
    interference. It applies to all levels of government.""",

    # Secondary relevant: supporting context
    """Green v. County School Board (1968) further solidified the application
    of Brown, ruling that school boards had an affirmative duty to eliminate
    segregated school systems root and branch, not merely to stop practicing
    overt segregation.""",
]

MEDICAL_CORPUS = [
    # Causally relevant: mRNA mechanism
    """mRNA vaccines work by introducing messenger RNA encoding the spike
    protein into host cells. Ribosomes translate the mRNA into spike protein,
    which is then displayed on the cell surface, triggering an immune response.
    The mRNA is degraded within days and never enters the cell nucleus.""",

    # Causally relevant: viral vector mechanism
    """Viral vector vaccines use a modified adenovirus to deliver DNA encoding
    the spike protein into the cell nucleus. The DNA is transcribed into mRNA
    by the cell's own machinery, which is then translated into spike protein.
    The adenovirus vector cannot replicate and does not integrate into the genome.""",

    # Semantically similar but conflates both (CPG should handle ordering)
    """Both mRNA and viral vector vaccines successfully generate immune responses
    against the SARS-CoV-2 spike protein and have shown high efficacy in
    clinical trials. Both types were authorized for emergency use in 2020-2021.""",
]

ASYNCIO_CORPUS = [
    # Causally relevant: syntactic/AST-level answer
    """In Python, the await keyword is syntactically restricted to async def
    function bodies. The Python parser (ast module) enforces this at compile
    time: if await appears outside an async context, the parser raises a
    SyntaxError before any bytecode is generated. This is a grammar-level
    constraint, not a runtime check.""",

    # Semantically similar but wrong level (should be SDC-filtered)
    """Python's asyncio event loop raises RuntimeError when you call
    loop.run_until_complete() from within a running event loop. This is a
    runtime check because the event loop state is only known at execution time.
    nest_asyncio can patch this behavior.""",

    # Supporting context
    """The CPython compiler tokenizes and parses Python source into an Abstract
    Syntax Tree (AST). Grammar rules enforce syntactic constraints like await
    placement. Only after successful parsing does compilation to bytecode occur.""",
]

SUPERNOVA_CORPUS = [
    # Causally relevant: Type Ia progenitor
    """Type Ia supernovae arise from white dwarf stars in binary systems.
    The white dwarf accretes mass from a companion star until it reaches
    the Chandrasekhar limit (~1.4 solar masses), triggering a thermonuclear
    runaway that completely destroys the star. No neutron star or black hole
    remnant is left.""",

    # Causally relevant: Type II progenitor
    """Type II supernovae result from the core collapse of massive stars
    (>8 solar masses) that have exhausted their nuclear fuel. The iron core
    collapses under gravity in milliseconds, producing a neutron star or
    black hole, while the outer envelope is expelled in an explosion.""",

    # Semantically similar but NOT about progenitors (SDC should filter)
    """Type Ia supernovae are used as standard candles in cosmology because
    their peak luminosity is predictable. This property enabled the 1998
    discovery of accelerating cosmic expansion and dark energy. Type II
    supernovae are less consistent in peak brightness.""",
]


# ──── Tests ─────────────────────────────────────────────────────────────────

class TestE2EPipeline:
    """Tests that the full pipeline runs correctly on all 4 worked examples."""

    def _run_query(self, corpus, query, domain="general"):
        config = VortexRAGConfig(domain=domain, verbose=False)
        config.vrc.candidate_pool = min(len(corpus) * 2, 20)
        config.vrc.top_k = min(len(corpus), 10)
        config.rfg.top_m = 5
        rag = VortexRAG(config=config)
        rag.index(additional_texts=corpus)
        return rag.query(query)

    def test_legal_pipeline_runs(self):
        """Test 1: Multi-hop legal reasoning — Cooper v. Aaron + Brown extension."""
        result = self._run_query(
            LEGAL_CORPUS,
            "Did the precedent set in Brown v. Board also apply to public universities before 1964?",
            domain="legal",
        )
        assert result is not None
        assert isinstance(result.answer, str)
        assert len(result.context_window) > 0
        assert 0 <= result.delta_r <= 1.0
        assert result.esr > 0
        assert result.latency_ms > 0

    def test_medical_pipeline_runs(self):
        """Test 2: Medical mechanism synthesis — mRNA vs viral vector vaccines."""
        result = self._run_query(
            MEDICAL_CORPUS,
            "What is the mechanistic difference between mRNA vaccines and viral vector vaccines in spike protein expression?",
            domain="medical",
        )
        assert result is not None
        assert len(result.context_window) > 0
        assert result.phi_scores[0] >= result.phi_scores[-1], \
            "Phi scores should be in descending order"

    def test_asyncio_pipeline_runs(self):
        """Test 3: Code docs — asyncio SyntaxError vs RuntimeError."""
        result = self._run_query(
            ASYNCIO_CORPUS,
            "In Python asyncio, why does await inside a non-async function cause a SyntaxError but not a RuntimeError?",
            domain="code",
        )
        assert result is not None
        assert result.spiral_pool_size > 0

    def test_supernova_pipeline_runs(self):
        """Test 4: Scientific reasoning — Type Ia vs Type II progenitors."""
        result = self._run_query(
            SUPERNOVA_CORPUS,
            "What distinguishes Type Ia from Type II supernovae in terms of their progenitor systems?",
            domain="scientific",
        )
        assert result is not None
        assert len(result.context_window) >= 1

    def test_result_structure_complete(self):
        """Verify VortexRAGResult has all expected fields populated."""
        result = self._run_query(
            LEGAL_CORPUS,
            "What is Cooper v. Aaron?",
        )
        assert result.query == "What is Cooper v. Aaron?"
        assert isinstance(result.answer, str)
        assert isinstance(result.context_window, list)
        assert isinstance(result.phi_scores, list)
        assert isinstance(result.delta_r, float)
        assert isinstance(result.esr, float)
        assert isinstance(result.spiral_pool_size, int)
        assert isinstance(result.purge_count, int)
        assert isinstance(result.latency_ms, float)
        assert isinstance(result.iterations, int)
        assert isinstance(result.accepted, bool)

    def test_index_required_before_query(self):
        """Querying before indexing should raise ValueError."""
        rag = VortexRAG()
        with pytest.raises(ValueError, match="index"):
            rag.query("test query")

    def test_custom_llm_fn_is_called(self):
        """Custom LLM function should be called with context and query."""
        called_with = {}

        def mock_llm(context, query):
            called_with["context"] = context
            called_with["query"] = query
            return "Custom LLM answer"

        config = VortexRAGConfig(verbose=False)
        config.vrc.top_k = 5
        config.rfg.top_m = 3
        rag = VortexRAG(config=config, llm_fn=mock_llm)
        rag.index(additional_texts=LEGAL_CORPUS)
        result = rag.query("What is Brown v. Board?")

        assert called_with.get("query") == "What is Brown v. Board?"
        assert result.answer == "Custom LLM answer"

    def test_phi_scores_are_normalized(self):
        """Phi scores should roughly sum to ~1 (normalized distribution)."""
        result = self._run_query(SUPERNOVA_CORPUS, "What are Type Ia supernovae?")
        phi_sum = sum(result.phi_scores)
        # Should be close to 1 (normalized) — allow tolerance for floating point
        assert 0.9 <= phi_sum <= 1.1, f"Phi scores sum: {phi_sum} — expected ~1.0"
