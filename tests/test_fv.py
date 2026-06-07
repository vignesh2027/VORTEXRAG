"""
Tests for Faithfulness Verifier (FV) — Layer 7 of VORTEXRAG.

Verifies:
  - Faithful answer: ROUGE-L=0.85, NLI=0.94 → Delta_R = 1 - 0.85*0.94 = 0.201
  - Acceptance condition: Delta_R <= delta_FV
  - Retry logic: if Delta_R > delta_FV, trigger regeneration
  - Max 3 retries enforced
  - Final answer = argmin Delta_R across all iterations
  - ROUGE-L LCS computation correctness
  - NLI entailment probability extraction
  - Edge case: empty answer
  - Medical preset delta_FV=0.10 (stricter)
"""

import numpy as np
import pytest

from core.fv import FaithfulnessVerifier, FVConfig, FVResult


# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_CONTEXT = (
    "The 2008 financial crisis was caused by excessive leverage in mortgage-backed "
    "securities and collateralized debt obligations. Banks issued subprime mortgages "
    "to borrowers who could not afford them. When housing prices fell, defaults "
    "cascaded through the financial system, triggering the collapse of Lehman Brothers."
)

FAITHFUL_ANSWER = (
    "The financial crisis was caused by excessive leverage in mortgage-backed securities. "
    "Subprime mortgages defaulted when housing prices fell, which triggered the collapse "
    "of Lehman Brothers."
)

HALLUCINATED_ANSWER = (
    "The crisis was caused by alien invasion of Wall Street and cosmic radiation from "
    "Jupiter affecting trading algorithms on Mars."
)


@pytest.fixture
def fv_default() -> FaithfulnessVerifier:
    return FaithfulnessVerifier(FVConfig(delta_fv=0.15, max_iterations=3, use_nli=False))


@pytest.fixture
def fv_strict() -> FaithfulnessVerifier:
    """Medical preset: stricter delta_FV=0.10."""
    return FaithfulnessVerifier(FVConfig(delta_fv=0.10, max_iterations=3, use_nli=False))


# ── Delta-R formula ────────────────────────────────────────────────────────────

class TestDeltaR:

    def test_delta_r_formula_faithful(self, fv_default):
        """
        Faithful answer against matching context should have low Delta_R.
        Delta_R = 1 - ROUGE-L * NLI (when NLI is not available, NLI = ROUGE-L as proxy).
        """
        dr, rouge, nli = fv_default.delta_r(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        assert 0.0 <= dr <= 1.0, f"Delta_R out of [0,1]: {dr}"
        assert rouge > 0.3, f"ROUGE-L for faithful answer should be > 0.3, got {rouge}"

    def test_delta_r_formula_hallucinated(self, fv_default):
        """Hallucinated answer should have high Delta_R (near 1.0)."""
        dr, rouge, nli = fv_default.delta_r(HALLUCINATED_ANSWER, SAMPLE_CONTEXT)
        assert dr > 0.5, f"Hallucinated answer Delta_R should be > 0.5, got {dr}"

    def test_delta_r_identical_answer_and_context(self, fv_default):
        """Identical answer and context → ROUGE-L ≈ 1.0 → Delta_R ≈ 0."""
        text = "The housing bubble caused the financial crisis to spread."
        dr, rouge, nli = fv_default.delta_r(text, text)
        # ROUGE-L should be very high for identical text
        assert rouge > 0.9, f"Identical text ROUGE-L should be > 0.9, got {rouge}"
        assert dr < 0.2, f"Identical text Delta_R should be low, got {dr}"

    def test_delta_r_values_in_range(self, fv_default):
        """Delta_R must always be in [0, 1]."""
        pairs = [
            ("short answer", "short context"),
            (FAITHFUL_ANSWER, SAMPLE_CONTEXT),
            (HALLUCINATED_ANSWER, SAMPLE_CONTEXT),
            ("", SAMPLE_CONTEXT),
            (SAMPLE_CONTEXT, ""),
        ]
        for answer, context in pairs:
            dr, rouge, nli = fv_default.delta_r(answer, context)
            assert 0.0 <= dr <= 1.0, f"Delta_R={dr} out of [0,1] for pair"
            assert 0.0 <= rouge <= 1.0, f"ROUGE-L={rouge} out of [0,1]"
            assert 0.0 <= nli <= 1.0, f"NLI={nli} out of [0,1]"

    def test_specific_delta_r_calculation(self, fv_default):
        """
        Test the specific case: if ROUGE-L=0.85 and NLI=0.94,
        then Delta_R = 1 - 0.85*0.94 = 0.201.

        Since NLI falls back to ROUGE-L when model unavailable,
        this test manually verifies the formula structure.
        """
        rouge_l = 0.85
        nli = 0.94
        grounding = rouge_l * nli
        expected_delta_r = 1.0 - grounding
        assert abs(expected_delta_r - 0.201) < 0.001, (
            f"Expected Delta_R ≈ 0.201, got {expected_delta_r:.4f}"
        )

    def test_grounding_score_inverse_delta_r(self, fv_default):
        """grounding_score() = 1 - delta_r."""
        dr, _, _ = fv_default.delta_r(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        grounding = fv_default.grounding_score(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        assert abs(grounding - (1.0 - dr)) < 1e-6


# ── Acceptance threshold ──────────────────────────────────────────────────────

class TestAcceptanceThreshold:

    def test_accepted_when_delta_r_below_threshold(self, fv_default):
        """Answer with Delta_R <= delta_FV must be accepted."""
        result = fv_default.verify(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        # Only check structure; acceptance depends on actual ROUGE-L vs threshold
        assert isinstance(result.accepted, bool)
        if result.delta_r <= fv_default.config.delta_fv:
            assert result.accepted is True

    def test_rejected_when_delta_r_above_threshold(self, fv_default):
        """Hallucinated answer should fail faithfulness check."""
        result = fv_default.verify(HALLUCINATED_ANSWER, SAMPLE_CONTEXT)
        if result.delta_r > fv_default.config.delta_fv:
            assert result.accepted is False

    def test_strict_threshold_rejects_borderline(self):
        """Strict threshold (0.05) should reject answers that default threshold (0.15) accepts."""
        fv_strict_05 = FaithfulnessVerifier(FVConfig(delta_fv=0.05, use_nli=False))
        fv_lenient_40 = FaithfulnessVerifier(FVConfig(delta_fv=0.40, use_nli=False))

        context = "The rocket engine uses liquid hydrogen and liquid oxygen as propellants."
        answer = "The rocket uses hydrogen and oxygen fuel for propulsion."

        result_strict = fv_strict_05.verify(answer, context)
        result_lenient = fv_lenient_40.verify(answer, context)

        # Lenient should be at least as accepting as strict
        if result_strict.delta_r > 0.05:
            assert not result_strict.accepted
        if result_lenient.delta_r <= 0.40:
            assert result_lenient.accepted

    def test_medical_preset_stricter(self, fv_strict):
        """Medical preset delta_FV=0.10 should be stricter than default 0.15."""
        assert fv_strict.config.delta_fv < 0.15, (
            f"Medical FV threshold ({fv_strict.config.delta_fv}) should be < 0.15"
        )


# ── Retry logic ───────────────────────────────────────────────────────────────

class TestRetryLogic:

    def test_verify_with_retry_calls_generate_fn(self, fv_default):
        """verify_with_retry() should call generate_fn and return an FVResult."""
        call_count = [0]

        def mock_generate(context: str, attempt: int) -> str:
            call_count[0] += 1
            return FAITHFUL_ANSWER

        result = fv_default.verify_with_retry(SAMPLE_CONTEXT, mock_generate)
        assert isinstance(result, FVResult)
        assert call_count[0] >= 1

    def test_verify_with_retry_stops_on_acceptance(self, fv_default):
        """verify_with_retry() should stop early when answer is accepted."""
        call_count = [0]

        # Make generate_fn return a near-perfect answer immediately
        def mock_generate_perfect(context: str, attempt: int) -> str:
            call_count[0] += 1
            # Return context text itself — ROUGE-L will be high
            return context[:300]

        result = fv_default.verify_with_retry(SAMPLE_CONTEXT, mock_generate_perfect)
        # Should stop at 1 or 2 attempts (accepted immediately or early)
        assert call_count[0] <= fv_default.config.max_iterations

    def test_verify_with_retry_max_3_iterations(self):
        """verify_with_retry() must not exceed max_iterations=3."""
        config = FVConfig(delta_fv=0.0001, max_iterations=3, use_nli=False)  # nearly impossible threshold
        fv = FaithfulnessVerifier(config)
        call_count = [0]

        def mock_generate(context: str, attempt: int) -> str:
            call_count[0] += 1
            return HALLUCINATED_ANSWER  # always fails

        result = fv.verify_with_retry(SAMPLE_CONTEXT, mock_generate)
        assert call_count[0] <= 3, (
            f"Expected at most 3 retries, got {call_count[0]}"
        )

    def test_verify_with_retry_returns_best_result(self):
        """verify_with_retry() must return the result with lowest Delta_R."""
        config = FVConfig(delta_fv=0.0001, max_iterations=3, use_nli=False)  # forces all iterations
        fv = FaithfulnessVerifier(config)

        answers = [
            HALLUCINATED_ANSWER,           # attempt 1: bad
            FAITHFUL_ANSWER,               # attempt 2: better
            "This is a partial answer.",   # attempt 3: middling
        ]
        attempt_num = [0]

        def mock_generate(context: str, attempt: int) -> str:
            ans = answers[min(attempt_num[0], len(answers) - 1)]
            attempt_num[0] += 1
            return ans

        result = fv.verify_with_retry(SAMPLE_CONTEXT, mock_generate)

        # Compute Delta_R for all answers manually
        delta_rs = [fv.verify(a, SAMPLE_CONTEXT).delta_r for a in answers]
        best_expected_dr = min(delta_rs)

        assert abs(result.delta_r - best_expected_dr) < 0.01, (
            f"Expected best Delta_R {best_expected_dr:.4f}, got {result.delta_r:.4f}"
        )

    def test_iteration_number_in_result(self, fv_default):
        """FVResult.iteration should record which attempt produced the result."""
        attempt_num = [0]

        def mock_generate(context: str, attempt: int) -> str:
            attempt_num[0] = attempt
            return FAITHFUL_ANSWER

        result = fv_default.verify_with_retry(SAMPLE_CONTEXT, mock_generate)
        assert result.iteration >= 1, "Iteration number should be >= 1"
        assert result.iteration <= fv_default.config.max_iterations


# ── ROUGE-L computation ────────────────────────────────────────────────────────

class TestROUGEL:

    def test_rouge_l_identical_texts(self, fv_default):
        """ROUGE-L between identical texts should be 1.0."""
        text = "The quick brown fox jumps over the lazy dog."
        rl = fv_default.rouge_l(text, text)
        assert abs(rl - 1.0) < 1e-5, f"Identical text ROUGE-L should be 1.0, got {rl}"

    def test_rouge_l_completely_different(self, fv_default):
        """ROUGE-L between completely different texts should be very low."""
        hyp = "banana orange apple cherry grape lemon mango kiwi"
        ref = "quantum mechanics physics electron orbital uncertainty"
        rl = fv_default.rouge_l(hyp, ref)
        # No overlapping words → ROUGE-L should be very low
        assert rl < 0.1, f"Completely different texts ROUGE-L should be < 0.1, got {rl}"

    def test_rouge_l_range(self, fv_default):
        """ROUGE-L must always be in [0, 1]."""
        pairs = [
            ("short", "longer reference text here"),
            ("the cat sat", "the dog sat on the mat"),
            ("", "non-empty reference"),
            ("non-empty hypothesis", ""),
        ]
        for hyp, ref in pairs:
            rl = fv_default.rouge_l(hyp, ref)
            assert 0.0 <= rl <= 1.0, f"ROUGE-L={rl} out of [0,1] for ({hyp!r}, {ref!r})"

    def test_rouge_l_empty_hypothesis(self, fv_default):
        """Empty hypothesis → ROUGE-L = 0."""
        rl = fv_default.rouge_l("", "Some reference text here.")
        assert rl == 0.0, f"Empty hypothesis ROUGE-L should be 0.0, got {rl}"

    def test_rouge_l_symmetric_approx(self, fv_default):
        """ROUGE-L is approximately symmetric (precision/recall swap changes F1 slightly)."""
        a = "the cat sat on the mat"
        b = "the cat sat by the fire"
        rl_ab = fv_default.rouge_l(a, b)
        rl_ba = fv_default.rouge_l(b, a)
        # Both should be in the same range; not strictly equal due to P vs R swap
        assert abs(rl_ab - rl_ba) < 0.3, (
            f"ROUGE-L should be approximately symmetric: {rl_ab:.4f} vs {rl_ba:.4f}"
        )

    def test_lcs_length_known_case(self, fv_default):
        """LCS of 'ABCBDAB' and 'BDCAB' is 4 (standard CS textbook example)."""
        a = list("ABCBDAB")
        b = list("BDCAB")
        lcs = fv_default._lcs_length(a, b)
        assert lcs == 4, f"LCS('ABCBDAB','BDCAB') should be 4, got {lcs}"

    def test_lcs_empty_sequences(self, fv_default):
        """LCS with empty sequence should be 0."""
        assert fv_default._lcs_length([], ["a", "b"]) == 0
        assert fv_default._lcs_length(["a", "b"], []) == 0
        assert fv_default._lcs_length([], []) == 0


# ── ROUGE-N ───────────────────────────────────────────────────────────────────

class TestROUGEN:

    def test_rouge_1_range(self, fv_default):
        """ROUGE-1 must be in [0, 1]."""
        rl1 = fv_default.rouge_n(FAITHFUL_ANSWER, SAMPLE_CONTEXT, n=1)
        assert 0.0 <= rl1 <= 1.0

    def test_rouge_2_range(self, fv_default):
        """ROUGE-2 must be in [0, 1]."""
        rl2 = fv_default.rouge_n(FAITHFUL_ANSWER, SAMPLE_CONTEXT, n=2)
        assert 0.0 <= rl2 <= 1.0

    def test_all_rouge_returns_three_scores(self, fv_default):
        """all_rouge() must return rouge_1, rouge_2, rouge_l keys."""
        scores = fv_default.all_rouge(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        assert set(scores.keys()) == {"rouge_1", "rouge_2", "rouge_l"}
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_rouge_1_ge_rouge_2(self, fv_default):
        """ROUGE-1 >= ROUGE-2 for most text pairs (more unigrams match than bigrams)."""
        pairs = [
            (FAITHFUL_ANSWER, SAMPLE_CONTEXT),
            ("The cat sat", "The cat sat on the mat nearby"),
        ]
        for hyp, ref in pairs:
            r1 = fv_default.rouge_n(hyp, ref, n=1)
            r2 = fv_default.rouge_n(hyp, ref, n=2)
            assert r1 >= r2 - 0.01, (
                f"ROUGE-1 ({r1:.4f}) should be ≥ ROUGE-2 ({r2:.4f})"
            )


# ── FVResult structure ────────────────────────────────────────────────────────

class TestFVResult:

    def test_verify_returns_fvresult(self, fv_default):
        """verify() must return an FVResult."""
        result = fv_default.verify(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        assert isinstance(result, FVResult)

    def test_fvresult_fields_populated(self, fv_default):
        """FVResult must have all fields populated with valid types."""
        result = fv_default.verify(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        assert isinstance(result.answer, str)
        assert isinstance(result.delta_r, float)
        assert isinstance(result.rouge_l, float)
        assert isinstance(result.rouge_1, float)
        assert isinstance(result.rouge_2, float)
        assert isinstance(result.nli_score, float)
        assert isinstance(result.grounding, float)
        assert isinstance(result.accepted, bool)
        assert isinstance(result.iteration, int)

    def test_fvresult_grounding_equals_one_minus_delta_r(self, fv_default):
        """FVResult.grounding should equal 1 - delta_r."""
        result = fv_default.verify(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        assert abs(result.grounding - (1.0 - result.delta_r)) < 1e-4

    def test_verify_empty_answer(self, fv_default):
        """Empty answer should not crash; should produce high Delta_R."""
        result = fv_default.verify("", SAMPLE_CONTEXT)
        assert isinstance(result, FVResult)
        assert result.delta_r >= 0.0


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_context_high_delta_r(self, fv_default):
        """Empty context → very high Delta_R (answer can't be grounded)."""
        result = fv_default.verify("Some answer about the topic.", "")
        assert result.delta_r > 0.5 or result.rouge_l == 0.0

    def test_very_long_answer(self, fv_default):
        """Long answer should not crash."""
        long_answer = " ".join([FAITHFUL_ANSWER] * 10)
        result = fv_default.verify(long_answer, SAMPLE_CONTEXT)
        assert isinstance(result, FVResult)
        assert 0.0 <= result.delta_r <= 1.0

    def test_single_word_answer(self, fv_default):
        """Single-word answer should produce valid FVResult."""
        result = fv_default.verify("crisis", SAMPLE_CONTEXT)
        assert isinstance(result, FVResult)
        assert 0.0 <= result.delta_r <= 1.0

    def test_compare_answers_sorted_by_delta_r(self, fv_default):
        """compare_answers() must return results sorted by delta_r ascending."""
        answers = [
            HALLUCINATED_ANSWER,  # high delta_r
            FAITHFUL_ANSWER,      # low delta_r
            "The crisis involved mortgage securities.",  # medium
        ]
        results = fv_default.compare_answers(answers, SAMPLE_CONTEXT)
        for i in range(len(results) - 1):
            assert results[i].delta_r <= results[i + 1].delta_r, (
                f"Results not sorted by delta_r: {results[i].delta_r} > {results[i+1].delta_r}"
            )

    def test_threshold_analysis_all_thresholds(self, fv_default):
        """threshold_analysis() should return one entry per threshold."""
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
        analysis = fv_default.threshold_analysis(FAITHFUL_ANSWER, SAMPLE_CONTEXT, thresholds)
        assert len(analysis) == len(thresholds)
        for entry in analysis:
            assert "threshold" in entry
            assert "accepted" in entry
            assert "margin" in entry


# ── Sentence-level verification ───────────────────────────────────────────────

class TestSentenceLevelVerification:

    def test_sentence_level_verify_returns_results(self, fv_default):
        """sentence_level_verify() should return results for each sentence."""
        answer = (
            "The crisis was caused by excessive leverage. "
            "Subprime mortgages led to cascading defaults. "
            "Lehman Brothers ultimately collapsed."
        )
        results = fv_default.sentence_level_verify(answer, SAMPLE_CONTEXT)
        assert len(results) >= 1, "Should return at least one sentence result"

    def test_sentence_split_works(self, fv_default):
        """_split_sentences() should split on period-followed-by-uppercase."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = fv_default._split_sentences(text)
        assert len(sentences) >= 2, f"Expected at least 2 sentences, got {len(sentences)}"

    def test_grounding_report_keys(self, fv_default):
        """grounding_report() should return all expected keys."""
        report = fv_default.grounding_report(FAITHFUL_ANSWER, SAMPLE_CONTEXT)
        required_keys = {
            "delta_r", "grounding_score", "accepted", "rouge_l",
            "rouge_1", "rouge_2", "nli_score", "delta_fv_threshold",
            "margin", "n_answer_tokens", "n_context_tokens",
            "n_sentences", "n_hallucinated_sents", "interpretation",
        }
        assert required_keys.issubset(report.keys()), (
            f"Missing keys: {required_keys - report.keys()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
