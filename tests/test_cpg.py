"""
Tests for Context Poison Guard (CPG).

Verifies that:
  1. Clean windows (all high SDS) pass immediately with high ESR
  2. Poisoned windows (low SDS chunks) trigger iterative purging
  3. Purging always improves ESR (monotone property)
  4. min_chunks constraint is respected
  5. Softmax weights correctly amplify high-TVE chunks
"""

import numpy as np
import pytest

from core.tve import TVEVector
from core.vrc import SpiralCandidate
from core.sdc import SDCResult, SemanticDriftCorrector, SDCConfig
from core.cpg import ContextPoisonGuard, CPGConfig


def make_unit_vec(dim=768, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)

def make_tve_vec(seed=0, dim=768) -> TVEVector:
    return TVEVector(
        semantic=make_unit_vec(dim, seed),
        syntactic=make_unit_vec(dim, seed+1),
        causal=make_unit_vec(dim, seed+2),
    )

def make_mock_sdc_result(
    chunk_id: int,
    sds_score: float,
    tve_score: float,
    text: str = "mock chunk text",
) -> SDCResult:
    """Build a mock SDCResult for testing."""
    tve_vec = make_tve_vec(seed=chunk_id)
    spiral = SpiralCandidate(
        chunk_id=chunk_id,
        chunk_text=text,
        tve_score=tve_score,
        radial_dist=1.0,
        theta=0.5,
        spiral_rank=tve_score,
        tve_vec=tve_vec,
    )
    return SDCResult(
        candidate=spiral,
        drift_norm=1.0 - sds_score,
        sds_score=sds_score,
        accepted=sds_score >= 0.72,
        drift_direction=np.zeros(768, dtype=np.float32),
    )


class TestCPGScoring:

    def test_clean_window_passes_immediately(self):
        """A window of all high-SDS chunks should pass without purging."""
        config = CPGConfig(theta_cpg=3.5)
        cpg = ContextPoisonGuard(config)
        q_vec = make_tve_vec(seed=0)

        # All chunks have high SDS (0.9) and high TVE (0.85)
        sdc_results = [
            make_mock_sdc_result(i, sds_score=0.9, tve_score=0.85)
            for i in range(10)
        ]

        result = cpg.evaluate(q_vec, sdc_results)
        assert result.is_clean, f"Clean window should pass, ESR={result.esr:.3f}"
        assert result.purge_count == 0, "Clean window should need no purging"

    def test_poisoned_window_triggers_purging(self):
        """Window with very low SDS chunks should trigger purging."""
        config = CPGConfig(theta_cpg=3.5, min_chunks=2)
        cpg = ContextPoisonGuard(config)
        q_vec = make_tve_vec(seed=0)

        # Mix: a few good chunks, many poisoned
        sdc_results = (
            [make_mock_sdc_result(i, sds_score=0.85, tve_score=0.80) for i in range(3)] +
            [make_mock_sdc_result(i+3, sds_score=0.20, tve_score=0.75) for i in range(7)]
        )

        result = cpg.evaluate(q_vec, sdc_results)
        assert result.purge_count > 0, "Poisoned window should trigger purging"

    def test_esr_increases_monotonically_with_purging(self):
        """Each purge step should increase (or maintain) ESR."""
        config = CPGConfig(theta_cpg=100.0, min_chunks=1)  # force maximum purging
        cpg = ContextPoisonGuard(config)
        q_vec = make_tve_vec(seed=0)

        sdc_results = [
            make_mock_sdc_result(i, sds_score=0.4 + i*0.05, tve_score=0.7)
            for i in range(10)
        ]

        result = cpg.evaluate(q_vec, sdc_results)
        # Check that each purge step improved ESR
        if result.purge_history:
            for round_num, chunk_id, esr_before, esr_after in result.purge_history:
                assert esr_after >= esr_before - 1e-6, \
                    f"Round {round_num}: ESR should not decrease ({esr_before:.3f} → {esr_after:.3f})"

    def test_min_chunks_constraint(self):
        """Purging should stop when min_chunks is reached."""
        config = CPGConfig(theta_cpg=1000.0, min_chunks=3)  # impossible threshold
        cpg = ContextPoisonGuard(config)
        q_vec = make_tve_vec(seed=0)

        sdc_results = [
            make_mock_sdc_result(i, sds_score=0.3 + i*0.02, tve_score=0.7)
            for i in range(8)
        ]

        result = cpg.evaluate(q_vec, sdc_results)
        assert len(result.window) >= config.min_chunks, \
            f"Window should have at least {config.min_chunks} chunks, got {len(result.window)}"

    def test_softmax_weights_sum_to_one(self):
        """Softmax weights must sum to 1."""
        config = CPGConfig()
        cpg = ContextPoisonGuard(config)
        q_vec = make_tve_vec(seed=0)

        sdc_results = [
            make_mock_sdc_result(i, sds_score=0.8, tve_score=0.5 + i*0.05)
            for i in range(8)
        ]

        result = cpg.evaluate(q_vec, sdc_results)
        weight_sum = result.softmax_weights.sum()
        assert abs(weight_sum - 1.0) < 1e-5, f"Weights should sum to 1, got {weight_sum}"

    def test_esr_formula_components(self):
        """Manually verify ESR formula components."""
        config = CPGConfig()
        cpg = ContextPoisonGuard(config)

        # All equal SDS and TVE for predictable calculation
        sds = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        tve = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        weights = cpg._softmax_weights(tve)

        # With equal TVE scores, weights should be equal (1/3 each)
        np.testing.assert_allclose(weights, [1/3, 1/3, 1/3], atol=1e-5)

        esr, p = cpg._compute_esr(sds, weights)
        expected_p = np.mean((1 - sds) * weights)  # ≈ 0.067
        expected_signal = np.sum(sds * weights)     # ≈ 0.8
        expected_esr = expected_signal / (expected_p + 1e-8)

        assert abs(p - expected_p) < 1e-5, f"P mismatch: {p} vs {expected_p}"
        assert abs(esr - expected_esr) < 1e-3, f"ESR mismatch: {esr} vs {expected_esr}"

    def test_empty_sdc_results_handled(self):
        """Empty input should not crash — fallback behavior."""
        cpg = ContextPoisonGuard()
        q_vec = make_tve_vec(seed=0)
        # Only rejected chunks (accepted=False) — CPG should fallback
        sdc_results = [
            make_mock_sdc_result(i, sds_score=0.5, tve_score=0.6)
            for i in range(3)
        ]
        # Mark all as rejected
        for r in sdc_results:
            object.__setattr__(r, 'accepted', False)

        # Should not raise
        result = cpg.evaluate(q_vec, sdc_results)
        assert result is not None
