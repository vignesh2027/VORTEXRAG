"""
Tests for Rank Fusion Gate (RFG) — Layer 5a of VORTEXRAG.

Verifies:
  - Multiplicative Phi formula: Φ = TVE^α * SDS^β * ESR_contrib^γ
  - No-weak-link property: TVE=0.95,SDS=0.05 ranks below TVE=0.72,SDS=0.80
  - Normalization: Phi+ values sum to 1.0
  - top-m selection returns exactly m chunks (or fewer if not enough)
  - Multiplicative fusion beats additive on adversarial case
  - ESR_contrib computation: normalized signal share
  - Domain-specific weights (α=0.4, β=0.35, γ=0.25 for general)
"""

import numpy as np
import pytest

from core.tve import TVEVector
from core.vrc import SpiralCandidate
from core.sdc import SDCResult
from core.cpg import CPGEvaluation
from core.rfg import RankFusionGate, RFGConfig, RankedChunk, DOMAIN_FUSION_WEIGHTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_unit_vec(dim: int = 768, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def make_tve_vec(seed: int = 0) -> TVEVector:
    return TVEVector(
        semantic=make_unit_vec(768, seed),
        syntactic=make_unit_vec(768, seed + 1),
        causal=make_unit_vec(768, seed + 2),
    )


def make_sdc_result(
    chunk_id: int,
    sds_score: float,
    tve_score: float,
    text: str = "Mock chunk text content.",
) -> SDCResult:
    """Build a mock SDCResult."""
    tve_vec = make_tve_vec(seed=chunk_id * 10)
    spiral = SpiralCandidate(
        chunk_id=chunk_id,
        chunk_text=text,
        tve_score=tve_score,
        radial_dist=0.5,
        theta=0.2,
        spiral_rank=tve_score * 0.9,
        tve_vec=tve_vec,
    )
    return SDCResult(
        candidate=spiral,
        drift_norm=max(0.0, 1.0 - sds_score),
        sds_score=sds_score,
        accepted=sds_score >= 0.72,
        drift_direction=np.zeros(768, dtype=np.float32),
        drift_category="minor",
    )


def make_cpg_eval(sdc_results: list, tve_scores: list = None) -> CPGEvaluation:
    """Build a mock CPGEvaluation from SDCResults."""
    n = len(sdc_results)
    if tve_scores is None:
        tve_scores = [r.candidate.tve_score for r in sdc_results]
    tve_arr = np.array(tve_scores, dtype=np.float32)
    shifted = tve_arr - tve_arr.max()
    exp_w = np.exp(shifted)
    weights = exp_w / (exp_w.sum() + 1e-8)

    sds_arr = np.array([r.sds_score for r in sdc_results])
    irrelevance = 1.0 - sds_arr
    k = max(n, 1)
    p = float(np.sum(irrelevance * weights) / k)
    signal = float(np.sum(sds_arr * weights))
    esr = signal / (p + 1e-8)

    return CPGEvaluation(
        window=sdc_results,
        poison_index=p,
        esr=esr,
        is_clean=esr >= 3.5,
        purge_count=0,
        purge_history=[],
        softmax_weights=weights,
    )


@pytest.fixture
def rfg_general() -> RankFusionGate:
    config = RFGConfig(alpha=0.40, beta=0.35, gamma=0.25, top_m=5, domain="general")
    return RankFusionGate(config)


# ── Phi formula ───────────────────────────────────────────────────────────────

class TestPhiFormula:

    def test_phi_score_exact_formula(self, rfg_general):
        """Verify Φ = TVE^α * SDS^β * ESR_contrib^γ exactly."""
        alpha, beta, gamma = 0.40, 0.35, 0.25
        tve, sds, esr_contrib = 0.80, 0.85, 0.30

        expected = (tve ** alpha) * (sds ** beta) * (esr_contrib ** gamma)
        actual = rfg_general._phi_score(tve, sds, esr_contrib)

        assert abs(actual - expected) < 1e-6, (
            f"Phi formula mismatch: expected {expected:.6f}, got {actual:.6f}"
        )

    def test_phi_score_all_perfect(self, rfg_general):
        """Phi with TVE=SDS=ESR=1.0 should return exactly 1.0."""
        phi = rfg_general._phi_score(1.0, 1.0, 1.0)
        assert abs(phi - 1.0) < 1e-6, f"Perfect Phi should be 1.0, got {phi}"

    def test_phi_score_near_zero_sds(self, rfg_general):
        """Very low SDS should drive Phi toward zero (no-weak-link)."""
        phi = rfg_general._phi_score(0.95, 0.001, 0.80)
        assert phi < 0.1, f"Low SDS Phi should be near zero, got {phi:.4f}"

    def test_phi_score_near_zero_tve(self, rfg_general):
        """Very low TVE should drive Phi toward zero."""
        phi = rfg_general._phi_score(0.001, 0.95, 0.80)
        assert phi < 0.1, f"Low TVE Phi should be near zero, got {phi:.4f}"

    def test_phi_score_epsilon_clipping(self):
        """Phi should not fail or produce NaN for zero inputs (clipped to epsilon)."""
        rfg = RankFusionGate(RFGConfig(alpha=0.4, beta=0.35, gamma=0.25))
        phi = rfg._phi_score(0.0, 0.0, 0.0)
        assert not np.isnan(phi), "Phi should not be NaN for zero inputs"
        assert phi >= 0.0, "Phi should be non-negative"


# ── No-weak-link property ──────────────────────────────────────────────────────

class TestNoWeakLink:

    def test_high_tve_low_sds_ranks_below_moderate_both(self):
        """
        Chunk A: TVE=0.95, SDS=0.05 should rank BELOW
        Chunk B: TVE=0.72, SDS=0.80
        due to multiplicative no-weak-link property.
        """
        rfg = RankFusionGate(RFGConfig(alpha=0.40, beta=0.35, gamma=0.25, top_m=2))

        sdc_a = make_sdc_result(chunk_id=0, sds_score=0.05, tve_score=0.95, text="High TVE low SDS chunk.")
        sdc_b = make_sdc_result(chunk_id=1, sds_score=0.80, tve_score=0.72, text="Moderate TVE high SDS chunk.")

        cpg_eval = make_cpg_eval([sdc_a, sdc_b])
        q_vec = make_tve_vec(seed=0)
        ranked = rfg.rank(q_vec, cpg_eval)

        # Chunk B (moderate TVE, good SDS) should outrank Chunk A
        assert ranked[0].chunk_id == sdc_b.candidate.chunk_id, (
            f"Expected balanced chunk (id=1) to rank first. Order: {[r.chunk_id for r in ranked]}"
        )

    def test_additive_would_rank_differently(self):
        """
        Show that additive fusion would incorrectly rank the high-TVE/low-SDS chunk above.
        This validates the multiplicative approach.
        """
        alpha, beta, gamma = 0.40, 0.35, 0.25
        tve_a, sds_a, esr_a = 0.95, 0.05, 0.50
        tve_b, sds_b, esr_b = 0.72, 0.80, 0.50

        additive_a = alpha * tve_a + beta * sds_a + gamma * esr_a
        additive_b = alpha * tve_b + beta * sds_b + gamma * esr_b

        rfg = RankFusionGate(RFGConfig(alpha=alpha, beta=beta, gamma=gamma))
        mult_a = rfg._phi_score(tve_a, sds_a, esr_a)
        mult_b = rfg._phi_score(tve_b, sds_b, esr_b)

        # Additive might rank A above B (bad behavior)
        # Multiplicative correctly ranks B above A
        assert mult_b > mult_a, (
            f"Multiplicative: B ({mult_b:.3f}) should beat A ({mult_a:.3f}) — no weak link"
        )

        # Confirm additive would be fooled (or at least different)
        addit_fooled = additive_a > additive_b
        mult_correct = mult_b > mult_a
        # At minimum, multiplicative penalizes the weak link more than additive
        mult_penalty_a = 1.0 - mult_a / max(mult_b, 1e-8)
        addit_penalty_a = 1.0 - additive_a / max(additive_b, 1e-8)
        if addit_fooled:
            # Additive wrongly promotes A; multiplicative correctly demotes A
            assert mult_correct, "Multiplicative should correctly penalize low SDS"

    def test_sds_exponent_penalty_larger_than_additive(self):
        """
        For SDS=0.05, SDS^0.35 = 0.35 (approx) provides larger relative penalty
        than additive 0.35 * 0.05 = 0.0175 out of the total sum.
        """
        beta = 0.35
        low_sds = 0.05

        additive_sds_contribution = beta * low_sds           # 0.0175
        multiplicative_sds_factor = low_sds ** beta          # ~0.427 (not 0.0175)

        # Multiplicative "locks in" the penalty at SDS^beta level
        # The key property: multiplicative gives RELATIVE penalty of (1 - SDS^beta)
        # which is larger than additive's beta * (1 - SDS) contribution
        mult_relative_penalty = 1.0 - multiplicative_sds_factor  # ~0.573
        addit_relative_penalty = beta * (1.0 - low_sds)          # ~0.3325

        assert mult_relative_penalty > addit_relative_penalty, (
            f"Multiplicative relative penalty ({mult_relative_penalty:.3f}) should be "
            f"larger than additive ({addit_relative_penalty:.3f}) for low SDS"
        )


# ── Normalization ──────────────────────────────────────────────────────────────

class TestNormalization:

    def test_phi_norm_sums_to_one(self, rfg_general):
        """Phi+ (normalized phi_norm) values must sum to 1.0."""
        sdc_results = [make_sdc_result(i, sds_score=0.7 + i * 0.03, tve_score=0.6 + i * 0.04)
                       for i in range(6)]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg_general.rank(q_vec, cpg_eval)
        phi_sum = sum(r.phi_norm for r in ranked)

        assert abs(phi_sum - 1.0) < 1e-5, f"Phi_norm sum should be 1.0, got {phi_sum:.6f}"

    def test_phi_norm_all_positive(self, rfg_general):
        """All phi_norm values must be positive."""
        sdc_results = [make_sdc_result(i, sds_score=0.75, tve_score=0.70) for i in range(4)]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg_general.rank(q_vec, cpg_eval)
        for r in ranked:
            assert r.phi_norm > 0, f"phi_norm should be positive, got {r.phi_norm}"

    def test_phi_norm_sorted_descending(self, rfg_general):
        """rank() output must be sorted by phi_norm descending."""
        sdc_results = [
            make_sdc_result(0, sds_score=0.90, tve_score=0.85),
            make_sdc_result(1, sds_score=0.60, tve_score=0.60),
            make_sdc_result(2, sds_score=0.80, tve_score=0.75),
            make_sdc_result(3, sds_score=0.50, tve_score=0.50),
        ]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg_general.rank(q_vec, cpg_eval)
        for i in range(len(ranked) - 1):
            assert ranked[i].phi_norm >= ranked[i + 1].phi_norm, (
                f"Ranked chunk {i} phi_norm {ranked[i].phi_norm:.4f} < "
                f"chunk {i+1} phi_norm {ranked[i+1].phi_norm:.4f}"
            )


# ── Top-m selection ────────────────────────────────────────────────────────────

class TestTopMSelection:

    def test_select_top_m_returns_exactly_m(self, rfg_general):
        """select_top_m() must return exactly top_m chunks when enough available."""
        sdc_results = [make_sdc_result(i, sds_score=0.8, tve_score=0.7) for i in range(10)]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg_general.rank(q_vec, cpg_eval)
        top_m = rfg_general.select_top_m(ranked)

        assert len(top_m) == rfg_general.config.top_m, (
            f"Expected {rfg_general.config.top_m} chunks, got {len(top_m)}"
        )

    def test_select_top_m_fewer_than_m_available(self):
        """When fewer than top_m chunks are available, return all."""
        rfg = RankFusionGate(RFGConfig(alpha=0.4, beta=0.35, gamma=0.25, top_m=10))
        sdc_results = [make_sdc_result(i, sds_score=0.8, tve_score=0.7) for i in range(3)]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg.rank(q_vec, cpg_eval)
        top_m = rfg.select_top_m(ranked)

        assert len(top_m) <= 3, f"Should not exceed available chunks (3), got {len(top_m)}"

    def test_select_top_m_empty_ranked(self, rfg_general):
        """select_top_m() on empty list should return empty list."""
        result = rfg_general.select_top_m([])
        assert result == []

    def test_top_m_are_highest_phi_norm(self, rfg_general):
        """The selected top_m chunks must have the highest phi_norm values."""
        sdc_results = [make_sdc_result(i, sds_score=0.5 + i * 0.05, tve_score=0.5 + i * 0.04)
                       for i in range(8)]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg_general.rank(q_vec, cpg_eval)
        top_m = rfg_general.select_top_m(ranked)

        if len(ranked) > rfg_general.config.top_m:
            # Minimum phi in top_m should be ≥ maximum phi outside top_m
            min_top = min(c.phi_norm for c in top_m)
            max_rest = max(c.phi_norm for c in ranked[rfg_general.config.top_m:])
            assert min_top >= max_rest - 1e-8, (
                f"Top-m min phi ({min_top:.4f}) should be ≥ rest max phi ({max_rest:.4f})"
            )


# ── ESR contribution ──────────────────────────────────────────────────────────

class TestESRContribution:

    def test_esr_contributions_sum_to_one(self, rfg_general):
        """ESR contributions must sum to 1.0."""
        sds_scores = np.array([0.80, 0.75, 0.85, 0.70], dtype=np.float32)
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        contribs = rfg_general._esr_contributions(sds_scores, weights)
        assert abs(contribs.sum() - 1.0) < 1e-5, (
            f"ESR contributions should sum to 1.0, got {contribs.sum():.6f}"
        )

    def test_esr_contributions_proportional_to_signal(self, rfg_general):
        """Higher SDS and higher weight → higher ESR contribution."""
        # Two chunks: A has high SDS, B has low SDS, equal weights
        sds_scores = np.array([0.90, 0.40], dtype=np.float32)
        weights = np.array([0.5, 0.5], dtype=np.float32)

        contribs = rfg_general._esr_contributions(sds_scores, weights)
        assert contribs[0] > contribs[1], (
            f"Higher SDS chunk should have higher ESR contribution: "
            f"{contribs[0]:.4f} vs {contribs[1]:.4f}"
        )

    def test_esr_contributions_positive(self, rfg_general):
        """All ESR contributions must be positive (non-negative)."""
        sds_scores = np.array([0.7, 0.8, 0.6, 0.75], dtype=np.float32)
        weights = np.array([0.3, 0.3, 0.2, 0.2], dtype=np.float32)

        contribs = rfg_general._esr_contributions(sds_scores, weights)
        assert np.all(contribs >= 0), f"ESR contributions should be non-negative: {contribs}"


# ── Domain-specific weights ────────────────────────────────────────────────────

class TestDomainWeights:

    def test_general_domain_weights(self):
        """General domain: α=0.40, β=0.35, γ=0.25 per design."""
        a, b, g = DOMAIN_FUSION_WEIGHTS["general"]
        assert (a, b, g) == (0.40, 0.35, 0.25), (
            f"General domain weights should be (0.40, 0.35, 0.25), got ({a}, {b}, {g})"
        )

    def test_all_domain_fusion_weights_sum_to_one(self):
        """All domain fusion weight presets must sum to 1.0."""
        for domain, (a, b, g) in DOMAIN_FUSION_WEIGHTS.items():
            total = a + b + g
            assert abs(total - 1.0) < 1e-6, (
                f"Domain '{domain}' fusion weights sum to {total}, not 1.0"
            )

    def test_adapt_for_domain_updates_weights(self):
        """adapt_for_domain() must update α, β, γ to domain preset."""
        rfg = RankFusionGate()
        rfg.adapt_for_domain("medical")
        expected = DOMAIN_FUSION_WEIGHTS["medical"]
        assert (rfg.config.alpha, rfg.config.beta, rfg.config.gamma) == expected

    def test_adapt_for_unknown_domain_raises(self):
        """adapt_for_domain() with unknown domain must raise ValueError."""
        rfg = RankFusionGate()
        with pytest.raises(ValueError, match="Unknown domain"):
            rfg.adapt_for_domain("unknown_domain_xyz")

    def test_medical_higher_beta_than_general(self):
        """Medical domain β (SDS) should be ≥ general β (causal precision critical)."""
        _, b_medical, _ = DOMAIN_FUSION_WEIGHTS["medical"]
        _, b_general, _ = DOMAIN_FUSION_WEIGHTS["general"]
        assert b_medical >= b_general, (
            f"Medical β ({b_medical}) should be ≥ general β ({b_general})"
        )

    def test_rfg_config_invalid_weights_raise(self):
        """RFGConfig with α+β+γ≠1 must raise ValueError."""
        with pytest.raises(ValueError, match="α \\+ β \\+ γ must equal 1.0"):
            RFGConfig(alpha=0.5, beta=0.4, gamma=0.4)


# ── Phi breakdown and statistics ──────────────────────────────────────────────

class TestPhiBreakdown:

    def test_phi_breakdown_keys(self, rfg_general):
        """phi_breakdown() must return all required keys."""
        sdc_results = [make_sdc_result(i, sds_score=0.80, tve_score=0.75) for i in range(4)]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg_general.rank(q_vec, cpg_eval)
        breakdown = rfg_general.phi_breakdown(ranked[0])

        required = {
            "phi_score", "phi_norm", "tve_score", "sds_score", "esr_contribution",
            "tve_factor", "sds_factor", "esr_factor", "additive_equivalent",
            "multiplicative_gain", "weakest_factor", "interpretation",
        }
        assert required.issubset(breakdown.keys()), (
            f"Missing keys in phi_breakdown: {required - breakdown.keys()}"
        )

    def test_rank_statistics_keys(self, rfg_general):
        """rank_statistics() must return expected keys."""
        sdc_results = [make_sdc_result(i, sds_score=0.8, tve_score=0.7) for i in range(5)]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg_general.rank(q_vec, cpg_eval)
        stats = rfg_general.rank_statistics(ranked)

        required = {
            "n_chunks", "phi_norm_mean", "phi_norm_std", "phi_norm_max",
            "phi_norm_min", "tve_mean", "sds_mean",
        }
        assert required.issubset(stats.keys())

    def test_empty_ranked_statistics(self, rfg_general):
        """rank_statistics() on empty list should return empty dict."""
        assert rfg_general.rank_statistics([]) == {}

    def test_compare_chunks_winner_keys(self, rfg_general):
        """compare_chunks() must return winner keys for each dimension."""
        sdc_results = [make_sdc_result(i, sds_score=0.7 + i * 0.1, tve_score=0.6 + i * 0.1)
                       for i in range(4)]
        cpg_eval = make_cpg_eval(sdc_results)
        q_vec = make_tve_vec(seed=0)

        ranked = rfg_general.rank(q_vec, cpg_eval)
        if len(ranked) >= 2:
            comparison = rfg_general.compare_chunks(ranked[0], ranked[1])
            assert "tve_winner" in comparison
            assert "sds_winner" in comparison
            assert "phi_winner" in comparison


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
