"""
Tests for Semantic Drift Corrector (SDC).

Verifies that:
  1. SDS = 1 when chunk is identical to query (zero drift)
  2. SDS approaches 0 for maximally drifted chunks
  3. SDC gate correctly accepts/rejects based on δ_SDC
  4. Domain-tuned tau values change sensitivity
"""

import numpy as np
import pytest

from core.tve import TriVectorEncoder, TVEConfig, TVEVector
from core.sdc import SemanticDriftCorrector, SDCConfig


def make_zero_vec(dim=768) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)

def make_unit_vec(dim=768, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)

def make_tve_vec(sem_seed=0, syn_seed=1, cau_seed=2, dim=768) -> TVEVector:
    return TVEVector(
        semantic=make_unit_vec(dim, sem_seed),
        syntactic=make_unit_vec(dim, syn_seed),
        causal=make_unit_vec(dim, cau_seed),
    )


class TestSDCScoring:

    def test_zero_drift_gives_sds_one(self):
        """When chunk causal vec == query causal vec, D=0 → SDS=1."""
        sdc = SemanticDriftCorrector(SDCConfig(tau=0.8))
        q = make_tve_vec(cau_seed=10)
        c = TVEVector(
            semantic=make_unit_vec(seed=20),
            syntactic=make_unit_vec(seed=21),
            causal=q.causal.copy(),  # identical causal vec
        )
        sds, drift_norm = sdc.sds(q, c)
        assert drift_norm < 1e-6, "Drift should be zero"
        assert abs(sds - 1.0) < 1e-6, f"SDS should be 1.0, got {sds}"

    def test_large_drift_gives_low_sds(self):
        """When chunk causal vec is orthogonal to query, SDS should be low."""
        sdc = SemanticDriftCorrector(SDCConfig(tau=0.8))
        q = make_tve_vec(cau_seed=10)
        # Create maximally different causal vector (negated + scaled)
        c = TVEVector(
            semantic=make_unit_vec(seed=20),
            syntactic=make_unit_vec(seed=21),
            causal=-q.causal * 5.0,  # large opposite drift
        )
        sds, drift_norm = sdc.sds(q, c)
        assert drift_norm > 5.0, "Drift norm should be large"
        assert sds < 0.05, f"SDS should be near 0, got {sds}"

    def test_sds_range(self):
        """SDS must always be in (0, 1]."""
        sdc = SemanticDriftCorrector()
        for seed in range(20):
            q = make_tve_vec(cau_seed=seed)
            c = make_tve_vec(cau_seed=seed + 100)
            sds, _ = sdc.sds(q, c)
            assert 0 < sds <= 1.0, f"SDS out of range: {sds}"

    def test_gate_accepts_low_drift(self):
        """SDC gate should accept chunks with SDS ≥ δ_SDC=0.72."""
        config = SDCConfig(tau=0.8, delta_sdc=0.72)
        sdc = SemanticDriftCorrector(config)
        q = make_tve_vec(cau_seed=0)
        # Small perturbation — should be close to query
        c = TVEVector(
            semantic=make_unit_vec(seed=1),
            syntactic=make_unit_vec(seed=2),
            causal=q.causal + make_unit_vec(seed=99) * 0.05,  # tiny drift
        )
        # Normalize causal
        c = TVEVector(
            semantic=c.semantic,
            syntactic=c.syntactic,
            causal=c.causal / (np.linalg.norm(c.causal) + 1e-8),
        )
        sds, _ = sdc.sds(q, c)
        assert sds >= config.delta_sdc, f"Expected acceptance, SDS={sds}"

    def test_gate_rejects_high_drift(self):
        """SDC gate should reject chunks with SDS < δ_SDC."""
        config = SDCConfig(tau=0.3, delta_sdc=0.72)  # strict tau
        sdc = SemanticDriftCorrector(config)
        q = make_tve_vec(cau_seed=0)
        c = TVEVector(
            semantic=make_unit_vec(seed=1),
            syntactic=make_unit_vec(seed=2),
            causal=make_unit_vec(seed=999),  # random orthogonal causal
        )
        sds, _ = sdc.sds(q, c)
        # With strict tau=0.3, orthogonal causal should fail
        assert sds < config.delta_sdc or sds < 0.5, f"Expected rejection, SDS={sds}"

    def test_domain_tau_legal_stricter_than_general(self):
        """Legal domain should have stricter tau (lower) than general."""
        legal_config = SDCConfig(domain="legal")
        general_config = SDCConfig(domain="general")
        assert legal_config.tau < general_config.tau, \
            f"Legal tau ({legal_config.tau}) should be < general tau ({general_config.tau})"

    def test_drift_vector_direction(self):
        """D(q, c_i) = v_cau(q) − v_cau(c_i) — verify correct direction."""
        sdc = SemanticDriftCorrector()
        q_cau = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        c_cau = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        q = TVEVector(make_unit_vec(dim=4, seed=0), make_unit_vec(dim=4, seed=1), q_cau)
        c = TVEVector(make_unit_vec(dim=4, seed=2), make_unit_vec(dim=4, seed=3), c_cau)
        D = sdc.drift_vector(q, c)
        expected = q_cau - c_cau
        np.testing.assert_allclose(D, expected, rtol=1e-5)

    def test_tau_sensitivity(self):
        """Higher tau should give higher SDS for the same drift."""
        q = make_tve_vec(cau_seed=0)
        c = make_tve_vec(cau_seed=100)  # different causal

        sdc_strict = SemanticDriftCorrector(SDCConfig(tau=0.2))
        sdc_lenient = SemanticDriftCorrector(SDCConfig(tau=2.0))

        sds_strict, _ = sdc_strict.sds(q, c)
        sds_lenient, _ = sdc_lenient.sds(q, c)

        assert sds_lenient > sds_strict, \
            f"Lenient tau should give higher SDS: {sds_lenient:.3f} vs {sds_strict:.3f}"
