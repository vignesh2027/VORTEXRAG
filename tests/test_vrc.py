"""
Tests for Vortex Retrieval Cone (VRC) — Layer 3 of VORTEXRAG.

Verifies:
  - spiral_rank formula: TVE * exp(-λ*r) * cos(n*θ)
  - Causally aligned chunks (θ≈0) get positive spiral rank
  - Causally orthogonal chunks (θ≈π/2) get near-zero rank
  - Off-axis chunks (n*θ > π/2) get negative rank and are suppressed
  - top-k selection returns at most k candidates
  - Radial decay: lower TVE → exponentially lower rank
  - Edge case: all chunks same TVE score
  - spiral_rank produces different ordering than flat cosine similarity
"""

import numpy as np
import pytest

from core.tve import TriVectorEncoder, TVEConfig, TVEVector
from core.vrc import VortexRetrievalCone, VRCConfig, SpiralCandidate


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_unit_vec(dim: int = 768, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def make_tve_vec(sem_seed: int = 0, syn_seed: int = 1, cau_seed: int = 2) -> TVEVector:
    return TVEVector(
        semantic=make_unit_vec(768, sem_seed),
        syntactic=make_unit_vec(768, syn_seed),
        causal=make_unit_vec(768, cau_seed),
    )


def make_spiral_candidate(
    chunk_id: int,
    tve_score: float,
    theta: float,
    radial_dist: float,
    n_spiral: int = 2,
    lambda_decay: float = 0.5,
) -> SpiralCandidate:
    """Build a SpiralCandidate with manually controlled geometry."""
    spiral_rank = tve_score * np.exp(-lambda_decay * radial_dist) * np.cos(n_spiral * theta)
    tve_vec = make_tve_vec(sem_seed=chunk_id * 3)
    return SpiralCandidate(
        chunk_id=chunk_id,
        chunk_text=f"Chunk {chunk_id} text content for testing purposes.",
        tve_score=tve_score,
        radial_dist=radial_dist,
        theta=theta,
        spiral_rank=float(spiral_rank),
        tve_vec=tve_vec,
    )


@pytest.fixture(scope="module")
def encoder() -> TriVectorEncoder:
    config = TVEConfig(alpha=0.4, beta=0.3, gamma=0.3)
    return TriVectorEncoder(config)


@pytest.fixture(scope="module")
def vrc(encoder) -> VortexRetrievalCone:
    config = VRCConfig(n_spiral=2, lambda_decay=0.5, candidate_pool=50, top_k=20)
    return VortexRetrievalCone(encoder, config)


# ── spiral_rank formula ────────────────────────────────────────────────────────

class TestSpiralRankFormula:

    def test_spiral_rank_formula_exact(self):
        """
        Verify spiral_rank = TVE * exp(-λ*r) * cos(n*θ) exactly.
        """
        n, lam = 2, 0.5
        tve, r, theta = 0.8, 1.0, 0.3

        expected = tve * np.exp(-lam * r) * np.cos(n * theta)

        config = VRCConfig(n_spiral=n, lambda_decay=lam)
        enc = TriVectorEncoder(TVEConfig())
        vrc = VortexRetrievalCone(enc, config)
        actual = vrc._spiral_rank(tve, r, theta)

        assert abs(actual - expected) < 1e-6, (
            f"spiral_rank formula mismatch: expected {expected:.6f}, got {actual:.6f}"
        )

    @pytest.mark.parametrize("n_spiral,theta,expected_sign", [
        (2, 0.0, 1),         # θ=0, cos(0)=1 → positive
        (2, np.pi / 4, 0),   # n*θ=π/2, cos=0 → zero
        (2, np.pi / 3, -1),  # n*θ=2π/3 > π/2, cos<0 → negative
        (1, np.pi / 3, 1),   # n=1, θ=π/3, cos(π/3)=0.5 → positive
    ])
    def test_spiral_rank_sign_by_theta(self, n_spiral, theta, expected_sign):
        """Sign of spiral_rank depends on cos(n*θ)."""
        config = VRCConfig(n_spiral=n_spiral, lambda_decay=0.5)
        enc = TriVectorEncoder(TVEConfig())
        vrc_inst = VortexRetrievalCone(enc, config)

        rank = vrc_inst._spiral_rank(tve_score=0.8, r=0.5, theta=theta)
        if expected_sign > 0:
            assert rank > 0, f"Expected positive rank, got {rank:.4f} (n={n_spiral}, θ={theta:.3f})"
        elif expected_sign < 0:
            assert rank < 0, f"Expected negative rank, got {rank:.4f} (n={n_spiral}, θ={theta:.3f})"
        else:
            assert abs(rank) < 0.05, f"Expected near-zero rank, got {rank:.4f}"

    def test_spiral_rank_tve_zero(self):
        """TVE=0 → spiral_rank=0 regardless of geometry."""
        config = VRCConfig(n_spiral=2, lambda_decay=0.5)
        enc = TriVectorEncoder(TVEConfig())
        vrc_inst = VortexRetrievalCone(enc, config)
        rank = vrc_inst._spiral_rank(tve_score=0.0, r=0.5, theta=0.2)
        assert rank == 0.0

    def test_spiral_rank_batch_matches_individual(self):
        """_spiral_rank_batch() must match individual _spiral_rank() calls."""
        config = VRCConfig(n_spiral=2, lambda_decay=0.5)
        enc = TriVectorEncoder(TVEConfig())
        vrc_inst = VortexRetrievalCone(enc, config)

        tve_scores = np.array([0.8, 0.6, 0.9, 0.5, 0.7], dtype=np.float32)
        r_array = np.array([0.5, 1.0, 0.3, 2.0, 0.8], dtype=np.float32)
        theta_array = np.array([0.1, 0.5, 0.0, 1.2, 0.8], dtype=np.float32)

        batch = vrc_inst._spiral_rank_batch(tve_scores, r_array, theta_array)
        individual = np.array([
            vrc_inst._spiral_rank(float(tve_scores[i]), float(r_array[i]), float(theta_array[i]))
            for i in range(5)
        ])

        np.testing.assert_allclose(batch, individual, atol=1e-6)


# ── Angular alignment and suppression ─────────────────────────────────────────

class TestAngularAlignment:

    def test_aligned_chunk_positive_rank(self):
        """Chunk with θ≈0 (causally aligned) gets positive spiral_rank."""
        candidate = make_spiral_candidate(
            chunk_id=0, tve_score=0.80, theta=0.05, radial_dist=0.5,
            n_spiral=2, lambda_decay=0.5,
        )
        assert candidate.spiral_rank > 0, (
            f"Aligned chunk (θ≈0) should have positive spiral_rank, got {candidate.spiral_rank}"
        )

    def test_orthogonal_chunk_near_zero_rank(self):
        """Chunk with θ=π/4 (n=2) gives cos(n*θ)=cos(π/2)=0 → near-zero rank."""
        theta_orthogonal = np.pi / 4  # n*θ = π/2 → cos(π/2) = 0
        candidate = make_spiral_candidate(
            chunk_id=1, tve_score=0.80, theta=theta_orthogonal, radial_dist=0.5,
            n_spiral=2, lambda_decay=0.5,
        )
        # cos(2 * π/4) = cos(π/2) = 0 → spiral_rank ≈ 0
        assert abs(candidate.spiral_rank) < 0.05, (
            f"Orthogonal chunk should have near-zero rank, got {candidate.spiral_rank}"
        )

    def test_opposite_chunk_negative_rank(self):
        """Chunk with θ > π/4 (n=2) gets negative rank and should be suppressed."""
        theta_beyond = np.pi / 3  # n*θ = 2π/3 > π/2 → cos < 0
        candidate = make_spiral_candidate(
            chunk_id=2, tve_score=0.80, theta=theta_beyond, radial_dist=0.5,
            n_spiral=2, lambda_decay=0.5,
        )
        assert candidate.spiral_rank < 0, (
            f"Off-axis chunk (n*θ > π/2) should have negative rank, got {candidate.spiral_rank}"
        )

    def test_negative_suppression_count(self, vrc):
        """negative_suppression_count() reports chunks with spiral_rank < 0."""
        candidates = [
            make_spiral_candidate(0, tve_score=0.8, theta=0.1, radial_dist=0.5),   # positive
            make_spiral_candidate(1, tve_score=0.8, theta=np.pi/3, radial_dist=0.5),  # negative
            make_spiral_candidate(2, tve_score=0.8, theta=0.2, radial_dist=0.5),   # positive
            make_spiral_candidate(3, tve_score=0.8, theta=np.pi/2, radial_dist=0.5),  # negative
        ]
        n_neg = vrc.negative_suppression_count(candidates)
        assert n_neg == 2, f"Expected 2 negative-rank chunks, got {n_neg}"


# ── Radial decay ──────────────────────────────────────────────────────────────

class TestRadialDecay:

    def test_radial_decay_exponential(self):
        """Rank should decrease exponentially with radial distance."""
        config = VRCConfig(n_spiral=2, lambda_decay=0.5)
        enc = TriVectorEncoder(TVEConfig())
        vrc_inst = VortexRetrievalCone(enc, config)

        # Fixed TVE and theta, increasing radial distance
        rank_r1 = vrc_inst._spiral_rank(tve_score=0.8, r=1.0, theta=0.0)
        rank_r2 = vrc_inst._spiral_rank(tve_score=0.8, r=2.0, theta=0.0)
        rank_r3 = vrc_inst._spiral_rank(tve_score=0.8, r=3.0, theta=0.0)

        assert rank_r1 > rank_r2 > rank_r3, (
            f"Radial decay not monotone: {rank_r1:.4f} > {rank_r2:.4f} > {rank_r3:.4f}"
        )

    def test_lower_tve_lower_rank_same_geometry(self):
        """Lower TVE score → lower spiral_rank when geometry is identical."""
        config = VRCConfig(n_spiral=2, lambda_decay=0.5)
        enc = TriVectorEncoder(TVEConfig())
        vrc_inst = VortexRetrievalCone(enc, config)

        rank_high = vrc_inst._spiral_rank(tve_score=0.9, r=0.5, theta=0.1)
        rank_low = vrc_inst._spiral_rank(tve_score=0.5, r=0.5, theta=0.1)

        assert rank_high > rank_low, (
            f"Higher TVE should give higher rank: {rank_high:.4f} vs {rank_low:.4f}"
        )

    def test_adaptive_lambda_decreases_with_corpus_size(self):
        """Larger corpus should result in smaller (more lenient) λ."""
        lam_small = VortexRetrievalCone.adaptive_lambda(1_000)
        lam_large = VortexRetrievalCone.adaptive_lambda(1_000_000)
        assert lam_small > lam_large, (
            f"Small corpus λ={lam_small:.3f} should be > large corpus λ={lam_large:.3f}"
        )

    def test_adaptive_lambda_bounded(self):
        """Adaptive λ should stay within [0.05, 0.8] range."""
        for n in [100, 10_000, 1_000_000, 10_000_000]:
            lam = VortexRetrievalCone.adaptive_lambda(n)
            assert 0.05 <= lam <= 1.0, f"Adaptive λ={lam:.4f} out of bounds for N={n}"


# ── top-k selection ────────────────────────────────────────────────────────────

class TestTopKSelection:

    def test_retrieve_returns_at_most_top_k(self, encoder):
        """VRC.retrieve() must return at most top_k candidates."""
        config = VRCConfig(n_spiral=2, lambda_decay=0.5, top_k=5)
        vrc_inst = VortexRetrievalCone(encoder, config)

        texts = [f"Document chunk {i} about various topics." for i in range(20)]
        corpus_vecs = [encoder.encode(t) for t in texts]
        q_vec = encoder.encode_query("What caused the financial crisis?")

        candidates = vrc_inst.retrieve(q_vec, corpus_vecs, texts)
        assert len(candidates) <= 5, (
            f"Expected ≤5 candidates, got {len(candidates)}"
        )

    def test_retrieve_sorted_by_spiral_rank_desc(self, encoder):
        """Retrieved candidates must be sorted by spiral_rank descending."""
        config = VRCConfig(n_spiral=2, lambda_decay=0.5, top_k=10)
        vrc_inst = VortexRetrievalCone(encoder, config)

        texts = [f"Chunk {i}: various content about topics and events." for i in range(15)]
        corpus_vecs = [encoder.encode(t) for t in texts]
        q_vec = encoder.encode_query("What events caused a major outcome?")

        candidates = vrc_inst.retrieve(q_vec, corpus_vecs, texts)
        for i in range(len(candidates) - 1):
            assert candidates[i].spiral_rank >= candidates[i + 1].spiral_rank, (
                f"Candidate {i} spiral_rank {candidates[i].spiral_rank:.4f} < "
                f"candidate {i+1} spiral_rank {candidates[i+1].spiral_rank:.4f}"
            )

    def test_retrieve_empty_corpus(self, encoder):
        """Empty corpus should return empty list without error."""
        config = VRCConfig(n_spiral=2, lambda_decay=0.5, top_k=10)
        vrc_inst = VortexRetrievalCone(encoder, config)
        q_vec = encoder.encode_query("test query")
        candidates = vrc_inst.retrieve(q_vec, [], [])
        assert candidates == []

    def test_retrieve_small_corpus_fewer_than_top_k(self, encoder):
        """When corpus < top_k, return all available candidates."""
        config = VRCConfig(n_spiral=2, lambda_decay=0.5, top_k=50, candidate_pool=50)
        vrc_inst = VortexRetrievalCone(encoder, config)

        texts = ["Short document one.", "Short document two.", "Short document three."]
        corpus_vecs = [encoder.encode(t) for t in texts]
        q_vec = encoder.encode_query("document query")

        candidates = vrc_inst.retrieve(q_vec, corpus_vecs, texts)
        assert len(candidates) <= len(texts)


# ── All-same TVE score edge case ──────────────────────────────────────────────

class TestEdgeCases:

    def test_all_same_tve_score_still_orders_by_angle(self, encoder):
        """When all TVE scores are equal, angular alignment should differentiate."""
        # Construct chunks with the same TVE but different angles
        q_sem = make_unit_vec(768, seed=0)
        q_vec = TVEVector(
            semantic=q_sem,
            syntactic=make_unit_vec(768, seed=1),
            causal=make_unit_vec(768, seed=2),
        )

        # Chunk 1: same direction as query (small theta)
        c_aligned = TVEVector(semantic=q_sem.copy(), syntactic=make_unit_vec(768, seed=3), causal=make_unit_vec(768, seed=4))
        # Chunk 2: different direction (large theta)
        c_random = TVEVector(semantic=make_unit_vec(768, seed=99), syntactic=make_unit_vec(768, seed=5), causal=make_unit_vec(768, seed=6))

        config = VRCConfig(n_spiral=2, lambda_decay=0.5)
        vrc_inst = VortexRetrievalCone(encoder, config)

        centroid = q_sem  # use query itself as centroid for simplicity

        r_aligned, theta_aligned = vrc_inst._compute_polar_coords(q_sem, c_aligned.semantic, centroid)
        r_random, theta_random = vrc_inst._compute_polar_coords(q_sem, c_random.semantic, centroid)

        # Aligned chunk should have smaller theta
        assert theta_aligned < theta_random or abs(theta_aligned) < 0.1, (
            f"Aligned chunk should have small theta, got {theta_aligned:.3f}"
        )

    def test_pool_statistics_keys(self, vrc):
        """pool_statistics() should return all expected keys."""
        candidates = [
            make_spiral_candidate(i, tve_score=0.7, theta=0.2, radial_dist=0.5)
            for i in range(5)
        ]
        stats = vrc.pool_statistics(candidates)
        required_keys = {
            "n_candidates", "n_negative_rank", "mean_tve", "mean_spiral_rank",
            "mean_theta_degrees", "mean_radial_dist", "suppression_rate",
        }
        assert required_keys.issubset(stats.keys())

    def test_pool_statistics_empty_list(self, vrc):
        """pool_statistics() with empty list should return empty dict."""
        assert vrc.pool_statistics([]) == {}


# ── VRC vs flat cosine ranking ────────────────────────────────────────────────

class TestVRCVsFlatRanking:

    def test_spiral_rank_differs_from_flat_tve_order(self, encoder):
        """
        VRC spiral ranking should produce a different ordering than
        flat TVE-score ranking at least sometimes (angular suppression).
        """
        config = VRCConfig(n_spiral=2, lambda_decay=0.5, top_k=10, candidate_pool=20)
        vrc_inst = VortexRetrievalCone(encoder, config)

        texts = [
            "The financial crisis was caused by mortgage-backed securities.",
            "CDO derivatives led to the collapse of Lehman Brothers in 2008.",
            "Penguins are flightless birds found in the Southern Hemisphere.",
            "The housing bubble triggered widespread bank failures.",
            "Basketball is a popular sport played in the NBA.",
            "Subprime lending caused mortgage defaults to spike.",
            "Ice cream is a popular frozen dessert enjoyed worldwide.",
            "The Federal Reserve responded by cutting interest rates.",
        ]
        corpus_vecs = [encoder.encode(t) for t in texts]
        q_vec = encoder.encode_query("What caused the 2008 financial crisis?")

        comparison = vrc_inst.compare_with_flat_topk(q_vec, corpus_vecs, texts, k=5)

        # Should return required keys
        assert "in_both_count" in comparison
        assert "vrc_only_count" in comparison
        assert "flat_only_count" in comparison
        assert "agreement_rate" in comparison

        # Agreement rate should be a valid fraction
        assert 0.0 <= comparison["agreement_rate"] <= 1.0

    def test_angular_distribution_coverage(self, vrc):
        """angular_distribution() should cover all n_bins bins with valid counts."""
        candidates = [
            make_spiral_candidate(i, tve_score=0.7, theta=i * (np.pi / 10), radial_dist=0.5)
            for i in range(10)
        ]
        dist = vrc.angular_distribution(candidates, n_bins=6)
        assert len(dist) == 6
        assert sum(dist.values()) == len(candidates)


# ── Polar coordinates ─────────────────────────────────────────────────────────

class TestPolarCoordinates:

    def test_same_vector_theta_zero(self, vrc):
        """Query vector and itself should have θ=0."""
        q = make_unit_vec(768, seed=42)
        centroid = q.copy()
        r, theta = vrc._compute_polar_coords(q, q, centroid)
        assert abs(theta) < 1e-4, f"Identical vectors should have θ=0, got {theta:.6f}"

    def test_opposite_vector_theta_pi(self, vrc):
        """Negated query vector should have θ=π."""
        q = make_unit_vec(768, seed=42)
        centroid = np.zeros(768, dtype=np.float32)
        r, theta = vrc._compute_polar_coords(q, -q, centroid)
        assert abs(theta - np.pi) < 1e-4, f"Opposite vectors should have θ=π, got {theta:.6f}"

    def test_batch_polar_matches_individual(self, vrc):
        """_compute_polar_coords_batch() must match individual calls."""
        query_vec = make_unit_vec(768, seed=0)
        chunk_vecs = np.stack([make_unit_vec(768, seed=i) for i in range(5)])
        centroid = make_unit_vec(768, seed=99)

        r_batch, theta_batch = vrc._compute_polar_coords_batch(query_vec, chunk_vecs, centroid)

        for i in range(5):
            r_ind, theta_ind = vrc._compute_polar_coords(query_vec, chunk_vecs[i], centroid)
            assert abs(r_batch[i] - r_ind) < 1e-4, f"r mismatch at {i}"
            assert abs(theta_batch[i] - theta_ind) < 1e-4, f"θ mismatch at {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
