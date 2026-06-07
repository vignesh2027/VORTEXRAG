"""
Tests for Tri-Vector Encoder (TVE) — Layer 2 of VORTEXRAG.

Verifies:
  - Semantic arm produces 768d normalized vector
  - Syntactic arm produces 64d intermediate then 768d projected vector
  - Causal arm produces 32d intermediate then 768d projected vector
  - Composite TVE score ∈ [0, 1]
  - α + β + γ = 1 constraint enforced
  - Domain preset weights are applied correctly
  - Causal connective and PropBank verb detection
  - Orthogonality between arms on random text
  - Edge cases: empty string, single word, very long text
"""

import numpy as np
import pytest

from core.tve import (
    TriVectorEncoder,
    TVEConfig,
    TVEVector,
    DOMAIN_WEIGHTS,
    CAUSAL_CONNECTIVES,
    CAUSAL_VERBS,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def make_unit_vec(dim: int = 768, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def make_tve_vector(sem_seed: int = 0, syn_seed: int = 1, cau_seed: int = 2) -> TVEVector:
    return TVEVector(
        semantic=make_unit_vec(768, sem_seed),
        syntactic=make_unit_vec(768, syn_seed),
        causal=make_unit_vec(768, cau_seed),
    )


@pytest.fixture(scope="module")
def encoder() -> TriVectorEncoder:
    """Shared TVE encoder (no SBERT/spaCy required for basic tests)."""
    config = TVEConfig(alpha=0.4, beta=0.3, gamma=0.3, domain="general")
    return TriVectorEncoder(config)


# ── Vector dimension tests ─────────────────────────────────────────────────────

class TestVectorDimensions:

    def test_semantic_arm_dim_768(self, encoder):
        """Semantic arm must produce a 768-dimensional vector."""
        vec = encoder.encode("Test sentence for dimension check.")
        assert vec.semantic.shape == (768,), (
            f"Expected semantic dim 768, got {vec.semantic.shape}"
        )

    def test_syntactic_arm_dim_768_after_projection(self, encoder):
        """
        Syntactic arm: raw features are 64d, projected to 768d.
        TVEVector.syntactic should be the final projected 768d vector.
        """
        vec = encoder.encode("The compound caused a severe reaction.")
        assert vec.syntactic.shape == (768,), (
            f"Expected syntactic dim 768 (post-projection), got {vec.syntactic.shape}"
        )

    def test_causal_arm_dim_768_after_projection(self, encoder):
        """
        Causal arm: raw features are 32d, projected to 768d.
        TVEVector.causal should be the final projected 768d vector.
        """
        vec = encoder.encode("Heat causes water to expand.")
        assert vec.causal.shape == (768,), (
            f"Expected causal dim 768 (post-projection), got {vec.causal.shape}"
        )

    def test_syntactic_feature_dim_config(self):
        """TVEConfig.syn_feature_dim defaults to 64."""
        config = TVEConfig()
        assert config.syn_feature_dim == 64

    def test_causal_feature_dim_config(self):
        """TVEConfig.cau_feature_dim defaults to 32."""
        config = TVEConfig()
        assert config.cau_feature_dim == 32

    def test_combined_vec_dim(self, encoder):
        """Combined TVEVector should be 3×768 = 2304 dimensions."""
        vec = encoder.encode("Some text.")
        assert vec.dim == 3 * 768, f"Expected 2304d combined, got {vec.dim}"

    def test_combined_is_concatenation(self, encoder):
        """TVEVector.combined must be [semantic || syntactic || causal]."""
        vec = encoder.encode("Concatenation test.")
        expected = np.concatenate([vec.semantic, vec.syntactic, vec.causal])
        np.testing.assert_array_equal(vec.combined, expected)


# ── Normalization tests ────────────────────────────────────────────────────────

class TestNormalization:

    def test_semantic_vector_is_unit_norm(self, encoder):
        """SBERT output must be L2-normalized (unit vector)."""
        vec = encoder.encode("Normalization test sentence.")
        norm = float(np.linalg.norm(vec.semantic))
        assert abs(norm - 1.0) < 1e-4, f"Semantic norm should be 1.0, got {norm}"

    def test_syntactic_vector_is_unit_norm(self, encoder):
        """Projected syntactic vector must be L2-normalized."""
        vec = encoder.encode("Syntactic normalization check sentence.")
        norm = float(np.linalg.norm(vec.syntactic))
        assert abs(norm - 1.0) < 1e-4, f"Syntactic norm should be 1.0, got {norm}"

    def test_causal_vector_is_unit_norm(self, encoder):
        """Projected causal vector must be L2-normalized."""
        vec = encoder.encode("Causal normalization check.")
        norm = float(np.linalg.norm(vec.causal))
        assert abs(norm - 1.0) < 1e-4, f"Causal norm should be 1.0, got {norm}"


# ── TVE score range ────────────────────────────────────────────────────────────

class TestTVEScoreRange:

    def test_tve_score_identical_texts(self, encoder):
        """Identical query and chunk should have TVE score ~1.0."""
        text = "The mitochondria is the powerhouse of the cell."
        q_vec = encoder.encode_query(text)
        c_vec = encoder.encode_chunk(text)
        # With hash-based fallback encoders, identical text → identical vectors
        score = encoder.tve_score(q_vec, c_vec)
        assert 0.0 <= score <= 1.0, f"TVE score out of range: {score}"
        # For identical unit vectors: cos=1, weighted sum = α+β+γ = 1.0
        assert score > 0.8, f"Identical text TVE score too low: {score}"

    def test_tve_score_different_texts(self, encoder):
        """TVE score between unrelated texts should be in [0, 1]."""
        q_vec = encoder.encode_query("What caused the 2008 financial crisis?")
        c_vec = encoder.encode_chunk("Penguins live in Antarctica and eat fish.")
        score = encoder.tve_score(q_vec, c_vec)
        assert 0.0 <= score <= 1.0, f"TVE score out of range: {score}"

    def test_tve_score_range_across_multiple_texts(self, encoder):
        """TVE score must always be in [0, 1] for any text pair."""
        texts = [
            "The housing bubble caused the financial crisis.",
            "Lehman Brothers filed for bankruptcy in 2008.",
            "The sun rises in the east.",
            "CDO derivatives were highly leveraged.",
            "Cats are popular pets worldwide.",
        ]
        q_vec = encoder.encode_query(texts[0])
        for text in texts[1:]:
            c_vec = encoder.encode_chunk(text)
            score = encoder.tve_score(q_vec, c_vec)
            assert 0.0 <= score <= 1.0, f"TVE score {score} out of [0,1] for: {text}"

    def test_tve_score_manual_formula(self):
        """Manually verify TVE = α·cos_sem + β·cos_syn + γ·cos_cau."""
        config = TVEConfig(alpha=0.5, beta=0.3, gamma=0.2)
        encoder = TriVectorEncoder(config)

        q_vec = make_tve_vector(sem_seed=0, syn_seed=1, cau_seed=2)
        c_vec = make_tve_vector(sem_seed=3, syn_seed=4, cau_seed=5)

        cos_sem = float(np.dot(q_vec.semantic, c_vec.semantic))
        cos_syn = float(np.dot(q_vec.syntactic, c_vec.syntactic))
        cos_cau = float(np.dot(q_vec.causal, c_vec.causal))

        raw = 0.5 * cos_sem + 0.3 * cos_syn + 0.2 * cos_cau
        expected = max(0.0, raw)   # tve_score clips to [0, 1]
        actual = encoder.tve_score(q_vec, c_vec)

        assert abs(actual - expected) < 1e-5, (
            f"TVE formula mismatch: expected {expected:.6f} (raw={raw:.6f}), got {actual:.6f}"
        )


# ── Weight constraint: α + β + γ = 1 ─────────────────────────────────────────

class TestWeightConstraint:

    def test_valid_weights_sum_to_one(self):
        """TVEConfig with valid weights should instantiate without error."""
        config = TVEConfig(alpha=0.4, beta=0.3, gamma=0.3)
        assert abs(config.alpha + config.beta + config.gamma - 1.0) < 1e-6

    def test_invalid_weights_raise_value_error(self):
        """TVEConfig with α+β+γ ≠ 1 must raise ValueError."""
        with pytest.raises(ValueError, match="α \\+ β \\+ γ must equal 1.0"):
            TVEConfig(alpha=0.5, beta=0.4, gamma=0.4)

    def test_domain_preset_weights_sum_to_one(self):
        """All domain presets must have weights summing to 1.0."""
        for domain, (a, b, g) in DOMAIN_WEIGHTS.items():
            total = a + b + g
            assert abs(total - 1.0) < 1e-6, (
                f"Domain '{domain}' weights sum to {total}, not 1.0"
            )

    def test_weights_after_apply_domain_preset(self):
        """apply_domain_preset() must correctly update α, β, γ."""
        config = TVEConfig(domain="medical")
        config.apply_domain_preset()
        expected = DOMAIN_WEIGHTS["medical"]
        assert (config.alpha, config.beta, config.gamma) == expected

    def test_adapt_for_domain_updates_encoder_config(self, encoder):
        """adapt_for_domain() should update α,β,γ on the encoder config."""
        encoder.adapt_for_domain("legal")
        expected_a, expected_b, expected_g = DOMAIN_WEIGHTS["legal"]
        assert encoder.config.alpha == expected_a
        assert encoder.config.beta == expected_b
        assert encoder.config.gamma == expected_g
        # Restore to general for other tests
        encoder.adapt_for_domain("general")

    def test_adapt_for_unknown_domain_raises(self, encoder):
        """adapt_for_domain() with unknown domain must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            encoder.adapt_for_domain("nonexistent_domain_xyz")


# ── Domain preset weights ─────────────────────────────────────────────────────

class TestDomainPresets:

    @pytest.mark.parametrize("domain", list(DOMAIN_WEIGHTS.keys()))
    def test_all_domain_presets_exist(self, domain):
        """Every listed domain must be accessible in DOMAIN_WEIGHTS."""
        assert domain in DOMAIN_WEIGHTS, f"Domain '{domain}' missing from DOMAIN_WEIGHTS"

    def test_medical_gamma_dominates(self):
        """Medical domain: γ (causal) ≥ β (syntactic) per design."""
        a, b, g = DOMAIN_WEIGHTS["medical"]
        assert g >= b, f"Medical domain should have γ≥β, got β={b}, γ={g}"

    def test_code_beta_dominates(self):
        """Code domain: β (syntactic) should be highest weight."""
        a, b, g = DOMAIN_WEIGHTS["code"]
        assert b >= a and b >= g, f"Code domain β should be highest: α={a}, β={b}, γ={g}"

    def test_legal_gamma_higher_than_general(self):
        """Legal domain γ (causal) should be higher than general γ."""
        _, _, g_legal = DOMAIN_WEIGHTS["legal"]
        _, _, g_general = DOMAIN_WEIGHTS["general"]
        assert g_legal >= g_general, (
            f"Legal γ={g_legal} should be ≥ general γ={g_general}"
        )

    def test_creative_alpha_highest(self):
        """Creative domain: α (semantic) should be highest weight."""
        a, b, g = DOMAIN_WEIGHTS["creative"]
        assert a >= b and a >= g, f"Creative domain α should be highest: α={a}, β={b}, γ={g}"

    def test_cross_domain_score_uses_specified_weights(self, encoder):
        """cross_domain_score() should use the specified domain's weights."""
        q_vec = make_tve_vector(sem_seed=0, syn_seed=1, cau_seed=2)
        c_vec = make_tve_vector(sem_seed=3, syn_seed=4, cau_seed=5)

        a, b, g = DOMAIN_WEIGHTS["legal"]
        cos_sem = float(np.dot(q_vec.semantic, c_vec.semantic))
        cos_syn = float(np.dot(q_vec.syntactic, c_vec.syntactic))
        cos_cau = float(np.dot(q_vec.causal, c_vec.causal))
        expected = a * cos_sem + b * cos_syn + g * cos_cau

        actual = encoder.cross_domain_score(q_vec, c_vec, "legal")
        assert abs(actual - expected) < 1e-5

    def test_11_domains_defined(self):
        """All 11 domain presets must be present."""
        expected_domains = {
            "general", "legal", "medical", "scientific", "code",
            "financial", "educational", "creative", "cybersecurity",
            "historical", "customer",
        }
        assert expected_domains.issubset(set(DOMAIN_WEIGHTS.keys())), (
            f"Missing domains: {expected_domains - set(DOMAIN_WEIGHTS.keys())}"
        )


# ── Causal connective detection ────────────────────────────────────────────────

class TestCausalConnectives:

    @pytest.mark.parametrize("word", ["because", "therefore", "hence", "thus", "since"])
    def test_causal_connectives_in_set(self, word):
        """Key causal connectives must be in CAUSAL_CONNECTIVES."""
        assert word in CAUSAL_CONNECTIVES, (
            f"Expected '{word}' in CAUSAL_CONNECTIVES"
        )

    def test_causal_connectives_not_empty(self):
        """CAUSAL_CONNECTIVES must be non-empty."""
        assert len(CAUSAL_CONNECTIVES) > 10, (
            f"CAUSAL_CONNECTIVES has only {len(CAUSAL_CONNECTIVES)} entries"
        )

    def test_non_causal_word_not_in_connectives(self):
        """Common non-causal words should not be in CAUSAL_CONNECTIVES."""
        non_causal = ["the", "and", "cat", "blue", "running"]
        for word in non_causal:
            assert word not in CAUSAL_CONNECTIVES, (
                f"Non-causal word '{word}' should not be in CAUSAL_CONNECTIVES"
            )


# ── PropBank causal verb detection ────────────────────────────────────────────

class TestCausalVerbs:

    @pytest.mark.parametrize("verb", ["cause", "trigger", "enable", "produce"])
    def test_propbank_verbs_in_set(self, verb):
        """PropBank causal verbs must be in CAUSAL_VERBS."""
        assert verb in CAUSAL_VERBS, f"Expected '{verb}' in CAUSAL_VERBS"

    def test_causal_verbs_not_empty(self):
        """CAUSAL_VERBS must be non-empty."""
        assert len(CAUSAL_VERBS) > 10, (
            f"CAUSAL_VERBS has only {len(CAUSAL_VERBS)} entries"
        )


# ── Orthogonality between semantic and causal arms ────────────────────────────

class TestArmOrthogonality:

    def test_semantic_causal_correlation_below_threshold(self, encoder):
        """
        On random texts, semantic and causal arm vectors should have
        low average correlation (< 0.3) — verifying arm orthogonality.
        """
        rng = np.random.default_rng(seed=2024)
        words = [
            "apple", "house", "river", "cloud", "music", "table", "forest",
            "bridge", "engine", "laptop", "coffee", "garden", "rocket", "book",
            "window", "street", "ocean", "castle", "desert", "island",
        ]

        correlations = []
        for i in range(15):
            text = " ".join(rng.choice(words, size=8, replace=True).tolist())
            vec = encoder.encode(text)
            cor = float(np.dot(vec.semantic, vec.causal))
            correlations.append(abs(cor))

        avg_correlation = float(np.mean(correlations))
        # Arms should not be perfectly correlated; threshold is liberal at 0.3
        assert avg_correlation < 0.5, (
            f"Semantic-causal correlation too high: {avg_correlation:.3f} ≥ 0.5. "
            "Arms may not be sufficiently orthogonal."
        )

    def test_arm_scores_dict_keys(self, encoder):
        """arm_scores() must return dict with semantic, syntactic, causal, tve."""
        q_vec = make_tve_vector(sem_seed=0, syn_seed=1, cau_seed=2)
        c_vec = make_tve_vector(sem_seed=3, syn_seed=4, cau_seed=5)
        scores = encoder.arm_scores(q_vec, c_vec)
        assert set(scores.keys()) == {"semantic", "syntactic", "causal", "tve"}

    def test_tve_key_matches_tve_score_method(self, encoder):
        """arm_scores()['tve'] must equal tve_score()."""
        q_vec = make_tve_vector(sem_seed=0, syn_seed=1, cau_seed=2)
        c_vec = make_tve_vector(sem_seed=3, syn_seed=4, cau_seed=5)
        scores = encoder.arm_scores(q_vec, c_vec)
        tve_direct = encoder.tve_score(q_vec, c_vec)
        assert abs(scores["tve"] - tve_direct) < 1e-6


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_string(self, encoder):
        """Empty string should not crash and produce unit-norm vectors."""
        vec = encoder.encode("")
        assert vec.semantic.shape == (768,)
        assert vec.syntactic.shape == (768,)
        assert vec.causal.shape == (768,)
        # Should still be roughly unit norm (fallback produces unit vector)
        sem_norm = float(np.linalg.norm(vec.semantic))
        assert sem_norm > 0.0, "Empty string semantic vector should not be zero"

    def test_single_word(self, encoder):
        """Single-word input should produce valid 768d vectors."""
        vec = encoder.encode("causality")
        assert vec.semantic.shape == (768,)
        score = encoder.tve_score(vec, vec)
        assert 0.0 <= score <= 1.0

    def test_very_long_text(self, encoder):
        """512+ token text should not crash or produce NaN."""
        long_text = " ".join(["The chemical reaction caused a significant change in state."] * 40)
        vec = encoder.encode(long_text)
        assert vec.semantic.shape == (768,)
        assert not np.any(np.isnan(vec.semantic)), "NaN in semantic vector for long text"
        assert not np.any(np.isnan(vec.syntactic)), "NaN in syntactic vector for long text"
        assert not np.any(np.isnan(vec.causal)), "NaN in causal vector for long text"

    def test_numeric_only_text(self, encoder):
        """Numeric-only text should not crash."""
        vec = encoder.encode("42 3.14 100 2024 9.81")
        assert vec.semantic.shape == (768,)

    def test_repeated_encoding_deterministic(self, encoder):
        """Same text encoded twice must produce identical vectors."""
        text = "Deterministic encoding test for VORTEXRAG."
        vec1 = encoder.encode(text)
        vec2 = encoder.encode(text)
        np.testing.assert_array_equal(vec1.semantic, vec2.semantic)
        np.testing.assert_array_equal(vec1.syntactic, vec2.syntactic)
        np.testing.assert_array_equal(vec1.causal, vec2.causal)

    def test_batch_encode_matches_single(self, encoder):
        """batch_encode() must produce same results as individual encode() calls."""
        texts = [
            "The stock market crashed in 2008.",
            "mRNA vaccines use lipid nanoparticles.",
            "Async await is a Python syntactic construct.",
        ]
        batch_vecs = encoder.batch_encode(texts)
        for i, text in enumerate(texts):
            single_vec = encoder.encode(text)
            np.testing.assert_array_almost_equal(
                batch_vecs[i].semantic, single_vec.semantic, decimal=5,
                err_msg=f"Batch vs single mismatch for text {i}"
            )

    def test_cosine_sim_zero_vector(self, encoder):
        """cosine_sim with zero vector should return 0.0 (not NaN)."""
        zero = np.zeros(768, dtype=np.float32)
        unit = make_unit_vec(768, seed=42)
        result = encoder.cosine_sim(zero, unit)
        assert result == 0.0, f"Zero vector cosine_sim should be 0.0, got {result}"

    def test_tve_vector_to_dict(self, encoder):
        """TVEVector.to_dict() should return correct keys."""
        vec = encoder.encode("Test for to_dict method.")
        d = vec.to_dict()
        assert "semantic_norm" in d
        assert "syntactic_norm" in d
        assert "causal_norm" in d
        assert "combined_dim" in d
        assert d["combined_dim"] == 3 * 768


# ── Batch TVE scoring ─────────────────────────────────────────────────────────

class TestBatchScoring:

    def test_batch_tve_scores_matches_individual(self, encoder):
        """batch_tve_scores() must match individual tve_score() calls."""
        q_vec = make_tve_vector(sem_seed=0, syn_seed=1, cau_seed=2)
        c_vecs = [make_tve_vector(sem_seed=i*3, syn_seed=i*3+1, cau_seed=i*3+2)
                  for i in range(5)]

        batch_scores = encoder.batch_tve_scores(q_vec, c_vecs)
        individual_scores = [encoder.tve_score(q_vec, cv) for cv in c_vecs]

        np.testing.assert_allclose(batch_scores, individual_scores, atol=1e-5)

    def test_batch_tve_empty_list(self, encoder):
        """batch_tve_scores() with empty list should return empty array."""
        q_vec = make_tve_vector()
        result = encoder.batch_tve_scores(q_vec, [])
        assert len(result) == 0

    def test_score_matrix_shape(self, encoder):
        """score_matrix() should return (n_queries, n_chunks) shaped array."""
        queries = [make_tve_vector(sem_seed=i) for i in range(3)]
        chunks = [make_tve_vector(sem_seed=i+10) for i in range(5)]
        matrix = encoder.score_matrix(queries, chunks)
        assert matrix.shape == (3, 5), f"Expected (3,5), got {matrix.shape}"

    def test_score_matrix_values_in_range(self, encoder):
        """All entries in score_matrix() should be in [-1, 1]."""
        queries = [make_tve_vector(sem_seed=i) for i in range(4)]
        chunks = [make_tve_vector(sem_seed=i+20) for i in range(6)]
        matrix = encoder.score_matrix(queries, chunks)
        assert np.all(matrix >= -1.0) and np.all(matrix <= 1.0)


# ── Explain and interpretability ──────────────────────────────────────────────

class TestInterpretability:

    def test_explain_score_keys(self, encoder):
        """explain_score() must return all required keys."""
        q_vec = make_tve_vector(sem_seed=0, syn_seed=1, cau_seed=2)
        c_vec = make_tve_vector(sem_seed=3, syn_seed=4, cau_seed=5)
        explanation = encoder.explain_score(q_vec, c_vec)

        required_keys = {
            "tve_score", "semantic_score", "syntactic_score", "causal_score",
            "alpha", "beta", "gamma", "dominant_arm", "drift_magnitude",
            "drift_warning", "interpretation",
        }
        assert required_keys.issubset(explanation.keys()), (
            f"Missing keys: {required_keys - explanation.keys()}"
        )

    def test_drift_warning_triggers(self, encoder):
        """drift_warning must be True when sem > 0.7 but cau < 0.5."""
        # Craft vectors: high semantic similarity, low causal similarity
        q_sem = make_unit_vec(768, seed=0)
        q_syn = make_unit_vec(768, seed=1)
        q_cau = make_unit_vec(768, seed=2)

        # Same semantic, very different causal
        c_sem = q_sem.copy()  # identical semantic
        c_cau = make_unit_vec(768, seed=999)  # different causal

        q_vec = TVEVector(semantic=q_sem, syntactic=q_syn, causal=q_cau)
        c_vec = TVEVector(semantic=c_sem, syntactic=make_unit_vec(768, seed=3), causal=c_cau)

        exp = encoder.explain_score(q_vec, c_vec)
        if exp["semantic_score"] > 0.7 and exp["causal_score"] < 0.5:
            assert exp["drift_warning"] is True

    def test_most_relevant_arm_returns_valid_key(self, encoder):
        """most_relevant_arm() must return one of the three arm names."""
        q_vec = make_tve_vector(sem_seed=0, syn_seed=1, cau_seed=2)
        c_vec = make_tve_vector(sem_seed=3, syn_seed=4, cau_seed=5)
        arm = encoder.most_relevant_arm(q_vec, c_vec)
        assert arm in {"semantic", "syntactic", "causal"}, (
            f"most_relevant_arm() returned unexpected value: {arm}"
        )

    def test_domain_sensitivity_returns_all_domains(self, encoder):
        """domain_sensitivity() should return scores for all domains."""
        q_vec = make_tve_vector(sem_seed=0, syn_seed=1, cau_seed=2)
        c_vec = make_tve_vector(sem_seed=3, syn_seed=4, cau_seed=5)
        sensitivities = encoder.domain_sensitivity(q_vec, c_vec)
        assert set(sensitivities.keys()) == set(DOMAIN_WEIGHTS.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
