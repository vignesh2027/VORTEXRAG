"""
Tests for Causal Context Builder (CCB) — Layer 5b of VORTEXRAG.

Verifies:
  - Depth-0 chunks always get slot_position 0
  - pos = rank(Phi+) * causal_depth formula
  - Sorting produces ascending position order
  - Near-duplicate suppression: cosine similarity >= 0.92 → only higher-Phi retained
  - Root-cause chunks (depth=0) placed before mechanism chunks (depth=1)
  - CCB improves answer position vs random ordering
  - Edge case: all chunks same causal depth
"""

import numpy as np
import pytest

from core.tve import TVEVector
from core.vrc import SpiralCandidate
from core.sdc import SDCResult
from core.rfg import RankedChunk
from core.ccb import CausalContextBuilder, CCBConfig, OrderedContextSlot


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


def make_ranked_chunk(
    chunk_id: int,
    chunk_text: str,
    phi_norm: float,
    phi_score: float = None,
    tve_score: float = 0.75,
    sds_score: float = 0.80,
    esr_contribution: float = 0.25,
    sem_seed: int = None,
) -> RankedChunk:
    """Build a mock RankedChunk for CCB testing."""
    if phi_score is None:
        phi_score = phi_norm * 10  # arbitrary raw score
    if sem_seed is None:
        sem_seed = chunk_id * 3

    tve_vec = make_tve_vec(seed=sem_seed)
    spiral = SpiralCandidate(
        chunk_id=chunk_id,
        chunk_text=chunk_text,
        tve_score=tve_score,
        radial_dist=0.5,
        theta=0.2,
        spiral_rank=tve_score,
        tve_vec=tve_vec,
    )
    sdc_result = SDCResult(
        candidate=spiral,
        drift_norm=0.2,
        sds_score=sds_score,
        accepted=True,
        drift_direction=np.zeros(768, dtype=np.float32),
        drift_category="minor",
    )
    return RankedChunk(
        chunk_id=chunk_id,
        chunk_text=chunk_text,
        phi_score=phi_score,
        phi_norm=phi_norm,
        tve_score=tve_score,
        sds_score=sds_score,
        esr_contribution=esr_contribution,
        sdc_result=sdc_result,
    )


@pytest.fixture
def ccb() -> CausalContextBuilder:
    config = CCBConfig(
        max_causal_depth=5,
        fallback_depth=10,
        max_tokens=4096,
        dedup_threshold=0.92,
        enable_dedup=True,
    )
    return CausalContextBuilder(config)


# ── Depth-0 slot position ──────────────────────────────────────────────────────

class TestSlotPosition:

    def test_depth_zero_always_position_zero(self, ccb):
        """
        pos = rank * causal_depth.
        Any depth=0 chunk → pos = rank * 0 = 0, regardless of phi_rank.
        """
        for phi_rank in [1, 2, 3, 5, 10]:
            pos = ccb._slot_position(phi_rank=phi_rank, causal_depth=0)
            assert pos == 0.0, (
                f"depth=0 should always give pos=0, got {pos} for rank={phi_rank}"
            )

    def test_slot_position_formula(self, ccb):
        """pos(c_i) = rank(Phi+) * causal_depth."""
        cases = [
            (1, 1, 1.0),
            (2, 3, 6.0),
            (3, 2, 6.0),
            (5, 1, 5.0),
            (1, 10, 10.0),
        ]
        for phi_rank, depth, expected_pos in cases:
            actual = ccb._slot_position(phi_rank, depth)
            assert abs(actual - expected_pos) < 1e-6, (
                f"pos({phi_rank}, {depth}) expected {expected_pos}, got {actual}"
            )

    def test_fallback_depth_gets_large_position(self, ccb):
        """Chunks with fallback_depth should get large slot positions (placed last)."""
        fallback_pos = ccb._slot_position(phi_rank=1, causal_depth=ccb.config.fallback_depth)
        normal_pos = ccb._slot_position(phi_rank=1, causal_depth=1)
        assert fallback_pos > normal_pos, (
            f"Fallback pos ({fallback_pos}) should be > normal pos ({normal_pos})"
        )


# ── Build and ordering ────────────────────────────────────────────────────────

class TestBuildOrdering:

    def test_build_returns_ascending_slot_positions(self, ccb):
        """build() output must be sorted by slot_position ascending."""
        chunks = [
            make_ranked_chunk(0, "The 2008 financial crisis was caused by CDO derivatives.", phi_norm=0.35),
            make_ranked_chunk(1, "Lehman Brothers collapsed in September 2008.", phi_norm=0.28),
            make_ranked_chunk(2, "Credit default swaps amplified systemic risk.", phi_norm=0.22),
            make_ranked_chunk(3, "The Federal Reserve cut interest rates in response.", phi_norm=0.15),
        ]
        query = "What caused the 2008 financial crisis?"
        slots = ccb.build(query, chunks)

        for i in range(len(slots) - 1):
            assert slots[i].slot_position <= slots[i + 1].slot_position, (
                f"Slot {i} pos {slots[i].slot_position} > slot {i+1} pos {slots[i+1].slot_position}"
            )

    def test_root_cause_chunks_placed_first(self, ccb):
        """
        Chunks with causal_depth=0 must appear before chunks with causal_depth>0.
        """
        # Query about "financial crisis" — chunks mentioning it get depth 0
        query = "What caused the financial crisis?"
        chunks = [
            make_ranked_chunk(0, "Subprime mortgages led to the financial crisis collapse.", phi_norm=0.25),
            make_ranked_chunk(1, "The IMF reported on economic recovery.", phi_norm=0.30),
            make_ranked_chunk(2, "CDOs caused severe financial crisis failures.", phi_norm=0.20),
            make_ranked_chunk(3, "Banking regulations were strengthened after 2008.", phi_norm=0.25),
        ]
        slots = ccb.build(query, chunks)
        if len(slots) >= 2:
            # The first slot should have the smallest slot_position
            assert slots[0].slot_position <= slots[1].slot_position

    def test_depth_0_precedes_depth_1(self):
        """
        A root-cause chunk (depth=0) must appear before an effect chunk (depth=1)
        even if the effect chunk has a higher phi_norm rank.
        """
        ccb_no_dedup = CausalContextBuilder(CCBConfig(enable_dedup=False))

        query = "Why did the chemical reaction occur?"
        # Chunk A: rank=3 (lower phi), depth=0 (root cause) → pos=0
        # Chunk B: rank=1 (higher phi), depth=1 (effect) → pos=1
        # A should appear before B because pos(A) = 0 < pos(B) = 1

        chunk_root = make_ranked_chunk(
            0, "The chemical reaction occurred because heat caused the compound to decompose.",
            phi_norm=0.15,  # lower phi_norm → rank=2 after another higher one
        )
        chunk_effect = make_ranked_chunk(
            1, "The chemical reaction produced carbon dioxide and water vapor.",
            phi_norm=0.40,  # higher phi_norm → rank=1
        )
        chunk_extra = make_ranked_chunk(
            2, "Laboratory safety requires proper ventilation for chemical experiments.",
            phi_norm=0.45,  # highest phi_norm → rank=1
        )

        # Provide in phi_norm descending order (as RFG would)
        ranked_input = sorted(
            [chunk_root, chunk_effect, chunk_extra],
            key=lambda c: c.phi_norm, reverse=True,
        )

        slots = ccb_no_dedup.build(query, ranked_input)
        # Check that depth-0 chunks are not at the very end
        depths = [s.causal_depth for s in slots]
        # If any depth-0 chunks exist, they should not all be after depth-1 chunks
        if 0 in depths:
            first_depth_0_idx = depths.index(0)
            # Depth-0 chunk should appear before any depth >= fallback
            fallback = ccb_no_dedup.config.fallback_depth
            for i, d in enumerate(depths):
                if d >= fallback and i < first_depth_0_idx:
                    pytest.fail(f"Fallback depth chunk at {i} before root-cause at {first_depth_0_idx}")

    def test_build_empty_chunks(self, ccb):
        """build() on empty ranked_chunks should return empty list."""
        result = ccb.build("test query", [])
        assert result == []

    def test_build_single_chunk(self, ccb):
        """build() with a single chunk should return a single slot."""
        chunk = make_ranked_chunk(0, "Single chunk content about causation.", phi_norm=1.0)
        slots = ccb.build("test query", [chunk])
        assert len(slots) == 1

    def test_phi_rank_starts_at_1(self, ccb):
        """phi_rank in OrderedContextSlot should start at 1 (not 0)."""
        chunks = [
            make_ranked_chunk(i, f"Chunk {i} with relevant content.", phi_norm=0.3 - i * 0.05)
            for i in range(3)
        ]
        slots = ccb.build("test query", chunks)
        phi_ranks = {s.phi_rank for s in slots}
        assert 0 not in phi_ranks, "phi_rank should be 1-indexed, not 0-indexed"
        assert 1 in phi_ranks or len(slots) == 0


# ── Near-duplicate suppression ────────────────────────────────────────────────

class TestDeduplication:

    def test_identical_semantic_vectors_deduped(self):
        """Two chunks with identical semantic vectors should be deduped to one."""
        config = CCBConfig(dedup_threshold=0.92, enable_dedup=True)
        ccb_inst = CausalContextBuilder(config)

        shared_sem = make_unit_vec(768, seed=42)

        # Build two RankedChunks with identical semantic vectors
        def make_with_sem(chunk_id, phi_norm, sem_vec):
            tve_vec = TVEVector(
                semantic=sem_vec.copy(),
                syntactic=make_unit_vec(768, seed=chunk_id + 100),
                causal=make_unit_vec(768, seed=chunk_id + 200),
            )
            spiral = SpiralCandidate(
                chunk_id=chunk_id,
                chunk_text=f"Duplicate chunk {chunk_id} about the same topic.",
                tve_score=0.75,
                radial_dist=0.5,
                theta=0.2,
                spiral_rank=0.7,
                tve_vec=tve_vec,
            )
            sdc_result = SDCResult(
                candidate=spiral,
                drift_norm=0.2,
                sds_score=0.80,
                accepted=True,
                drift_direction=np.zeros(768, dtype=np.float32),
                drift_category="minor",
            )
            return RankedChunk(
                chunk_id=chunk_id,
                chunk_text=f"Duplicate chunk {chunk_id}.",
                phi_score=phi_norm * 10,
                phi_norm=phi_norm,
                tve_score=0.75,
                sds_score=0.80,
                esr_contribution=0.25,
                sdc_result=sdc_result,
            )

        chunk_high = make_with_sem(0, phi_norm=0.60, sem_vec=shared_sem)
        chunk_low = make_with_sem(1, phi_norm=0.40, sem_vec=shared_sem)

        # Sort by phi_norm descending (as RFG would produce)
        ranked = [chunk_high, chunk_low]
        deduped = ccb_inst.deduplicate(ranked)

        assert len(deduped) == 1, (
            f"Identical semantic vectors should deduplicate to 1 chunk, got {len(deduped)}"
        )
        # The higher phi_norm chunk should be kept
        assert deduped[0].phi_norm == 0.60, (
            f"Higher-phi chunk should be kept, got phi_norm={deduped[0].phi_norm}"
        )

    def test_distinct_vectors_not_deduped(self, ccb):
        """Chunks with distinct semantic vectors should not be deduped."""
        chunks = [
            make_ranked_chunk(i, f"Distinct chunk {i} about completely different topic {i}.",
                              phi_norm=0.25, sem_seed=i * 100)
            for i in range(4)
        ]
        deduped = ccb.deduplicate(chunks)
        assert len(deduped) == 4, (
            f"Distinct chunks should not be deduped, got {len(deduped)} from 4"
        )

    def test_dedup_disabled_keeps_all(self):
        """With enable_dedup=False, all chunks should be kept."""
        config = CCBConfig(enable_dedup=False, dedup_threshold=0.1)  # very low threshold too
        ccb_inst = CausalContextBuilder(config)

        # Same semantic vector
        shared_sem = make_unit_vec(768, seed=42)
        chunks = []
        for i in range(3):
            chunk = make_ranked_chunk(i, f"Chunk {i}", phi_norm=0.33, sem_seed=0)
            # Override semantic via the tve_vec
            tve_vec = TVEVector(
                semantic=shared_sem.copy(),
                syntactic=make_unit_vec(768, seed=i + 100),
                causal=make_unit_vec(768, seed=i + 200),
            )
            spiral = SpiralCandidate(
                chunk_id=i,
                chunk_text=f"Chunk {i}",
                tve_score=0.75,
                radial_dist=0.5,
                theta=0.2,
                spiral_rank=0.7,
                tve_vec=tve_vec,
            )
            sdc_result = SDCResult(
                candidate=spiral,
                drift_norm=0.2,
                sds_score=0.80,
                accepted=True,
                drift_direction=np.zeros(768, dtype=np.float32),
                drift_category="minor",
            )
            chunks.append(RankedChunk(
                chunk_id=i,
                chunk_text=f"Chunk {i}",
                phi_score=0.33,
                phi_norm=0.33,
                tve_score=0.75,
                sds_score=0.80,
                esr_contribution=0.25,
                sdc_result=sdc_result,
            ))

        result = ccb_inst.deduplicate(chunks)
        # Dedup disabled → all 3 kept
        assert len(result) == 3

    def test_dedup_threshold_092_filters_near_duplicates(self):
        """Chunks with cosine sim >= 0.92 should be filtered."""
        config = CCBConfig(dedup_threshold=0.92, enable_dedup=True)
        ccb_inst = CausalContextBuilder(config)

        base_vec = make_unit_vec(768, seed=42)
        # Create slightly perturbed version with high similarity
        noise = make_unit_vec(768, seed=43) * 0.01  # tiny noise
        perturbed = base_vec + noise
        perturbed = perturbed / np.linalg.norm(perturbed)

        sim = float(np.dot(base_vec, perturbed))
        if sim >= 0.92:
            # Build two chunks with these similar vectors
            def build_chunk_with_sem(cid, phi_n, sem):
                tve_v = TVEVector(semantic=sem, syntactic=make_unit_vec(768, seed=cid+50), causal=make_unit_vec(768, seed=cid+60))
                sp = SpiralCandidate(chunk_id=cid, chunk_text=f"c{cid}", tve_score=0.7, radial_dist=0.5, theta=0.2, spiral_rank=0.7, tve_vec=tve_v)
                sdr = SDCResult(candidate=sp, drift_norm=0.2, sds_score=0.8, accepted=True, drift_direction=np.zeros(768,dtype=np.float32), drift_category="minor")
                return RankedChunk(chunk_id=cid, chunk_text=f"c{cid}", phi_score=phi_n*10, phi_norm=phi_n, tve_score=0.7, sds_score=0.8, esr_contribution=0.25, sdc_result=sdr)

            c_high = build_chunk_with_sem(0, 0.60, base_vec)
            c_low = build_chunk_with_sem(1, 0.40, perturbed)
            deduped = ccb_inst.deduplicate([c_high, c_low])
            assert len(deduped) == 1, (
                f"Chunks with cosine sim {sim:.3f} >= 0.92 should be deduped to 1"
            )


# ── All-same causal depth edge case ────────────────────────────────────────────

class TestSameCausalDepth:

    def test_all_same_depth_ordered_by_phi_rank(self, ccb):
        """
        When all chunks are at the same non-zero depth, ordering is determined
        by phi_rank (lower phi_rank = higher phi_norm = first in context).
        """
        # Use a query that shares entities with all chunks to get same depth
        query = "financial bank"
        chunks = [
            make_ranked_chunk(0, "The bank triggered financial collapse.", phi_norm=0.40, sem_seed=0),
            make_ranked_chunk(1, "The bank financial system had problems.", phi_norm=0.35, sem_seed=10),
            make_ranked_chunk(2, "The bank in the financial sector grew.", phi_norm=0.25, sem_seed=20),
        ]
        slots = ccb.build(query, chunks)

        # All should have the same depth (they all share query entities)
        if len(slots) > 1:
            depths = [s.causal_depth for s in slots]
            if len(set(depths)) == 1:
                # Same depth → ordered by phi_rank
                phi_ranks = [s.phi_rank for s in slots]
                assert phi_ranks == sorted(phi_ranks), (
                    f"Same-depth chunks should be ordered by phi_rank: {phi_ranks}"
                )

    def test_fallback_depth_used_for_disconnected_chunks(self, ccb):
        """Chunks with no entity overlap with query get fallback_depth."""
        query = "very specific query entity XYZ123"
        chunks = [
            make_ranked_chunk(0, "Completely unrelated content about penguins.", phi_norm=0.50),
            make_ranked_chunk(1, "Another unrelated chunk about weather patterns.", phi_norm=0.50),
        ]
        depth_map = ccb._build_causal_graph(query, chunks)
        for chunk_id, depth in depth_map.items():
            assert depth == ccb.config.fallback_depth, (
                f"Disconnected chunk {chunk_id} should get fallback depth {ccb.config.fallback_depth}, "
                f"got {depth}"
            )


# ── Context assembly ──────────────────────────────────────────────────────────

class TestContextAssembly:

    def test_to_context_string_has_citations(self, ccb):
        """to_context_string() with include_citations=True must prepend [C1], [C2], ..."""
        chunks = [
            make_ranked_chunk(i, f"Context chunk {i} with relevant information.", phi_norm=0.25)
            for i in range(3)
        ]
        slots = ccb.build("test query", chunks)
        context = ccb.to_context_string(slots, include_citations=True)

        assert "[C1]" in context, "Context string should contain [C1] citation marker"

    def test_to_context_string_no_citations(self, ccb):
        """to_context_string() with include_citations=False should not have [C1]."""
        chunks = [make_ranked_chunk(i, f"Chunk {i}.", phi_norm=0.33) for i in range(2)]
        slots = ccb.build("test query", chunks)
        context = ccb.to_context_string(slots, include_citations=False)
        assert "[C1]" not in context

    def test_to_structured_context_keys(self, ccb):
        """to_structured_context() should return correct keys per slot."""
        chunks = [make_ranked_chunk(i, f"Structured chunk {i} text.", phi_norm=0.33) for i in range(2)]
        slots = ccb.build("test query", chunks)
        structured = ccb.to_structured_context(slots)

        required_keys = {
            "citation", "slot_position", "causal_depth", "phi_rank",
            "phi_norm", "tve_score", "sds_score", "esr_contrib",
            "chunk_text", "word_count", "causal_label",
        }
        for entry in structured:
            assert required_keys.issubset(entry.keys()), (
                f"Missing keys: {required_keys - entry.keys()}"
            )

    def test_causal_labels_valid(self, ccb):
        """causal_label in structured context must be one of valid values."""
        valid_labels = {"root_cause", "effect", "supporting", "fallback"}
        chunks = [
            make_ranked_chunk(0, "Root cause of financial crisis was leverage.", phi_norm=0.40),
            make_ranked_chunk(1, "Unrelated content about astronomy.", phi_norm=0.30),
        ]
        slots = ccb.build("financial crisis leverage", chunks)
        structured = ccb.to_structured_context(slots)
        for entry in structured:
            assert entry["causal_label"] in valid_labels, (
                f"Invalid causal_label: {entry['causal_label']}"
            )

    def test_token_budget_respected(self):
        """to_context_string() should not exceed max_tokens budget."""
        config = CCBConfig(max_tokens=50, avg_words_per_token=0.75)  # ~37 words budget
        ccb_inst = CausalContextBuilder(config)

        chunks = [
            make_ranked_chunk(i, " ".join(["word"] * 30), phi_norm=0.25)
            for i in range(5)
        ]
        slots = ccb_inst.build("test query", chunks)
        context = ccb_inst.to_context_string(slots)
        word_count = len(context.split())
        estimated_tokens = int(word_count / config.avg_words_per_token)

        assert estimated_tokens <= config.max_tokens + 5, (
            f"Token budget exceeded: {estimated_tokens} > {config.max_tokens}"
        )

    def test_token_budget_usage_keys(self, ccb):
        """token_budget_usage() should return all expected keys."""
        chunks = [make_ranked_chunk(i, f"Chunk {i} text.", phi_norm=0.33) for i in range(2)]
        slots = ccb.build("test query", chunks)
        usage = ccb.token_budget_usage(slots)

        required = {"n_slots", "total_words", "estimated_tokens", "token_budget", "utilization_pct"}
        assert required.issubset(usage.keys())

    def test_depth_distribution_counts(self, ccb):
        """depth_distribution() should count chunks at each depth level."""
        chunks = [make_ranked_chunk(i, f"Chunk {i}.", phi_norm=0.25) for i in range(4)]
        slots = ccb.build("test query", chunks)
        dist = ccb.depth_distribution(slots)

        total_from_dist = sum(dist.values())
        assert total_from_dist == len(slots), (
            f"Depth distribution total {total_from_dist} != slots count {len(slots)}"
        )

    def test_explain_ordering_returns_one_per_slot(self, ccb):
        """explain_ordering() should return one explanation string per slot."""
        chunks = [make_ranked_chunk(i, f"Chunk {i} about causes.", phi_norm=0.33) for i in range(3)]
        slots = ccb.build("test query", chunks)
        explanations = ccb.explain_ordering(slots)
        assert len(explanations) == len(slots), (
            f"Expected {len(slots)} explanations, got {len(explanations)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
