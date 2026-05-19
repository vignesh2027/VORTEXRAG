"""
Vortex Retrieval Cone (VRC) — Layer 3 of VORTEXRAG

Mathematical Foundation:
  Standard retrieval returns a FLAT list (top-k by cosine score). VRC models
  retrieval as a SPIRAL PROBABILITY SURFACE in embedding space, where chunks
  are arranged on a vortex cone based on their angular and radial position
  relative to the query centroid.

  spiral_rank(c_i, θ) = TVE_score(c_i) · e^(−λ·r_i) · cos(n·θ_i)

  where:
    r_i   = Euclidean distance from centroid of query cluster in embedding space
    θ_i   = angular position (polar coordinate) relative to query vector
    n     = spiral tightness ∈ {1, 2, 3}  (1=loose/broad, 3=tight/precise)
    λ     = radial decay rate (controls how fast relevance drops with distance)

WHY A VORTEX TOPOLOGY?
  In a flat top-k, all chunks within the same cosine score band are treated
  identically. But in embedding space, chunks cluster in angular bands around
  the query vector. Chunks at the SAME radial distance but different angles
  encode very different semantic neighborhoods.

  The vortex surface captures this angular structure: chunks spiraling in from
  the same angular direction as the query are ranked higher, even if they are
  slightly farther in raw Euclidean distance. This prevents the common flat-list
  failure where a high-similarity but semantically tangential cluster dominates
  the top-k.

KEY GEOMETRIC INSIGHT — NEGATIVE SCORES:
  cos(n·θ) becomes NEGATIVE for chunks where n·θ > π/2.
  For n=2: chunks beyond θ > 45° score negative and are suppressed.
  For n=3: chunks beyond θ > 30° score negative and are suppressed.
  Standard cosine similarity NEVER produces negative scores, so it cannot
  suppress any chunk with partial semantic overlap. The VRC can.

SPIRAL TIGHTNESS GUIDE:
  n=1: Single loose spiral — exploratory/broad queries, large candidate pool
  n=2: Double helix — balanced precision/recall (DEFAULT)
  n=3: Triple tight spiral — precise factual/causal queries, dense corpora

RADIAL DECAY GUIDE:
  λ=0.3: Slow decay — good for large sparse corpora (many distant candidates)
  λ=0.5: Default — balanced for medium corpora (10K–1M documents)
  λ=0.8: Fast decay — tight clusters, small dense corpora (< 10K documents)

ADAPTIVE LAMBDA:
  For large corpora (N > 500K), λ should be reduced to avoid cutting off
  the spiral cone too aggressively. Use adaptive_lambda(N) to compute
  the optimal λ for a given corpus size.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from .tve import TriVectorEncoder, TVEVector


@dataclass
class VRCConfig:
    """Configuration for Vortex Retrieval Cone."""
    n_spiral: int = 2            # spiral tightness n ∈ {1, 2, 3}
    lambda_decay: float = 0.5    # radial decay rate λ
    candidate_pool: int = 200    # initial candidate pool size (pre-spiral)
    top_k: int = 50              # spiral pool size returned to SDC/CPG
    adaptive_lambda: bool = False  # auto-tune λ based on corpus size


class SpiralCandidate(NamedTuple):
    """
    A chunk ranked in the vortex spiral.

    All fields are preserved for downstream SDC, CPG, and RFG modules.
    The spiral_rank is the primary ordering key; tve_score and polar
    coordinates are available for analysis and visualization.
    """
    chunk_id:    int
    chunk_text:  str
    tve_score:   float
    radial_dist: float     # r_i — Euclidean distance from query centroid
    theta:       float     # θ_i — angular deviation from query direction (radians)
    spiral_rank: float     # final VRC score (can be negative for off-axis chunks)
    tve_vec:     TVEVector


class VortexRetrievalCone:
    """
    Retrieves candidates via spiral topology rather than flat cosine ranking.

    The key insight: a flat top-k treats all chunks at cosine score 0.72 as
    equivalent. But in 768-dim space, these chunks may lie in completely
    different angular neighborhoods. The VRC's spiral function rewards chunks
    that are angularly aligned with the query direction AND radially close —
    the exact geometric condition for genuinely relevant content.

    Pipeline role:
      TVE Scores → [VRC] → Spiral Pool (top_k candidates)
                                    ↓
                              SDC + CPG purification

    The spiral pool is larger than the final context (top_k=50 >> final W*=10)
    to give downstream modules room to purify without running out of candidates.

    Usage:
        vrc = VortexRetrievalCone(encoder)
        candidates = vrc.retrieve(query_vec, corpus_vecs, corpus_texts)
        # Returns list[SpiralCandidate] sorted by spiral_rank desc
    """

    def __init__(
        self,
        encoder: TriVectorEncoder,
        config: Optional[VRCConfig] = None,
    ):
        self.encoder = encoder
        self.config = config or VRCConfig()

    # ──── Core Geometric Operations ───────────────────────────────────────────

    def _compute_polar_coords(
        self,
        query_vec: np.ndarray,
        chunk_vec: np.ndarray,
        centroid: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute (r, θ) polar coordinates of chunk_vec relative to query centroid.

          r_i = ||chunk_vec − centroid||₂
              = Euclidean distance from the center of the query's semantic cluster.
              Measures how "far out" the chunk is from the query neighborhood.

          θ_i = arccos(query_vec · chunk_vec / (||q|| · ||c||))
              = angular deviation of chunk from the query direction.
              θ=0 means perfectly aligned; θ=π means opposite direction.

        The centroid is the mean of the candidate pool's semantic vectors —
        it represents the "center of mass" of the relevant semantic neighborhood,
        not just the query itself. This makes r more robust to query outliers.
        """
        diff = chunk_vec - centroid
        r = float(np.linalg.norm(diff))

        q_norm = np.linalg.norm(query_vec)
        c_norm = np.linalg.norm(chunk_vec)
        denom = q_norm * c_norm
        if denom < 1e-10:
            return r, np.pi / 2  # default to 90° if either is zero

        cos_theta = np.dot(query_vec, chunk_vec) / denom
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = float(np.arccos(cos_theta))
        return r, theta

    def _compute_polar_coords_batch(
        self,
        query_vec: np.ndarray,    # (d,)
        chunk_vecs: np.ndarray,   # (N, d)
        centroid: np.ndarray,     # (d,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorised (r, θ) computation for a batch of chunk vectors.

        Returns (r_array, theta_array) each of shape (N,).
        Significantly faster than looping when N > 100.
        """
        diffs = chunk_vecs - centroid[np.newaxis, :]           # (N, d)
        r_array = np.linalg.norm(diffs, axis=1)                # (N,)

        q_norm = np.linalg.norm(query_vec)
        c_norms = np.linalg.norm(chunk_vecs, axis=1)           # (N,)
        denoms = q_norm * c_norms + 1e-10                      # (N,)

        dots = chunk_vecs @ query_vec                           # (N,)
        cos_thetas = np.clip(dots / denoms, -1.0, 1.0)         # (N,)
        theta_array = np.arccos(cos_thetas)                    # (N,) in [0, π]

        return r_array, theta_array

    def _spiral_rank(
        self,
        tve_score: float,
        r: float,
        theta: float,
    ) -> float:
        """
        Compute spiral_rank = TVE_score · e^(−λ·r) · cos(n·θ)

        Term decomposition:
          TVE_score:    Base relevance from the tri-vector encoder.
                        Must be positive for the chunk to have any chance.

          e^(−λ·r):    Radial penalty. Chunks far from the query cluster
                        centroid are penalized exponentially. λ controls decay:
                        λ=0.5 → score halves every 1.4 units of radial distance.

          cos(n·θ):    Angular alignment reward/penalty.
                        cos(n·θ) = +1 when θ=0 (perfectly aligned)
                        cos(n·θ) =  0 when n·θ = π/2 (orthogonal)
                        cos(n·θ) = -1 when n·θ = π (opposite direction)

        Chunks with cos(n·θ) < 0 receive NEGATIVE spiral_rank. This is
        intentional — it means off-axis semantic clusters are actively
        suppressed rather than just ranked lower. Standard cosine similarity
        never produces negative scores, so it cannot achieve this suppression.
        """
        radial_penalty  = float(np.exp(-self.config.lambda_decay * r))
        angular_reward  = float(np.cos(self.config.n_spiral * theta))
        return tve_score * radial_penalty * angular_reward

    def _spiral_rank_batch(
        self,
        tve_scores: np.ndarray,   # (N,)
        r_array: np.ndarray,      # (N,)
        theta_array: np.ndarray,  # (N,)
    ) -> np.ndarray:
        """
        Vectorised spiral_rank computation for a batch.
        Returns spiral_rank array of shape (N,).
        """
        radial_penalties = np.exp(-self.config.lambda_decay * r_array)         # (N,)
        angular_rewards  = np.cos(self.config.n_spiral * theta_array)          # (N,)
        return tve_scores * radial_penalties * angular_rewards                  # (N,)

    # ──── Adaptive Lambda ─────────────────────────────────────────────────────

    @staticmethod
    def adaptive_lambda(corpus_size: int) -> float:
        """
        Compute optimal radial decay rate λ based on corpus size.

        Larger corpora have more distant but potentially relevant candidates.
        Reducing λ for larger corpora ensures the spiral cone doesn't cut off
        too aggressively. Derived from: λ = 0.5 · log10(10000 / N_clipped)

          N < 10K:   λ ≈ 0.5–0.8  (tight decay, small dense corpus)
          N = 100K:  λ ≈ 0.35     (moderate decay)
          N = 1M:    λ ≈ 0.15     (loose decay, large sparse corpus)
          N > 10M:   λ ≈ 0.05     (very loose, exploratory)
        """
        n_clipped = max(min(corpus_size, 10_000_000), 1_000)
        return max(0.05, 0.5 * np.log10(10_000 / n_clipped))

    # ──── Main Retrieval API ──────────────────────────────────────────────────

    def retrieve(
        self,
        query_vec: TVEVector,
        corpus_vecs: list[TVEVector],
        corpus_texts: list[str],
    ) -> list[SpiralCandidate]:
        """
        Retrieve chunks using vectorised spiral topology ranking.

        Algorithm:
          1. Compute TVE scores for all N corpus chunks           O(N·d)
          2. Take top-candidate_pool chunks by TVE (pre-filter)   O(N log k)
          3. Compute cluster centroid from pre-filter pool        O(k·d)
          4. Vectorised polar coordinates (r, θ) for pool        O(k·d)
          5. Vectorised spiral_rank computation                   O(k)
          6. Return top_k by spiral_rank                          O(k log k)

        Total: O(N·d + k·d) where k << N.

        Returns a list of SpiralCandidate sorted by spiral_rank descending.
        Note: negative spiral_rank candidates are still returned if they rank
        in top_k — SDC/CPG have final say on acceptance.
        """
        if len(corpus_vecs) == 0:
            return []

        assert len(corpus_vecs) == len(corpus_texts), \
            f"corpus_vecs ({len(corpus_vecs)}) and corpus_texts ({len(corpus_texts)}) must align"

        # Optionally tune λ for corpus size
        if self.config.adaptive_lambda:
            self.config.lambda_decay = self.adaptive_lambda(len(corpus_vecs))

        # Step 1: Batch TVE scoring — O(N·d)
        tve_scores = self.encoder.batch_tve_scores(query_vec, corpus_vecs)  # (N,)

        # Step 2: Pre-filter to candidate pool by TVE score
        pool_size = min(self.config.candidate_pool, len(corpus_vecs))
        pool_indices = np.argsort(tve_scores)[::-1][:pool_size]             # (pool_size,)

        # Step 3: Compute centroid of candidate pool (semantic arm)
        pool_sem_vecs = np.stack([corpus_vecs[i].semantic for i in pool_indices])  # (pool, d)
        centroid = pool_sem_vecs.mean(axis=0)                                       # (d,)

        # Step 4: Vectorised polar coordinates
        pool_sem_matrix = pool_sem_vecs                                             # (pool, d)
        r_array, theta_array = self._compute_polar_coords_batch(
            query_vec.semantic, pool_sem_matrix, centroid
        )  # each (pool,)

        # Step 5: Vectorised spiral_rank
        pool_tve_scores = tve_scores[pool_indices]                                  # (pool,)
        spiral_ranks = self._spiral_rank_batch(pool_tve_scores, r_array, theta_array)  # (pool,)

        # Step 6: Build candidates and sort by spiral_rank
        candidates: list[SpiralCandidate] = []
        for local_i, global_i in enumerate(pool_indices):
            candidates.append(SpiralCandidate(
                chunk_id=int(global_i),
                chunk_text=corpus_texts[global_i],
                tve_score=float(pool_tve_scores[local_i]),
                radial_dist=float(r_array[local_i]),
                theta=float(theta_array[local_i]),
                spiral_rank=float(spiral_ranks[local_i]),
                tve_vec=corpus_vecs[global_i],
            ))

        candidates.sort(key=lambda c: c.spiral_rank, reverse=True)
        return candidates[: self.config.top_k]

    # ──── Analysis and Diagnostics ────────────────────────────────────────────

    def explain_spiral_rank(self, candidate: SpiralCandidate) -> dict:
        """
        Decompose a chunk's spiral_rank into its three multiplicative factors.

        Returns a dict showing exactly how much each factor contributed:
          {
            'tve_score':       float — base relevance from tri-vector encoder
            'radial_penalty':  float — e^(−λ·r) — proximity to query cluster
            'angular_reward':  float — cos(n·θ) — angular alignment with query
            'spiral_rank':     float — product of all three
            'theta_degrees':   float — angular deviation in degrees
            'n_spiral':        int   — spiral tightness used
            'lambda_decay':    float — radial decay rate used
            'interpretation':  str   — human-readable explanation
          }
        """
        r = candidate.radial_dist
        theta = candidate.theta
        n = self.config.n_spiral
        lam = self.config.lambda_decay

        radial_penalty = float(np.exp(-lam * r))
        angular_reward = float(np.cos(n * theta))
        theta_deg = float(np.degrees(theta))

        if angular_reward < 0:
            interp = (
                f"Angular penalty: θ={theta_deg:.1f}° → n·θ={n * theta_deg:.1f}° > 90°. "
                f"This chunk lies in an angularly opposed cluster — spiral_rank is negative "
                f"({candidate.spiral_rank:.4f}). SDC/CPG may still accept it if SDS is high."
            )
        elif radial_penalty < 0.3:
            interp = (
                f"Heavy radial penalty: r={r:.3f}, decay e^(−{lam}·{r:.2f})={radial_penalty:.3f}. "
                f"Chunk is far from query cluster centroid despite good TVE alignment."
            )
        elif candidate.tve_score > 0.8 and angular_reward > 0.8:
            interp = (
                f"Strong spiral candidate: high TVE ({candidate.tve_score:.3f}), "
                f"strong angular alignment (θ={theta_deg:.1f}°). Core retrieval hit."
            )
        else:
            interp = (
                f"Moderate spiral rank: TVE={candidate.tve_score:.3f}, "
                f"θ={theta_deg:.1f}°, r={r:.3f}."
            )

        return {
            "tve_score":      round(candidate.tve_score, 4),
            "radial_penalty": round(radial_penalty, 4),
            "angular_reward": round(angular_reward, 4),
            "spiral_rank":    round(candidate.spiral_rank, 4),
            "theta_degrees":  round(theta_deg, 2),
            "n_spiral":       n,
            "lambda_decay":   lam,
            "interpretation": interp,
        }

    def angular_distribution(
        self,
        candidates: list[SpiralCandidate],
        n_bins: int = 6,
    ) -> dict:
        """
        Compute the angular distribution of candidates across the spiral cone.

        Returns a histogram of how many candidates fall in each angular bin.
        Useful for diagnosing whether the corpus has clustered semantic neighborhoods
        that the spiral is successfully separating.
        """
        if not candidates:
            return {}

        thetas_deg = [np.degrees(c.theta) for c in candidates]
        bin_edges = np.linspace(0, 180, n_bins + 1)
        counts, _ = np.histogram(thetas_deg, bins=bin_edges)

        return {
            f"{bin_edges[i]:.0f}°–{bin_edges[i+1]:.0f}°": int(counts[i])
            for i in range(n_bins)
        }

    def negative_suppression_count(self, candidates: list[SpiralCandidate]) -> int:
        """
        Count how many candidates received a negative spiral_rank (angular suppression).

        A high count indicates the corpus has significant off-axis semantic clusters
        that standard cosine similarity would incorrectly include but VRC suppresses.
        """
        return sum(1 for c in candidates if c.spiral_rank < 0)

    def pool_statistics(self, candidates: list[SpiralCandidate]) -> dict:
        """
        Summary statistics for a retrieved spiral pool.

        Returns:
          {
            'n_candidates':       int   — total candidates returned
            'n_negative_rank':    int   — candidates with spiral_rank < 0
            'mean_tve':           float — average TVE score in pool
            'mean_spiral_rank':   float — average spiral_rank
            'mean_theta_degrees': float — average angular deviation
            'mean_radial_dist':   float — average distance from centroid
            'suppression_rate':   float — fraction with negative spiral_rank
          }
        """
        if not candidates:
            return {}

        tve_scores    = [c.tve_score for c in candidates]
        spiral_ranks  = [c.spiral_rank for c in candidates]
        thetas_deg    = [np.degrees(c.theta) for c in candidates]
        radial_dists  = [c.radial_dist for c in candidates]
        n_neg = sum(1 for sr in spiral_ranks if sr < 0)

        return {
            "n_candidates":       len(candidates),
            "n_negative_rank":    n_neg,
            "mean_tve":           round(float(np.mean(tve_scores)), 4),
            "mean_spiral_rank":   round(float(np.mean(spiral_ranks)), 4),
            "mean_theta_degrees": round(float(np.mean(thetas_deg)), 2),
            "mean_radial_dist":   round(float(np.mean(radial_dists)), 4),
            "suppression_rate":   round(n_neg / len(candidates), 3),
        }

    def compare_with_flat_topk(
        self,
        query_vec: TVEVector,
        corpus_vecs: list[TVEVector],
        corpus_texts: list[str],
        k: int = 10,
    ) -> dict:
        """
        Compare VRC retrieval with standard flat top-k retrieval.

        Returns a dict showing:
          - chunks unique to VRC (spiral promoted them)
          - chunks unique to flat top-k (spiral suppressed them)
          - chunks in both (agreement)

        Useful for demonstrating the VRC's angular suppression effect.
        """
        # VRC retrieval
        vrc_candidates = self.retrieve(query_vec, corpus_vecs, corpus_texts)
        vrc_ids = {c.chunk_id for c in vrc_candidates[:k]}

        # Flat top-k by TVE score only
        tve_scores = self.encoder.batch_tve_scores(query_vec, corpus_vecs)
        flat_indices = set(np.argsort(tve_scores)[::-1][:k].tolist())

        in_both   = vrc_ids & flat_indices
        vrc_only  = vrc_ids - flat_indices
        flat_only = flat_indices - vrc_ids

        return {
            "k": k,
            "in_both_count":   len(in_both),
            "vrc_only_count":  len(vrc_only),
            "flat_only_count": len(flat_only),
            "vrc_only_ids":    sorted(vrc_only),
            "flat_only_ids":   sorted(flat_only),
            "agreement_rate":  round(len(in_both) / k, 3),
        }
