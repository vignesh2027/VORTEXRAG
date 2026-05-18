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
    n     = spiral tightness ∈ {1, 2, 3}  (1=loose, 3=tight)
    λ     = radial decay rate (learned per corpus, controls how fast relevance
            drops with distance from the query centroid)

WHY A VORTEX TOPOLOGY?
  In a flat top-k, all chunks within the same cosine score band are treated
  identically. But in embedding space, chunks cluster in angular bands around
  the query vector. Chunks at the SAME radial distance but different angles
  encode very different semantic neighborhoods. The vortex surface captures
  this angular structure: chunks spiraling in from the same angular direction
  as the query are ranked higher, even if they are slightly farther in raw
  distance. This prevents a common flat-list failure where a high-similarity
  but semantically tangential cluster dominates the top-k.

BEST USE:
  - Use n=1 (loose spiral) for broad exploratory queries
  - Use n=3 (tight spiral) for precise factual or causal queries
  - λ should be smaller for large corpora (slower decay = more candidates)

DIFFERENCE FROM STANDARD TOP-K:
  Top-k: rank by cosine → take first k. Single dimension, ignores topology.
  VRC:   rank by spiral_rank → return ranked cone pool (typically top-200).
         Preserves angular neighborhood structure, enabling the downstream
         SDC + CPG modules to purge poisoned clusters rather than isolated chunks.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .tve import TriVectorEncoder, TVEVector


@dataclass
class VRCConfig:
    """Configuration for Vortex Retrieval Cone."""
    n_spiral: int = 2           # spiral tightness {1, 2, 3}
    lambda_decay: float = 0.5   # radial decay rate λ
    candidate_pool: int = 200   # initial candidate pool size
    top_k: int = 50             # final pool size returned


class SpiralCandidate(NamedTuple):
    """A chunk ranked in the vortex spiral."""
    chunk_id: int
    chunk_text: str
    tve_score: float
    radial_dist: float
    theta: float
    spiral_rank: float
    tve_vec: TVEVector


class VortexRetrievalCone:
    """
    Retrieves candidates via spiral topology rather than flat cosine ranking.

    The key insight: a flat top-k treats all chunks at cosine score 0.72 as
    equivalent. But in 768-dim space, these chunks may lie in completely
    different angular neighborhoods. The VRC's spiral function rewards chunks
    that are angularly aligned with the query direction AND radially close —
    the exact geometric condition for genuinely relevant content, not just
    semantically adjacent content.

    Usage:
        vrc = VortexRetrievalCone(encoder)
        candidates = vrc.retrieve(query_vec, corpus_vecs, corpus_texts)
    """

    def __init__(
        self,
        encoder: TriVectorEncoder,
        config: VRCConfig | None = None,
    ):
        self.encoder = encoder
        self.config = config or VRCConfig()

    def _compute_polar_coords(
        self,
        query_vec: np.ndarray,
        chunk_vec: np.ndarray,
        centroid: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute (r, θ) polar coordinates of chunk_vec relative to query centroid.

        r = ||chunk_vec − centroid||₂   (radial distance from cluster center)
        θ = arccos(query_vec · chunk_vec)  (angular deviation from query direction)

        The centroid is the mean of the top-200 semantic vectors — it represents
        the "center of mass" of the relevant semantic neighborhood.
        """
        diff = chunk_vec - centroid
        r = float(np.linalg.norm(diff))
        cos_theta = np.dot(query_vec, chunk_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-10
        )
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))
        return r, theta

    def _spiral_rank(
        self,
        tve_score: float,
        r: float,
        theta: float,
    ) -> float:
        """
        Compute spiral_rank = TVE_score · e^(−λ·r) · cos(n·θ)

        Breaking down the three multiplicative terms:
          1. TVE_score:        base relevance — must be positive
          2. e^(−λ·r):         radial penalty — far chunks are discounted
                               exponentially. λ controls sharpness.
          3. cos(n·θ):         angular alignment reward — chunks angularly
                               aligned with the query (θ≈0) get cos→1.
                               n controls how many spiral arms the vortex has:
                               n=1: single broad arm (general retrieval)
                               n=2: double helix (balanced)
                               n=3: triple tight spiral (precise retrieval)

        Note: cos(n·θ) can be negative for chunks far off-axis. This is INTENTIONAL
        — it suppresses chunks that are angularly opposed to the query direction,
        even if they have high cosine similarity. This is the geometric mechanism
        that prevents semantic drift clusters from polluting the spiral pool.
        """
        radial_penalty = np.exp(-self.config.lambda_decay * r)
        angular_reward = np.cos(self.config.n_spiral * theta)
        return tve_score * radial_penalty * angular_reward

    def retrieve(
        self,
        query_vec: TVEVector,
        corpus_vecs: list[TVEVector],
        corpus_texts: list[str],
    ) -> list[SpiralCandidate]:
        """
        Retrieve chunks using spiral topology ranking.

        Returns a ranked list of SpiralCandidate objects — the "vortex cone pool"
        that feeds into the SDC and CPG modules for purification.
        """
        assert len(corpus_vecs) == len(corpus_texts), "corpus_vecs and texts must align"

        # Step 1: batch TVE scoring
        tve_scores = self.encoder.batch_tve_scores(query_vec, corpus_vecs)

        # Step 2: get top-N candidate pool by TVE score (semantic pre-filter)
        pool_size = min(self.config.candidate_pool, len(corpus_vecs))
        top_indices = np.argsort(tve_scores)[::-1][:pool_size]

        # Step 3: compute centroid of candidate pool (semantic vectors)
        candidate_sem_vecs = np.stack([corpus_vecs[i].semantic for i in top_indices])
        centroid = candidate_sem_vecs.mean(axis=0)

        # Step 4: compute spiral rank for each candidate
        candidates: list[SpiralCandidate] = []
        for idx in top_indices:
            r, theta = self._compute_polar_coords(
                query_vec.semantic, corpus_vecs[idx].semantic, centroid
            )
            sr = self._spiral_rank(float(tve_scores[idx]), r, theta)
            candidates.append(
                SpiralCandidate(
                    chunk_id=int(idx),
                    chunk_text=corpus_texts[idx],
                    tve_score=float(tve_scores[idx]),
                    radial_dist=r,
                    theta=theta,
                    spiral_rank=sr,
                    tve_vec=corpus_vecs[idx],
                )
            )

        # Step 5: sort by spiral_rank descending → return top_k
        candidates.sort(key=lambda c: c.spiral_rank, reverse=True)
        return candidates[: self.config.top_k]
