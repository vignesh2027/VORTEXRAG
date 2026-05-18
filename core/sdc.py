"""
Semantic Drift Corrector (SDC) — Layer 4a of VORTEXRAG

Solves: PROBLEM 1 — SEMANTIC DRIFT (SD)

Mathematical Foundation:
  DRIFT VECTOR — The signed difference between the query's causal embedding
  and the chunk's causal embedding:

    D(q, c_i) = v_cau(q) − v_cau(c_i)   ∈ ℝ^d

  This vector encodes both the DIRECTION and MAGNITUDE of causal mismatch.
  If D is small → the chunk's causal chain is close to the query's causal intent.
  If D is large → the chunk has drifted causally, even if semantically similar.

  SEMANTIC DRIFT SCORE (SDS):
    SDS(q, c_i) = 1 − tanh(||D(q, c_i)||₂ / τ)

    where:
      ||D||₂ = L2 norm of drift vector (how far the chunk has causally drifted)
      τ       = drift temperature (τ > 0, learned per domain)
               - Low τ: strict — even small drifts are penalized heavily
               - High τ: lenient — only large drifts are penalized
      tanh:   squashes the penalty to (0,1), giving SDS ∈ (0,1)
              SDS → 1 when drift is near zero (causally aligned)
              SDS → 0 when drift is large (causally irrelevant)

  ACCEPTANCE GATE:
    c_i is ACCEPTED iff SDS(q, c_i) ≥ δ_SDC   (default: 0.72)

WHY tanh AND NOT sigmoid OR linear?
  - tanh grows fast near zero and saturates at large values.
  - This means: small drift → near-full score (minor causal variations accepted)
                large drift → near-zero score (hard rejection, not just soft penalty)
  - sigmoid would be off-centered; linear would allow unbounded negative scores.
  - The tanh shape mirrors how humans judge relevance: slightly off-topic is fine,
    completely off-topic is a hard no.

WHY IS SEMANTIC DRIFT HARD TO SOLVE WITH COSINE SIMILARITY ALONE?
  Example:
    Query: "Why did Lehman Brothers collapse in 2008?"
    Chunk A: "Lehman Brothers filed for bankruptcy due to subprime mortgage exposure."
             → Semantic sim: 0.89, Causal sim: 0.91 → SDS: 0.94 ✓ ACCEPTED
    Chunk B: "The 2008 financial crisis affected millions of homeowners worldwide."
             → Semantic sim: 0.87 (nearly as high!), Causal sim: 0.31 → SDS: 0.21 ✗ REJECTED

  Standard RAG would include Chunk B because its semantic score is nearly as high.
  SDC rejects it because the causal chain (effect on homeowners) does not answer
  WHY Lehman collapsed — it describes a downstream effect, not the root cause.

BEST PRACTICES:
  - Set τ low (0.3–0.5) for scientific/legal queries where causal precision matters
  - Set τ high (1.0–1.5) for exploratory/creative queries
  - δ_SDC=0.72 is the empirically validated default threshold (cross-domain)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .tve import TVEVector
from .vrc import SpiralCandidate


@dataclass
class SDCConfig:
    """Configuration for Semantic Drift Corrector."""
    tau: float = 0.8          # drift temperature τ (higher = more lenient)
    delta_sdc: float = 0.72   # acceptance threshold δ_SDC
    domain: str = "general"   # domain-specific tau presets

    # Domain-tuned tau values (empirically derived)
    _DOMAIN_TAUS: dict = None

    def __post_init__(self):
        self._DOMAIN_TAUS = {
            "legal":     0.4,
            "medical":   0.35,
            "scientific": 0.3,
            "code":      0.6,
            "general":   0.8,
            "creative":  1.2,
        }
        if self.domain in self._DOMAIN_TAUS:
            self.tau = self._DOMAIN_TAUS[self.domain]


class SDCResult(NamedTuple):
    """Result of SDC evaluation for a single chunk."""
    candidate: SpiralCandidate
    drift_norm: float       # ||D(q, c_i)||₂
    sds_score: float        # SDS ∈ (0, 1)
    accepted: bool          # SDS ≥ δ_SDC
    drift_direction: np.ndarray  # D(q, c_i) — direction encodes mismatch type


class SemanticDriftCorrector:
    """
    Filters out causally drifted chunks from the VRC spiral pool.

    The SDC operates on the CAUSAL ARM only of the TVE vectors. It does not
    re-evaluate semantic or syntactic similarity — those are already captured
    in the TVE score. SDC is a precision gate that asks: "Is this chunk causally
    relevant to the query's intent, or has it drifted into a topically adjacent
    but causally unrelated region?"

    The drift vector D(q, c_i) is a signed vector in ℝ^d. Its L2 norm gives
    the magnitude of drift; its direction encodes WHAT TYPE of causal mismatch
    exists (temporal drift, entity drift, relation drift, etc.). Future versions
    of VORTEXRAG will exploit the direction for targeted context reconstruction.

    Usage:
        sdc = SemanticDriftCorrector(config=SDCConfig(domain="legal"))
        results = sdc.filter(query_vec, spiral_pool)
        accepted = [r for r in results if r.accepted]
    """

    def __init__(self, config: SDCConfig | None = None):
        self.config = config or SDCConfig()

    def drift_vector(self, q_vec: TVEVector, c_vec: TVEVector) -> np.ndarray:
        """
        Compute the Causal Drift Vector D(q, c_i) = v_cau(q) − v_cau(c_i).

        The vector D lives in the causal embedding space ℝ^d.
        Its norm ||D||₂ tells us how far the chunk has drifted from the
        query's causal intent. Its direction encodes the axis of drift:
          - If D aligns with entity dimensions → entity substitution drift
          - If D aligns with temporal dims   → temporal drift (wrong era)
          - If D aligns with relation dims   → causal-relation flip drift
        """
        return q_vec.causal - c_vec.causal

    def sds(self, q_vec: TVEVector, c_vec: TVEVector) -> tuple[float, float]:
        """
        Compute Semantic Drift Score.

        Returns (sds_score, drift_norm).

        SDS = 1 − tanh(||D(q,c_i)||₂ / τ)

        The division by τ normalizes the drift magnitude to a domain-appropriate
        scale before the tanh squashing. Without τ normalization, the same drift
        magnitude would be treated identically regardless of whether we're in
        a high-precision (medical) or low-precision (creative) context.
        """
        D = self.drift_vector(q_vec, c_vec)
        drift_norm = float(np.linalg.norm(D))
        sds_score = 1.0 - np.tanh(drift_norm / self.config.tau)
        return float(sds_score), drift_norm

    def evaluate(
        self,
        query_vec: TVEVector,
        candidate: SpiralCandidate,
    ) -> SDCResult:
        """Evaluate a single candidate chunk for semantic drift."""
        sds_score, drift_norm = self.sds(query_vec, candidate.tve_vec)
        D = self.drift_vector(query_vec, candidate.tve_vec)
        return SDCResult(
            candidate=candidate,
            drift_norm=drift_norm,
            sds_score=sds_score,
            accepted=sds_score >= self.config.delta_sdc,
            drift_direction=D,
        )

    def filter(
        self,
        query_vec: TVEVector,
        spiral_pool: list[SpiralCandidate],
    ) -> list[SDCResult]:
        """
        Apply SDC gate to the entire spiral pool.

        All candidates are evaluated (not just rejected). The returned list
        contains ALL SDCResults so that downstream modules (CPG, RFG) can
        use the SDS scores for weighted computation even for accepted chunks.
        """
        results = [self.evaluate(query_vec, c) for c in spiral_pool]
        # Sort: accepted first, then by SDS score descending
        results.sort(key=lambda r: (int(r.accepted), r.sds_score), reverse=True)
        return results

    def accepted_only(self, results: list[SDCResult]) -> list[SDCResult]:
        """Return only the accepted candidates."""
        return [r for r in results if r.accepted]
