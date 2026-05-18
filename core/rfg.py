"""
Rank Fusion Gate (RFG) — Layer 5a of VORTEXRAG

Mathematical Foundation:
  After SDC and CPG purification, surviving chunks are re-scored using the
  Φ-score (phi-score), which fuses all three quality signals into a single
  multiplicative ranking function.

  Φ-SCORE (phi-score):
    Φ(c_i, q) = TVE_score(q, c_i)^α  ×  SDS(q, c_i)^β  ×  ESR_contribution(c_i, W)^γ

    where:
      TVE_score(q, c_i)           — tri-vector relevance (semantic+syntactic+causal)
      SDS(q, c_i)                 — causal drift correction score
      ESR_contribution(c_i, W)    — chunk's contribution to the window's signal ratio
      α, β, γ ∈ (0,1), α+β+γ=1   — fusion weights (default: 0.4, 0.35, 0.25)

  NORMALIZED PHI:
    Φ̃(c_i) = Φ(c_i) / Σⱼ Φ(c_j)

    Normalization converts Φ into a proper probability distribution over chunks,
    enabling sampling-based and threshold-based selection of final context.

  FINAL CONTEXT W*:
    W* = top-m by Φ̃, subject to ESR(W*, q) ≥ θ_CPG

WHY MULTIPLICATIVE, NOT ADDITIVE FUSION?
  Additive fusion (e.g., 0.4·TVE + 0.35·SDS + 0.25·ESR) allows a chunk with
  very high TVE score to compensate for terrible SDS score. But a chunk with
  high semantic similarity and near-zero causal relevance is WORSE than a chunk
  with moderate scores on all three dimensions — it's a precision killer.

  Multiplicative fusion enforces a "no weak link" policy:
    Φ = 0.9^0.4 × 0.1^0.35 × 0.8^0.25 = 0.961 × 0.427 × 0.946 ≈ 0.388
  vs additive:
    0.4×0.9 + 0.35×0.1 + 0.25×0.8 = 0.36 + 0.035 + 0.2 = 0.595

  Additive would rank the poisoned chunk at 0.595 (high). Multiplicative
  correctly penalizes it to 0.388 because of the terrible SDS=0.1.

ESR CONTRIBUTION:
  ESR_contribution(c_i, W) = SDS(c_i) · w_i / (Σⱼ SDS(c_j) · w_j)

  This measures how much chunk c_i contributes to the window's positive signal.
  Chunks with high SDS AND high attention weight are the most valuable —
  they're both causally relevant AND the LLM will attend to them strongly.

DIFFERENCE FROM STANDARD RRF (Reciprocal Rank Fusion):
  RRF: Σ 1/(k + rank_i) across retrieval systems — purely rank-based, no quality.
  RFG: Φ̃ score — quality-weighted, multiplicative, domain-tunable via α,β,γ.
  RRF ignores whether a chunk is causally relevant; RFG makes causal quality
  the equal partner of retrieval rank.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .tve import TVEVector
from .sdc import SDCResult
from .cpg import CPGEvaluation


@dataclass
class RFGConfig:
    """Configuration for Rank Fusion Gate."""
    alpha: float = 0.40     # TVE weight α
    beta: float = 0.35      # SDS weight β
    gamma: float = 0.25     # ESR contribution weight γ
    top_m: int = 10         # final context size
    epsilon: float = 1e-10  # avoid zero in power computation

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma
        assert abs(total - 1.0) < 1e-6, f"α+β+γ must equal 1.0, got {total}"


class RankedChunk(NamedTuple):
    """A chunk with its final Φ-score."""
    chunk_id: int
    chunk_text: str
    phi_score: float        # Φ(c_i, q) — raw
    phi_norm: float         # Φ̃(c_i) — normalized
    tve_score: float
    sds_score: float
    esr_contribution: float
    sdc_result: SDCResult


class RankFusionGate:
    """
    Computes Φ-scores and selects the optimal final context W*.

    The Φ-score is the VORTEXRAG master ranking signal. It replaces all
    intermediate scores (TVE, SDS, ESR) with a single number that encodes
    the full quality profile of a chunk relative to the query.

    Design principle: each score factor must clear a minimum threshold before
    the multiplicative structure allows a high final score. A chunk with
    TVE=0.95 but SDS=0.05 will have Φ ≈ 0.05^0.35 ≈ 0.19 — the high TVE
    score cannot rescue it from causal irrelevance.

    Usage:
        rfg = RankFusionGate()
        ranked = rfg.rank(query_vec, cpg_result)
        final_context = rfg.select_top_m(ranked)
    """

    def __init__(self, config: RFGConfig | None = None):
        self.config = config or RFGConfig()

    def _esr_contributions(
        self,
        sds_scores: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute each chunk's fractional contribution to the window's signal.

        ESR_contribution(c_i) = SDS_i · w_i / Σⱼ (SDS_j · w_j)

        This normalizes the per-chunk signal contribution so that all
        contributions sum to 1, making it comparable across window sizes.
        """
        signal_i = sds_scores * weights
        total_signal = signal_i.sum() + self.config.epsilon
        return signal_i / total_signal

    def _phi_score(
        self,
        tve: float,
        sds: float,
        esr_contrib: float,
    ) -> float:
        """
        Compute Φ(c_i, q) = TVE^α × SDS^β × ESR_contrib^γ

        All inputs are clipped to (ε, 1) before the power computation to
        prevent log(0) and ensure numerical stability. The ε clip means
        a zero-score chunk gets Φ ≈ ε^α ≈ near-zero, not exactly zero,
        preserving rank ordering even for terrible chunks.
        """
        ε = self.config.epsilon
        α, β, γ = self.config.alpha, self.config.beta, self.config.gamma
        tve_c = np.clip(tve, ε, 1.0)
        sds_c = np.clip(sds, ε, 1.0)
        ec_c  = np.clip(esr_contrib, ε, 1.0)
        return float(tve_c ** α * sds_c ** β * ec_c ** γ)

    def rank(
        self,
        query_vec: TVEVector,
        cpg_eval: CPGEvaluation,
    ) -> list[RankedChunk]:
        """
        Compute Φ̃ scores for all chunks in the CPG-clean window.

        Returns a list of RankedChunk sorted by phi_norm descending.
        """
        window = cpg_eval.window
        weights = cpg_eval.softmax_weights

        if len(window) == 0:
            return []

        sds_scores = np.array([r.sds_score for r in window])
        tve_scores = np.array([r.candidate.tve_score for r in window])
        esr_contribs = self._esr_contributions(sds_scores, weights)

        # Compute raw Φ for each chunk
        raw_phis = np.array([
            self._phi_score(tve_scores[i], sds_scores[i], esr_contribs[i])
            for i in range(len(window))
        ])

        # Normalize to get Φ̃
        phi_sum = raw_phis.sum() + self.config.epsilon
        phi_norm = raw_phis / phi_sum

        ranked = [
            RankedChunk(
                chunk_id=window[i].candidate.chunk_id,
                chunk_text=window[i].candidate.chunk_text,
                phi_score=float(raw_phis[i]),
                phi_norm=float(phi_norm[i]),
                tve_score=float(tve_scores[i]),
                sds_score=float(sds_scores[i]),
                esr_contribution=float(esr_contribs[i]),
                sdc_result=window[i],
            )
            for i in range(len(window))
        ]

        ranked.sort(key=lambda c: c.phi_norm, reverse=True)
        return ranked

    def select_top_m(self, ranked: list[RankedChunk]) -> list[RankedChunk]:
        """Select the top-m chunks by Φ̃ for final context W*."""
        return ranked[: self.config.top_m]
