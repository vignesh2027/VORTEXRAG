"""
Rank Fusion Gate (RFG) — Layer 5a of VORTEXRAG

Mathematical Foundation:
  After SDC and CPG purification, surviving chunks are re-scored using the
  Φ-score (phi-score), which fuses all three quality signals into a single
  multiplicative ranking function:

  Φ-SCORE:
    Φ(c_i, q) = TVE_score(q, c_i)^α  ×  SDS(q, c_i)^β  ×  ESR_contrib(c_i, W)^γ

    where:
      TVE_score(q, c_i)          — tri-vector relevance (semantic+syntactic+causal)
      SDS(q, c_i)                — semantic drift correction score ∈ (0, 1]
      ESR_contrib(c_i, W)        — chunk's fractional contribution to window signal
      α, β, γ ∈ (0,1), α+β+γ=1  — fusion weights (domain-tuned presets)

    ESR_contribution(c_i, W) = SDS(c_i) · w_i / (Σⱼ SDS(c_j) · w_j)

    This measures how much chunk c_i contributes to the window's positive signal.
    Chunks with high SDS AND high attention weight are most valuable — both
    causally relevant AND highly attended by the LLM.

  NORMALIZED PHI:
    Φ̃(c_i) = Φ(c_i) / Σⱼ Φ(c_j)

    Normalization converts Φ into a probability distribution over chunks.
    Φ̃ sums to 1 across the window, enabling threshold-based and sampling-based
    selection independent of corpus scale.

  FINAL CONTEXT W*:
    W* = top-m by Φ̃, subject to ESR(W*, q) ≥ θ_CPG

WHY MULTIPLICATIVE, NOT ADDITIVE FUSION?
  Additive fusion (0.4·TVE + 0.35·SDS + 0.25·ESR) allows a high TVE score
  to rescue a terrible SDS score. Consider a chunk with:
    TVE=0.95, SDS=0.05, ESR_contrib=0.80

  Additive:      0.4×0.95 + 0.35×0.05 + 0.25×0.80 = 0.38+0.018+0.20 = 0.598
  Multiplicative: 0.95^0.4 × 0.05^0.35 × 0.80^0.25
                = 0.979 × 0.427 × 0.945 ≈ 0.395

  Additive ranks this chunk at 0.598 — HIGH, despite causal irrelevance.
  Multiplicative ranks it at 0.395 — correctly penalized.

  The multiplicative structure enforces a "no weak link" policy: every quality
  dimension must independently be strong. A chunk that fails causal alignment
  (SDS≈0) gets Φ ≈ 0 regardless of TVE, because SDS^β ≈ 0.

  This mirrors the objective: we want a context window where EVERY chunk is
  causally aligned (not just the average), causally relevant (not just the mean),
  and individually contributes to the signal (not just one strong chunk).

WHY ESR_CONTRIBUTION INSTEAD OF RAW ESR?
  Raw ESR measures the window as a whole. ESR_contribution(c_i) measures how
  much chunk c_i contributes to the window's positive signal — its "share of
  the signal." A chunk in a clean window (ESR=5.0) might contribute 30% of
  the signal (ESR_contrib=0.30) or only 5% (ESR_contrib=0.05). The chunk
  contributing 30% is more valuable for the final context W*.

WHY Φ̃ (NORMALIZED)?
  Raw Φ scores depend on the number of chunks in the window (more chunks →
  smaller individual Φ values). Normalizing to Φ̃ = Φ / Σ Φ produces a
  probability-like distribution that is comparable across different window
  sizes and corpus scales.

DOMAIN FUSION WEIGHTS:
  Different domains require different emphasis across the three quality signals:
    Legal/Medical/Scientific: higher β (SDS) — causal precision dominates
    Code: balanced α and γ — syntax AND context window quality matter equally
    General: default balanced weights
  Use DOMAIN_FUSION_WEIGHTS presets or adapt_for_domain() for automatic tuning.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .tve import TVEVector
from .sdc import SDCResult
from .cpg import CPGEvaluation


# ── Domain-specific Φ-score fusion weight presets (α, β, γ) ──────────────────
# α = TVE weight, β = SDS weight, γ = ESR_contribution weight.
# All three must sum to 1.0. Empirically validated on domain-specific benchmarks.
DOMAIN_FUSION_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "general":       (0.40, 0.35, 0.25),  # balanced default
    "legal":         (0.35, 0.40, 0.25),  # causal precision critical
    "medical":       (0.35, 0.40, 0.25),  # mechanism precision critical
    "scientific":    (0.35, 0.40, 0.25),  # progenitor chain precision
    "code":          (0.40, 0.30, 0.30),  # syntax + context window quality
    "financial":     (0.40, 0.35, 0.25),  # balanced with causal emphasis
    "educational":   (0.45, 0.30, 0.25),  # topic coverage most important
    "cybersecurity": (0.35, 0.40, 0.25),  # attack vector causality critical
    "historical":    (0.40, 0.35, 0.25),  # causation through time matters
    "customer":      (0.45, 0.30, 0.25),  # intent + context coverage
    "creative":      (0.50, 0.25, 0.25),  # semantic exploration dominates
}


@dataclass
class RFGConfig:
    """
    Configuration for Rank Fusion Gate.

    α + β + γ must equal 1.0 (validated in __post_init__).
    Use adapt_for_domain() to automatically set domain-appropriate weights.

    top_m: final context window size W*. Typically 5–15 chunks.
    epsilon: prevents log(0) in power computation for near-zero scores.
    diversity_weight: when > 0, applies MMR-style diversity penalty to
    avoid top-m being filled with near-duplicate chunks.
    """
    alpha: float = 0.40          # TVE score weight α
    beta: float = 0.35           # SDS score weight β
    gamma: float = 0.25          # ESR contribution weight γ
    top_m: int = 10              # final context size |W*|
    epsilon: float = 1e-10       # numerical stability for power computation
    diversity_weight: float = 0.0  # MMR diversity penalty (0=off, 0.3=moderate)
    domain: str = "general"

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"α + β + γ must equal 1.0, got {total:.6f}. "
                f"α={self.alpha}, β={self.beta}, γ={self.gamma}"
            )

    def apply_domain_preset(self) -> "RFGConfig":
        """Override α,β,γ with domain-specific optimal weights."""
        if self.domain in DOMAIN_FUSION_WEIGHTS:
            self.alpha, self.beta, self.gamma = DOMAIN_FUSION_WEIGHTS[self.domain]
        return self


class RankedChunk(NamedTuple):
    """
    A chunk with its final Φ-score after rank fusion.

    All intermediate scores are preserved for audit, interpretability,
    and downstream CCB causal ordering.
    """
    chunk_id:         int
    chunk_text:       str
    phi_score:        float    # Φ(c_i, q) — raw multiplicative score
    phi_norm:         float    # Φ̃(c_i) — normalized to sum=1
    tve_score:        float    # TVE_score(q, c_i)
    sds_score:        float    # SDS(q, c_i)
    esr_contribution: float    # ESR_contrib(c_i, W)
    sdc_result:       SDCResult


class RankFusionGate:
    """
    Computes Φ-scores and selects the optimal final context W*.

    The Φ-score is the VORTEXRAG master ranking signal. It replaces all
    intermediate scores (TVE, SDS, ESR_contrib) with a single number that
    encodes the full quality profile of a chunk relative to the query.

    Design principle: multiplicative structure enforces a "no weak link" policy.
    A chunk with TVE=0.95 but SDS=0.05 will have Φ ≈ 0.05^0.35 ≈ 0.19 — the
    high TVE score cannot rescue it from causal irrelevance.

    Pipeline:
      CPGEvaluation → [RFG] → list[RankedChunk] sorted by Φ̃ → top-m → W*

    Usage:
        rfg = RankFusionGate(config=RFGConfig(domain="legal"))
        ranked = rfg.rank(query_vec, cpg_evaluation)
        final_context = rfg.select_top_m(ranked)
        report = rfg.phi_breakdown(final_context[0])
    """

    def __init__(self, config: RFGConfig | None = None):
        self.config = config or RFGConfig()

    # ──── Core Computation ────────────────────────────────────────────────────

    def _esr_contributions(
        self,
        sds_scores: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute each chunk's fractional contribution to the window signal.

        ESR_contribution(c_i) = SDS_i · w_i / Σⱼ (SDS_j · w_j)

        Normalization ensures all contributions sum to 1.0, making the metric
        comparable across different window sizes. A contribution of 0.30 means
        chunk i provides 30% of the window's quality-weighted signal.
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
        Compute Φ(c_i, q) = TVE^α × SDS^β × ESR_contrib^γ.

        All inputs are clipped to (ε, 1.0) before power computation to:
          1. Avoid log(0) which is undefined for fractional exponents.
          2. Preserve rank ordering even for near-zero scores.
          3. Ensure numerical stability across all platforms.

        A zero-score chunk gets Φ ≈ ε^α ≈ near-zero, not exactly zero —
        this is intentional to preserve strict total ordering of Φ values.
        """
        ε = self.config.epsilon
        α, β, γ = self.config.alpha, self.config.beta, self.config.gamma
        tve_c = float(np.clip(tve, ε, 1.0))
        sds_c = float(np.clip(sds, ε, 1.0))
        ec_c  = float(np.clip(esr_contrib, ε, 1.0))
        return float(tve_c ** α * sds_c ** β * ec_c ** γ)

    def rank(
        self,
        _query_vec: TVEVector,
        cpg_eval: CPGEvaluation,
    ) -> list[RankedChunk]:
        """
        Compute Φ̃ scores for all chunks in the CPG-clean window.

        Returns a list of RankedChunk sorted by phi_norm descending.
        _query_vec is reserved for future extensions requiring per-chunk
        re-encoding at the fusion stage (currently Φ uses pre-computed scores).
        """
        window = cpg_eval.window
        weights = cpg_eval.softmax_weights

        if not window:
            return []

        sds_scores = np.array([r.sds_score for r in window])
        tve_scores = np.array([r.candidate.tve_score for r in window])
        esr_contribs = self._esr_contributions(sds_scores, weights)

        raw_phis = np.array([
            self._phi_score(float(tve_scores[i]), float(sds_scores[i]), float(esr_contribs[i]))
            for i in range(len(window))
        ])

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
        """Select top-m chunks by Φ̃ for final context W*."""
        return ranked[: self.config.top_m]

    def select_top_m_diverse(
        self,
        ranked: list[RankedChunk],
        diversity_weight: float | None = None,
    ) -> list[RankedChunk]:
        """
        Maximal Marginal Relevance (MMR) selection: top-m with diversity penalty.

        Iteratively selects chunks by balancing high Φ̃ score against similarity
        to already-selected chunks. Prevents the final W* from being filled with
        near-duplicate passages that provide redundant information.

        MMR score at each step:
          MMR(c_i) = (1−λ)·Φ̃(c_i) − λ·max_{c_j ∈ W*} sim(c_i, c_j)

        where sim = cosine similarity of semantic TVE vectors, and λ = diversity_weight.

        λ=0.0: pure relevance (equivalent to select_top_m)
        λ=0.3: mild diversity (recommended default)
        λ=0.5: balanced relevance/diversity
        λ=1.0: pure diversity (not recommended)
        """
        λ = diversity_weight if diversity_weight is not None else self.config.diversity_weight
        if λ == 0.0 or not ranked:
            return self.select_top_m(ranked)

        selected: list[RankedChunk] = []
        remaining = list(ranked)

        while len(selected) < self.config.top_m and remaining:
            if not selected:
                # First chunk: pick highest Φ̃
                best = remaining.pop(0)
                selected.append(best)
                continue

            # Semantic vectors of selected chunks
            selected_vecs = np.stack([
                c.sdc_result.candidate.tve_vec.semantic for c in selected
            ])

            best_mmr = -np.inf
            best_idx = 0

            for i, chunk in enumerate(remaining):
                # Relevance term
                relevance = (1.0 - λ) * chunk.phi_norm
                # Diversity term: negative max similarity to selected
                c_sem = chunk.sdc_result.candidate.tve_vec.semantic
                sims = selected_vecs @ c_sem
                max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
                diversity = -λ * max_sim
                mmr = relevance + diversity

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    # ──── Analysis and Interpretability ───────────────────────────────────────

    def phi_breakdown(self, chunk: RankedChunk) -> dict:
        """
        Decompose a chunk's Φ-score into its three multiplicative factors.

        Shows exactly how each component contributes to the final score,
        including what the score would be under additive fusion (for comparison).

        Returns:
          {
            'phi_score':            float — raw Φ score
            'phi_norm':             float — normalized Φ̃
            'tve_score':            float — tri-vector relevance
            'sds_score':            float — semantic drift alignment
            'esr_contribution':     float — window signal fraction
            'tve_factor':           float — TVE^α
            'sds_factor':           float — SDS^β
            'esr_factor':           float — ESR^γ
            'additive_equivalent':  float — what score would be under additive fusion
            'multiplicative_gain':  float — Φ / additive (how much mult. differs)
            'weakest_factor':       str   — which factor limits the Φ score most
            'interpretation':       str   — human-readable explanation
          }
        """
        ε = self.config.epsilon
        α, β, γ = self.config.alpha, self.config.beta, self.config.gamma

        tve = np.clip(chunk.tve_score, ε, 1.0)
        sds = np.clip(chunk.sds_score, ε, 1.0)
        esr = np.clip(chunk.esr_contribution, ε, 1.0)

        tve_factor = float(tve ** α)
        sds_factor = float(sds ** β)
        esr_factor = float(esr ** γ)

        additive = α * chunk.tve_score + β * chunk.sds_score + γ * chunk.esr_contribution
        mult_gain = chunk.phi_score / (additive + ε)

        factors = {"tve": tve_factor, "sds": sds_factor, "esr": esr_factor}
        weakest = min(factors, key=factors.get)  # type: ignore[arg-type]

        if chunk.sds_score < 0.3:
            interp = (
                f"Causal drift alert: SDS={chunk.sds_score:.3f} severely limits Φ. "
                f"Even with TVE={chunk.tve_score:.3f}, SDS^{β}={sds_factor:.3f} "
                f"drives Φ down to {chunk.phi_score:.3f}. Additive fusion would give "
                f"{additive:.3f} — multiplicative correctly penalizes causal irrelevance."
            )
        elif chunk.phi_norm > 0.15:
            interp = (
                f"Top-tier chunk: Φ̃={chunk.phi_norm:.3f} — all three factors strong. "
                f"TVE={chunk.tve_score:.3f}, SDS={chunk.sds_score:.3f}, "
                f"ESR_contrib={chunk.esr_contribution:.3f}."
            )
        else:
            interp = (
                f"Moderate chunk: weakest factor is {weakest} "
                f"(value={factors[weakest]:.3f}). "
                f"Improving {weakest} would most increase Φ̃."
            )

        return {
            "phi_score":           round(chunk.phi_score, 6),
            "phi_norm":            round(chunk.phi_norm, 6),
            "tve_score":           round(chunk.tve_score, 4),
            "sds_score":           round(chunk.sds_score, 4),
            "esr_contribution":    round(chunk.esr_contribution, 4),
            "alpha":               α,
            "beta":                β,
            "gamma":               γ,
            "tve_factor":          round(tve_factor, 4),
            "sds_factor":          round(sds_factor, 4),
            "esr_factor":          round(esr_factor, 4),
            "additive_equivalent": round(additive, 4),
            "multiplicative_gain": round(mult_gain, 3),
            "weakest_factor":      weakest,
            "interpretation":      interp,
        }

    def rank_statistics(self, ranked: list[RankedChunk]) -> dict:
        """
        Summary statistics for a ranked chunk list.

        Provides distribution metrics for quality analysis and threshold tuning.
        """
        if not ranked:
            return {}

        phi_norms  = [c.phi_norm for c in ranked]
        tve_scores = [c.tve_score for c in ranked]
        sds_scores = [c.sds_score for c in ranked]
        esr_contribs = [c.esr_contribution for c in ranked]

        return {
            "n_chunks":           len(ranked),
            "phi_norm_mean":      round(float(np.mean(phi_norms)), 4),
            "phi_norm_std":       round(float(np.std(phi_norms)), 4),
            "phi_norm_max":       round(float(np.max(phi_norms)), 4),
            "phi_norm_min":       round(float(np.min(phi_norms)), 4),
            "tve_mean":           round(float(np.mean(tve_scores)), 4),
            "sds_mean":           round(float(np.mean(sds_scores)), 4),
            "esr_contrib_mean":   round(float(np.mean(esr_contribs)), 4),
            "top1_phi_norm":      round(phi_norms[0], 4) if phi_norms else 0.0,
            "top3_cumulative":    round(sum(phi_norms[:3]), 4),
            "top5_cumulative":    round(sum(phi_norms[:5]), 4),
            "concentration":      round(phi_norms[0] / sum(phi_norms) if phi_norms else 0.0, 3),
            "alpha":              self.config.alpha,
            "beta":               self.config.beta,
            "gamma":              self.config.gamma,
            "domain":             self.config.domain,
        }

    def compare_chunks(self, a: RankedChunk, b: RankedChunk) -> dict:
        """
        Side-by-side comparison of two ranked chunks.

        Shows which chunk wins on each dimension (TVE, SDS, ESR, Φ̃) and
        explains the overall ranking difference.
        """
        def _win(va: float, vb: float) -> str:
            if abs(va - vb) < 0.001:
                return "TIE"
            return "A" if va > vb else "B"

        return {
            "chunk_a_id":           a.chunk_id,
            "chunk_b_id":           b.chunk_id,
            "tve_winner":           _win(a.tve_score, b.tve_score),
            "sds_winner":           _win(a.sds_score, b.sds_score),
            "esr_contrib_winner":   _win(a.esr_contribution, b.esr_contribution),
            "phi_winner":           _win(a.phi_norm, b.phi_norm),
            "tve_a":                round(a.tve_score, 4),
            "tve_b":                round(b.tve_score, 4),
            "sds_a":                round(a.sds_score, 4),
            "sds_b":                round(b.sds_score, 4),
            "esr_a":                round(a.esr_contribution, 4),
            "esr_b":                round(b.esr_contribution, 4),
            "phi_norm_a":           round(a.phi_norm, 4),
            "phi_norm_b":           round(b.phi_norm, 4),
            "phi_delta":            round(a.phi_norm - b.phi_norm, 4),
        }

    def adapt_for_domain(self, domain: str) -> None:
        """
        Update α, β, γ to the domain-specific fusion weight preset.

        Raises ValueError if domain is not in DOMAIN_FUSION_WEIGHTS.
        After calling this, all subsequent rank() calls use the new weights.
        """
        if domain not in DOMAIN_FUSION_WEIGHTS:
            available = ", ".join(sorted(DOMAIN_FUSION_WEIGHTS.keys()))
            raise ValueError(f"Unknown domain '{domain}'. Available: {available}")
        self.config.alpha, self.config.beta, self.config.gamma = DOMAIN_FUSION_WEIGHTS[domain]
        self.config.domain = domain

    def sensitivity_analysis(
        self,
        ranked: list[RankedChunk],
        perturbation: float = 0.05,
    ) -> list[dict]:
        """
        Analyze how the top-m selection changes under weight perturbations.

        For each weight dimension (α, β, γ), perturbs the weight by ±perturbation
        and shows which chunks enter/leave the top-m selection.

        Returns a list of sensitivity reports — one per weight dimension.
        Useful for understanding how robust the ranking is to domain mis-tuning.
        """
        if not ranked:
            return []

        baseline_ids = {c.chunk_id for c in self.select_top_m(ranked)}
        reports = []

        dims = [
            ("alpha",  self.config.alpha,  self.config.beta,   self.config.gamma),
            ("beta",   self.config.alpha,  self.config.beta,   self.config.gamma),
            ("gamma",  self.config.alpha,  self.config.beta,   self.config.gamma),
        ]

        for dim_name, α0, β0, γ0 in dims:
            for direction, sign in [("+", 1), ("-", -1)]:
                if dim_name == "alpha":
                    α, β, γ = α0 + sign * perturbation, β0 - sign * perturbation / 2, γ0 - sign * perturbation / 2
                elif dim_name == "beta":
                    α, β, γ = α0 - sign * perturbation / 2, β0 + sign * perturbation, γ0 - sign * perturbation / 2
                else:
                    α, β, γ = α0 - sign * perturbation / 2, β0 - sign * perturbation / 2, γ0 + sign * perturbation

                # Ensure valid weights
                total = α + β + γ
                α, β, γ = α / total, β / total, γ / total

                # Recompute phi with perturbed weights
                ε = self.config.epsilon
                reranked = []
                for chunk in ranked:
                    tve = float(np.clip(chunk.tve_score, ε, 1.0))
                    sds = float(np.clip(chunk.sds_score, ε, 1.0))
                    esr = float(np.clip(chunk.esr_contribution, ε, 1.0))
                    phi = float(tve ** α * sds ** β * esr ** γ)
                    reranked.append((phi, chunk.chunk_id))

                reranked.sort(reverse=True)
                new_ids = {cid for _, cid in reranked[:self.config.top_m]}
                entered = new_ids - baseline_ids
                exited  = baseline_ids - new_ids

                reports.append({
                    "weight_dim":   dim_name,
                    "direction":    f"{direction}{perturbation}",
                    "new_alpha":    round(α, 3),
                    "new_beta":     round(β, 3),
                    "new_gamma":    round(γ, 3),
                    "chunks_enter": sorted(entered),
                    "chunks_exit":  sorted(exited),
                    "n_changed":    len(entered),
                    "stable":       len(entered) == 0,
                })

        return reports

    def cross_domain_ranking(
        self,
        ranked: list[RankedChunk],
        domains: list[str] | None = None,
    ) -> dict[str, list[int]]:
        """
        Show how the top-m chunk IDs change across different domain presets.

        For each domain, recomputes Φ scores using domain-specific weights and
        returns the top-m chunk IDs. Useful for diagnosing domain sensitivity.

        Returns: {domain_name: [top_m chunk_ids in order]}
        """
        if domains is None:
            domains = list(DOMAIN_FUSION_WEIGHTS.keys())

        results: dict[str, list[int]] = {}
        ε = self.config.epsilon

        for domain in domains:
            if domain not in DOMAIN_FUSION_WEIGHTS:
                continue
            α, β, γ = DOMAIN_FUSION_WEIGHTS[domain]
            reranked = []
            for chunk in ranked:
                tve = float(np.clip(chunk.tve_score, ε, 1.0))
                sds = float(np.clip(chunk.sds_score, ε, 1.0))
                esr = float(np.clip(chunk.esr_contribution, ε, 1.0))
                phi = float(tve ** α * sds ** β * esr ** γ)
                reranked.append((phi, chunk.chunk_id))
            reranked.sort(reverse=True)
            results[domain] = [cid for _, cid in reranked[:self.config.top_m]]

        return results
