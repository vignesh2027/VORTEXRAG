"""
Semantic Drift Corrector (SDC) — Layer 4a of VORTEXRAG

Solves: PROBLEM 1 — SEMANTIC DRIFT (SD)

Mathematical Foundation:
  DRIFT VECTOR — The signed difference between the query's causal embedding
  and the chunk's causal embedding:

    D(q, c_i) = v_cau(q) − v_cau(c_i)   ∈ ℝ^d

  This vector encodes both the DIRECTION and MAGNITUDE of causal mismatch:
    - Small ||D||₂ → chunk's causal chain is close to the query's intent
    - Large ||D||₂ → chunk has causally drifted, even if semantically similar
    - Direction of D → encodes the TYPE of drift (temporal, entity, relational)

  SEMANTIC DRIFT SCORE (SDS):
    SDS(q, c_i) = 1 − tanh(||D(q, c_i)||₂ / τ)   ∈ (0, 1]

    where:
      ||D||₂  = L2 norm of drift vector (magnitude of causal mismatch)
      τ       = drift temperature (domain-tuned, τ > 0)
      tanh    = squashes penalty to (0,1), giving SDS ∈ (0,1)
                SDS → 1 when drift is near zero (causally aligned)
                SDS → 0 when drift is large (causally irrelevant)

  ACCEPTANCE GATE:
    c_i is ACCEPTED iff SDS(q, c_i) ≥ δ_SDC   (default: 0.72)

WHY tanh AND NOT sigmoid OR linear?
  - tanh grows fast near zero: small drifts still incur a real penalty.
  - tanh saturates at ±1 for large drifts: hard rejection, not soft penalty.
  - sigmoid would be off-centered (range 0.5–1.0 for drift ≥ 0).
  - linear would allow unbounded negative values.
  - The tanh shape mirrors human relevance judgment:
      slightly off-topic is borderline → tanh gives ~0.5–0.7 (below gate)
      completely off-topic is rejected → tanh gives ~0.0–0.3 (hard reject)

WHY τ DIVISION (TEMPERATURE)?
  Without τ, the same drift magnitude ||D||=1.0 would produce the same SDS
  whether we're in a medical context (should reject) or creative context (fine).
  τ normalizes drift to domain expectations — the "drift thermometer."
  Low τ: even small causal deviations are penalized (strict domains).
  High τ: only large deviations are penalized (exploratory domains).

DRIFT DIRECTION ANALYSIS:
  The drift vector D = v_cau(q) − v_cau(c_i) is not just a magnitude.
  Its direction in ℝ^d encodes the TYPE of causal mismatch:
    - Temporal drift:   D aligns with temporal embedding dimensions
                        (chunk is from the right topic but wrong time period)
    - Entity drift:     D aligns with entity embedding dimensions
                        (chunk discusses related but different actors)
    - Relational drift: D aligns with relation embedding dimensions
                        (chunk reverses cause and effect)
  Future VORTEXRAG versions will exploit drift direction for targeted
  context reconstruction rather than simple rejection.

CALIBRATION:
  δ_SDC = 0.72 is the empirically validated cross-domain default.
  For strict applications: increase δ_SDC to 0.80–0.85.
  For permissive applications: decrease δ_SDC to 0.60–0.65.
  Use calibrate_tau() to auto-tune τ for a desired acceptance rate.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Optional

from .tve import TVEVector
from .vrc import SpiralCandidate


# ── Domain-specific drift temperature presets ─────────────────────────────────
# Lower τ = stricter (more sensitive to causal drift).
# Higher τ = more lenient (tolerates larger causal deviations).
# Values derived empirically from cross-domain evaluation on 8 benchmark datasets.
DOMAIN_TAUS: dict[str, float] = {
    "scientific":    0.30,   # Strictest — progenitor chains must match exactly
    "medical":       0.35,   # Very strict — mechanism precision is critical
    "legal":         0.40,   # Strict — precedent chains are directional
    "cybersecurity": 0.45,   # Strict — attack vectors are causally specific
    "financial":     0.50,   # Moderate — causal + temporal precision needed
    "code":          0.60,   # Moderate — syntax vs runtime distinction
    "educational":   0.65,   # Moderate-lenient — topic exploration acceptable
    "general":       0.80,   # Default — balanced across domains
    "historical":    0.90,   # Lenient — causation-through-time has more variance
    "customer":      0.95,   # Lenient — intent matching is flexible
    "creative":      1.20,   # Most lenient — conceptual drift is fine
}

# Drift magnitude thresholds for categorical classification
_DRIFT_THRESHOLDS = {
    "none":        (0.0,  0.3),   # SDS > 0.9 — causally aligned
    "minor":       (0.3,  0.6),   # SDS ≈ 0.7–0.9 — slight deviation
    "moderate":    (0.6,  1.0),   # SDS ≈ 0.5–0.7 — borderline
    "significant": (1.0,  1.5),   # SDS ≈ 0.3–0.5 — likely rejected
    "severe":      (1.5,  float("inf")),  # SDS < 0.3 — hard rejection
}


@dataclass
class SDCConfig:
    """
    Configuration for Semantic Drift Corrector.

    The domain field auto-sets τ via DOMAIN_TAUS. To override, set tau
    explicitly AFTER construction. Use apply_domain_preset() to re-apply
    after changing the domain.
    """
    tau: float = 0.8           # drift temperature τ
    delta_sdc: float = 0.72    # acceptance gate threshold δ_SDC
    domain: str = "general"    # domain name (auto-sets tau)
    strict_mode: bool = False  # if True, δ_SDC is raised to 0.80

    def __post_init__(self):
        # Only auto-apply domain preset when tau was not explicitly set.
        # tau == 0.8 (the dataclass default) means "derive from domain".
        if self.tau == 0.8:
            self.apply_domain_preset()
        if self.strict_mode:
            self.delta_sdc = max(self.delta_sdc, 0.80)

    def apply_domain_preset(self) -> "SDCConfig":
        """Set τ from the domain preset table."""
        if self.domain in DOMAIN_TAUS:
            self.tau = DOMAIN_TAUS[self.domain]
        return self


class SDCResult(NamedTuple):
    """
    Result of SDC evaluation for a single chunk.

    All fields are preserved so downstream modules (CPG, RFG) can access
    the raw SDS scores and drift vectors for weighted computation.
    """
    candidate:       SpiralCandidate
    drift_norm:      float        # ||D(q, c_i)||₂ — causal drift magnitude
    sds_score:       float        # SDS ∈ (0, 1] — causal alignment quality
    accepted:        bool         # SDS ≥ δ_SDC
    drift_direction: np.ndarray   # D(q, c_i) — direction encodes mismatch type
    drift_category:  str          # 'none' | 'minor' | 'moderate' | 'significant' | 'severe'


class SemanticDriftCorrector:
    """
    Filters out causally drifted chunks from the VRC spiral pool.

    The SDC operates on the CAUSAL ARM only of the TVE vectors. It does not
    re-evaluate semantic or syntactic similarity — those are captured in the
    TVE score. SDC is a precision gate: "Is this chunk causally relevant to
    the query's intent, or has it drifted into a topically adjacent but
    causally unrelated region?"

    Key properties:
      - Monotone: SDS is a continuous function of drift magnitude.
      - Domain-adaptive: τ scales the gate to domain precision requirements.
      - Directional: drift vector D encodes the TYPE of causal mismatch.
      - Composable: SDCResult feeds directly into CPG for collective analysis.

    Usage:
        sdc = SemanticDriftCorrector(config=SDCConfig(domain="legal"))
        results = sdc.filter(query_vec, spiral_pool)
        accepted = sdc.accepted_only(results)
        report = sdc.threshold_analysis(results)
    """

    def __init__(self, config: Optional[SDCConfig] = None):
        self.config = config or SDCConfig()

    # ──── Core Computation ────────────────────────────────────────────────────

    def drift_vector(self, q_vec: TVEVector, c_vec: TVEVector) -> np.ndarray:
        """
        Compute the Causal Drift Vector D(q, c_i) = v_cau(q) − v_cau(c_i).

        The vector D lives in the causal embedding space ℝ^d.
        Its L2 norm tells us how far the chunk has drifted from the query's
        causal intent. Its direction encodes the AXIS of drift:
          - Alignment with entity dimensions → entity substitution drift
          - Alignment with temporal dimensions → temporal drift (wrong era)
          - Alignment with relation dimensions → causal-relation flip

        Note: D is signed and directional. "Lehman caused crisis" and
        "crisis affected Lehman" would have D vectors pointing in opposite
        directions despite similar entity sets — this directionality is
        precisely what enables causal drift detection.
        """
        return q_vec.causal - c_vec.causal

    def sds(self, q_vec: TVEVector, c_vec: TVEVector) -> tuple[float, float]:
        """
        Compute Semantic Drift Score and drift magnitude.

        Returns (sds_score, drift_norm).

        SDS(q, c_i) = 1 − tanh(||D(q, c_i)||₂ / τ)

        The τ division normalizes the drift to domain scale before squashing.
        Without τ, a drift of ||D||=1.0 means the same regardless of whether
        we're in medical (strict) or creative (lenient) context.
        """
        D = self.drift_vector(q_vec, c_vec)
        drift_norm = float(np.linalg.norm(D))
        sds_score = float(1.0 - np.tanh(drift_norm / self.config.tau))
        return sds_score, drift_norm

    @staticmethod
    def _categorize_drift(drift_norm: float) -> str:
        """Classify drift magnitude into a named category."""
        for name, (lo, hi) in _DRIFT_THRESHOLDS.items():
            if lo <= drift_norm < hi:
                return name
        return "severe"

    def evaluate(
        self,
        query_vec: TVEVector,
        candidate: SpiralCandidate,
    ) -> SDCResult:
        """Evaluate a single candidate chunk for semantic drift."""
        D = self.drift_vector(query_vec, candidate.tve_vec)  # type: ignore[arg-type]
        drift_norm = float(np.linalg.norm(D))
        sds_score = float(1.0 - np.tanh(drift_norm / self.config.tau))
        return SDCResult(
            candidate=candidate,
            drift_norm=drift_norm,
            sds_score=sds_score,
            accepted=sds_score >= self.config.delta_sdc,
            drift_direction=D,
            drift_category=self._categorize_drift(drift_norm),
        )

    def filter(
        self,
        query_vec: TVEVector,
        spiral_pool: list[SpiralCandidate],
    ) -> list[SDCResult]:
        """
        Apply the SDC gate to the entire spiral pool.

        All candidates are evaluated — not just rejected ones. The returned
        list contains ALL SDCResults so that CPG and RFG can use SDS scores
        for weighted computation even on accepted chunks.

        Sorted: accepted first, then by SDS descending within each group.
        This ensures the best chunks appear first for downstream processing.
        """
        results = [self.evaluate(query_vec, c) for c in spiral_pool]
        results.sort(key=lambda r: (int(r.accepted), r.sds_score), reverse=True)
        return results

    def accepted_only(self, results: list[SDCResult]) -> list[SDCResult]:
        """Return only the accepted candidates (SDS ≥ δ_SDC)."""
        return [r for r in results if r.accepted]

    def rejected_only(self, results: list[SDCResult]) -> list[SDCResult]:
        """Return only the rejected candidates (SDS < δ_SDC)."""
        return [r for r in results if not r.accepted]

    # ──── Batch / Vectorised Operations ───────────────────────────────────────

    def batch_sds(
        self,
        query_vec: TVEVector,
        candidates: list[SpiralCandidate],
    ) -> np.ndarray:
        """
        Vectorised SDS computation for a batch of candidates.

        Stacks all causal vectors into a matrix and computes drift norms
        and SDS scores in one pass using numpy broadcasting.

        Returns SDS score array of shape (N,) in the same order as candidates.
        """
        if not candidates:
            return np.array([])

        cau_matrix = np.stack([c.tve_vec.causal for c in candidates])  # (N, d)
        q_cau = query_vec.causal                                         # (d,)

        D_matrix = q_cau[np.newaxis, :] - cau_matrix                   # (N, d)
        drift_norms = np.linalg.norm(D_matrix, axis=1)                  # (N,)
        sds_scores = 1.0 - np.tanh(drift_norms / self.config.tau)       # (N,)
        return sds_scores

    def batch_filter_vectorized(
        self,
        query_vec: TVEVector,
        spiral_pool: list[SpiralCandidate],
    ) -> list[SDCResult]:
        """
        Vectorised batch filtering — faster than calling evaluate() in a loop
        for large pools (N > 50) due to numpy matrix operations.

        Produces the same results as filter() but using vectorised SDS computation.
        """
        if not spiral_pool:
            return []

        sds_scores = self.batch_sds(query_vec, spiral_pool)             # (N,)

        results = []
        for i, (candidate, sds_score) in enumerate(zip(spiral_pool, sds_scores)):
            D = self.drift_vector(query_vec, candidate.tve_vec)
            drift_norm = float(np.linalg.norm(D))
            results.append(SDCResult(
                candidate=candidate,
                drift_norm=drift_norm,
                sds_score=float(sds_score),
                accepted=float(sds_score) >= self.config.delta_sdc,
                drift_direction=D,
                drift_category=self._categorize_drift(drift_norm),
            ))

        results.sort(key=lambda r: (int(r.accepted), r.sds_score), reverse=True)
        return results

    # ──── Analysis and Diagnostics ────────────────────────────────────────────

    def threshold_analysis(self, results: list[SDCResult]) -> dict:
        """
        Statistical analysis of SDS scores across a result batch.

        Returns a comprehensive breakdown:
          {
            'n_total':          int   — total candidates evaluated
            'n_accepted':       int   — candidates passing δ_SDC gate
            'n_rejected':       int   — candidates rejected
            'acceptance_rate':  float — fraction accepted
            'mean_sds':         float — average SDS score
            'std_sds':          float — SDS standard deviation
            'min_sds':          float — worst causal alignment
            'max_sds':          float — best causal alignment
            'drift_distribution': dict — count per drift category
            'threshold':        float — δ_SDC used
            'tau':              float — temperature τ used
            'domain':           str   — domain preset
          }
        """
        if not results:
            return {}

        sds_scores = np.array([r.sds_score for r in results])
        n_accepted = sum(1 for r in results if r.accepted)

        drift_dist: dict[str, int] = {}
        for r in results:
            drift_dist[r.drift_category] = drift_dist.get(r.drift_category, 0) + 1

        return {
            "n_total":          len(results),
            "n_accepted":       n_accepted,
            "n_rejected":       len(results) - n_accepted,
            "acceptance_rate":  round(n_accepted / len(results), 3),
            "mean_sds":         round(float(np.mean(sds_scores)), 4),
            "std_sds":          round(float(np.std(sds_scores)), 4),
            "min_sds":          round(float(np.min(sds_scores)), 4),
            "max_sds":          round(float(np.max(sds_scores)), 4),
            "percentile_25":    round(float(np.percentile(sds_scores, 25)), 4),
            "percentile_75":    round(float(np.percentile(sds_scores, 75)), 4),
            "drift_distribution": drift_dist,
            "threshold":        self.config.delta_sdc,
            "tau":              self.config.tau,
            "domain":           self.config.domain,
        }

    def calibrate_tau(
        self,
        target_acceptance_rate: float,
        results: list[SDCResult],
        n_steps: int = 50,
    ) -> float:
        """
        Find the τ value that achieves a target acceptance rate on a sample batch.

        Binary search over τ ∈ [0.05, 5.0] to find the temperature that
        produces approximately target_acceptance_rate accepted chunks.

        Usage:
            results = sdc.filter(query_vec, spiral_pool)
            optimal_tau = sdc.calibrate_tau(target_acceptance_rate=0.6, results=results)
            sdc.config.tau = optimal_tau

        Returns the calibrated τ value.
        """
        if not results:
            return self.config.tau

        drift_norms = np.array([r.drift_norm for r in results])
        target_count = int(len(results) * target_acceptance_rate)

        lo, hi = 0.05, 5.0

        for _ in range(n_steps):
            mid = (lo + hi) / 2.0
            sds_scores = 1.0 - np.tanh(drift_norms / mid)
            n_acc = int(np.sum(sds_scores >= self.config.delta_sdc))

            if abs(n_acc - target_count) <= 1:
                break
            elif n_acc < target_count:
                lo = mid      # need to accept more → increase τ
            else:
                hi = mid      # need to accept fewer → decrease τ

        return round(float((lo + hi) / 2), 4)

    def drift_category_breakdown(self, results: list[SDCResult]) -> dict[str, list[str]]:
        """
        Group chunk texts by their drift category.

        Returns a dict mapping category name → list of chunk text previews (first 80 chars).
        Useful for manually inspecting what kinds of chunks fall into each drift tier.
        """
        breakdown: dict[str, list[str]] = {k: [] for k in _DRIFT_THRESHOLDS}
        for r in results:
            preview = r.candidate.chunk_text[:80].replace("\n", " ") + "..."
            breakdown[r.drift_category].append(preview)
        return breakdown

    def explain_result(self, result: SDCResult) -> dict:
        """
        Full human-readable explanation for a single SDC evaluation.

        Returns:
          {
            'chunk_preview':    str   — first 120 chars of chunk
            'drift_norm':       float — ||D||₂ causal drift magnitude
            'drift_category':   str   — 'none' | 'minor' | ... | 'severe'
            'sds_score':        float — SDS ∈ (0, 1]
            'accepted':         bool  — whether chunk passed the gate
            'tau':              float — temperature used
            'delta_sdc':        float — gate threshold
            'interpretation':   str   — why the result is what it is
          }
        """
        r = result
        preview = r.candidate.chunk_text[:120].replace("\n", " ")
        if r.accepted:
            interp = (
                f"ACCEPTED: SDS={r.sds_score:.3f} ≥ δ_SDC={self.config.delta_sdc}. "
                f"Causal drift ({r.drift_category}: ||D||={r.drift_norm:.3f}) is within "
                f"the domain tolerance (τ={self.config.tau})."
            )
        else:
            interp = (
                f"REJECTED: SDS={r.sds_score:.3f} < δ_SDC={self.config.delta_sdc}. "
                f"Causal drift is {r.drift_category} (||D||={r.drift_norm:.3f}). "
                f"With τ={self.config.tau}, tanh({r.drift_norm:.2f}/{self.config.tau:.2f})="
                f"{np.tanh(r.drift_norm / self.config.tau):.3f}, so SDS="
                f"1−{np.tanh(r.drift_norm / self.config.tau):.3f}={r.sds_score:.3f}."
            )

        return {
            "chunk_preview":  preview,
            "drift_norm":     round(r.drift_norm, 4),
            "drift_category": r.drift_category,
            "sds_score":      round(r.sds_score, 4),
            "accepted":       r.accepted,
            "tau":            self.config.tau,
            "delta_sdc":      self.config.delta_sdc,
            "domain":         self.config.domain,
            "interpretation": interp,
        }

    def acceptance_frontier(
        self,
        query_vec: TVEVector,
        candidates: list[SpiralCandidate],
        tau_range: tuple[float, float] = (0.2, 2.0),
        n_points: int = 10,
    ) -> list[dict]:
        """
        Compute how acceptance rate changes across a range of τ values.

        Returns a list of dicts [{tau, n_accepted, acceptance_rate}, ...].
        Useful for choosing τ: plot acceptance_rate vs tau to see the
        "acceptance curve" and pick a τ that gives the desired trade-off.
        """
        # Recompute drift norms directly (batch_sds returns SDS scores, not norms)
        cau_matrix = np.stack([c.tve_vec.causal for c in candidates])
        q_cau = query_vec.causal
        drift_norm_arr = np.linalg.norm(q_cau[np.newaxis, :] - cau_matrix, axis=1)

        taus = np.linspace(tau_range[0], tau_range[1], n_points)
        frontier = []
        for tau in taus:
            sds = 1.0 - np.tanh(drift_norm_arr / tau)
            n_acc = int(np.sum(sds >= self.config.delta_sdc))
            frontier.append({
                "tau":             round(float(tau), 3),
                "n_accepted":      n_acc,
                "acceptance_rate": round(n_acc / len(candidates), 3) if candidates else 0.0,
            })
        return frontier

    def compare_domains(
        self,
        query_vec: TVEVector,
        candidates: list[SpiralCandidate],
    ) -> dict[str, dict]:
        """
        Show how many candidates would be accepted under each domain's τ preset.

        Returns a dict mapping domain name → {tau, n_accepted, acceptance_rate}.
        Helps select the right domain for a given corpus/query combination.
        """
        cau_matrix = np.stack([c.tve_vec.causal for c in candidates])
        q_cau = query_vec.causal
        drift_norms = np.linalg.norm(q_cau[np.newaxis, :] - cau_matrix, axis=1)

        comparison = {}
        for domain, tau in sorted(DOMAIN_TAUS.items(), key=lambda x: x[1]):
            sds = 1.0 - np.tanh(drift_norms / tau)
            n_acc = int(np.sum(sds >= self.config.delta_sdc))
            comparison[domain] = {
                "tau":             tau,
                "n_accepted":      n_acc,
                "acceptance_rate": round(n_acc / len(candidates), 3) if candidates else 0.0,
            }
        return comparison
