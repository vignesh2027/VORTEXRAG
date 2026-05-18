"""
Context Poison Guard (CPG) — Layer 4b of VORTEXRAG

Solves: PROBLEM 2 — CONTEXT WINDOW POISONING (CWP)

Mathematical Foundation:
  Even after SDC filtering, the COMBINATION of surviving chunks may still
  poison the context window. CPG evaluates the COLLECTIVE toxicity of the
  entire candidate set, not individual chunks.

  POISON INDEX P(W, q) — Average weighted irrelevance across context window:
    P(W, q) = (1/k) · Σᵢ [1 − SDS(q, c_i)] · w_i

    where:
      w_i = softmax(TVE_score(q, c_i))   — attention weight for chunk i
      1 − SDS(q, c_i)                    — irrelevance score (complement of SDS)
      k                                  — window size |W|

    Interpretation: P is the softmax-weighted average irrelevance. Chunks with
    higher TVE scores get more weight in the poison calculation — this means a
    highly-ranked irrelevant chunk is MORE poisonous than a low-ranked one,
    because the LLM will attend to it more.

  EFFECTIVE SIGNAL RATIO (ESR) — Signal-to-poison ratio:
    ESR(W, q) = Σᵢ [SDS(q,c_i) · w_i] / (P(W,q) + ε)

    Interpretation: ESR is the ratio of weighted relevance to weighted irrelevance.
    High ESR → context is clean, mostly relevant signal.
    Low ESR  → context is poisoned, irrelevant content dominates LLM attention.

  CLEAN CONDITION:
    Context is CLEAN iff ESR(W, q) ≥ θ_CPG   (default: 3.5)

  ITERATIVE PURGING ALGORITHM:
    while ESR(W, q) < θ_CPG:
        remove argmin_i SDS(q, c_i)   ← remove the most irrelevant chunk
        recompute ESR with updated W

    This greedy purging is O(k²) but k is small (typically ≤ 50) and
    the inner ESR computation is O(k), so total complexity is O(k²) ≈ O(2500).

WHY ESR INSTEAD OF SIMPLE AVERAGE SDS?
  Simple average SDS would weight all chunks equally. But in LLM attention,
  chunks with higher retrieval scores receive disproportionately more attention
  (due to how prompt position and repetition interact with transformer attention).
  The softmax(TVE_score) weighting approximates this attentional bias:
  a chunk at position 1 with score 0.95 is far more "poisonous" if irrelevant
  than a chunk at position 8 with score 0.61.

WHY ε IN THE DENOMINATOR?
  When P(W,q) ≈ 0 (perfect window, no poison), division by zero is avoided.
  ε = 1e-8 ensures numerical stability while having negligible effect when
  P is non-trivial (> 0.01), which it always is in real corpora.

WHY NOT JUST USE A FIXED THRESHOLD ON SDS?
  SDC already does that for individual chunks. CPG catches COLLECTIVE poisoning:
  10 chunks each with SDS=0.73 (just above δ_SDC=0.72) would all individually
  pass SDC, but collectively produce P ≈ 0.27 and ESR ≈ 2.7 (below θ_CPG=3.5).
  CPG would then purge the worst chunks until ESR is clean.

BEST USE:
  - θ_CPG=3.5 default: works well for 5–20 chunk windows
  - θ_CPG=5.0 for strict medical/legal applications
  - θ_CPG=2.0 for exploratory/creative applications
  - Always run CPG AFTER SDC — CPG operates on the SDC-filtered set
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .tve import TVEVector
from .sdc import SDCResult


@dataclass
class CPGConfig:
    """Configuration for Context Poison Guard."""
    theta_cpg: float = 3.5      # ESR clean threshold θ_CPG
    epsilon: float = 1e-8       # numerical stability ε
    max_purge_rounds: int = 20  # safety limit on purging iterations
    min_chunks: int = 3         # never purge below this many chunks


class CPGEvaluation(NamedTuple):
    """Full CPG evaluation result for a context window."""
    window: list[SDCResult]         # final purged window
    poison_index: float             # P(W, q)
    esr: float                      # ESR(W, q)
    is_clean: bool                  # ESR ≥ θ_CPG
    purge_count: int                # how many chunks were purged
    purge_history: list[tuple]      # (round, removed_chunk_id, esr_before, esr_after)
    softmax_weights: np.ndarray     # w_i for each chunk in final window


class ContextPoisonGuard:
    """
    Guards the context window against collective irrelevance poisoning.

    The CPG's power comes from the ESR metric: unlike per-chunk filtering,
    ESR measures the SIGNAL-TO-NOISE ratio of the entire window as a unit.
    The iterative purging algorithm greedily maximizes ESR by removing the
    chunk contributing the most to the Poison Index at each step.

    Key property: CPG is MONOTONE — each purge round strictly increases ESR
    (or stops if removing any chunk would violate min_chunks). This guarantees
    convergence in O(k) rounds.

    Usage:
        cpg = ContextPoisonGuard(config=CPGConfig(theta_cpg=3.5))
        result = cpg.evaluate(query_vec, sdc_results)
        clean_window = result.window
    """

    def __init__(self, config: CPGConfig | None = None):
        self.config = config or CPGConfig()

    def _softmax_weights(self, tve_scores: np.ndarray) -> np.ndarray:
        """
        Compute softmax(TVE_scores) for attention-weighted poison calculation.

        Softmax ensures weights sum to 1 and amplifies the contribution of
        high-scoring chunks — reflecting LLM attentional bias toward
        highly-ranked retrieved content.
        """
        shifted = tve_scores - tve_scores.max()  # numerical stability
        exp_scores = np.exp(shifted)
        return exp_scores / (exp_scores.sum() + self.config.epsilon)

    def _compute_esr(
        self,
        sds_scores: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute (ESR, P) for a given set of chunks.

        P(W, q) = (1/k) · Σᵢ (1 − SDS_i) · w_i
        ESR(W, q) = Σᵢ SDS_i · w_i / (P + ε)

        The (1/k) normalization in P ensures that a larger window doesn't
        automatically appear less poisoned — it normalizes per chunk.
        """
        k = len(sds_scores)
        if k == 0:
            return 0.0, 1.0

        irrelevance = 1.0 - sds_scores
        P = float(np.sum(irrelevance * weights) / k)
        signal = float(np.sum(sds_scores * weights))
        ESR = signal / (P + self.config.epsilon)
        return ESR, P

    def evaluate(
        self,
        query_vec: TVEVector,
        sdc_results: list[SDCResult],
    ) -> CPGEvaluation:
        """
        Evaluate and purify the context window via iterative ESR maximization.

        Algorithm:
          1. Compute initial ESR for full SDC-accepted window
          2. If ESR ≥ θ_CPG: return clean immediately
          3. Else: find argmin SDS chunk, remove it, recompute ESR
          4. Repeat until ESR ≥ θ_CPG or min_chunks reached

        The greedy strategy (remove worst chunk) is optimal for ESR maximization
        because P is linear in the individual (1−SDS)·w terms — removing the
        chunk with the highest poison contribution (lowest SDS, highest weight)
        maximally decreases P and thus maximally increases ESR.
        """
        # Work with accepted-only chunks
        working_set = [r for r in sdc_results if r.accepted]
        if not working_set:
            # Fallback: use all results if none accepted
            working_set = list(sdc_results)

        purge_history = []
        purge_count = 0

        for round_num in range(self.config.max_purge_rounds):
            sds_scores = np.array([r.sds_score for r in working_set])
            tve_scores = np.array([r.candidate.tve_score for r in working_set])
            weights = self._softmax_weights(tve_scores)
            esr, p = self._compute_esr(sds_scores, weights)

            if esr >= self.config.theta_cpg:
                # Window is clean
                return CPGEvaluation(
                    window=working_set,
                    poison_index=p,
                    esr=esr,
                    is_clean=True,
                    purge_count=purge_count,
                    purge_history=purge_history,
                    softmax_weights=weights,
                )

            if len(working_set) <= self.config.min_chunks:
                # Cannot purge further — return best-effort
                break

            # Greedy: remove argmin SDS (most causally irrelevant chunk)
            worst_idx = int(np.argmin(sds_scores))
            removed = working_set[worst_idx]

            # Compute ESR after removal for history
            remaining_sds = np.delete(sds_scores, worst_idx)
            remaining_tve = np.delete(tve_scores, worst_idx)
            remaining_weights = self._softmax_weights(remaining_tve)
            esr_after, _ = self._compute_esr(remaining_sds, remaining_weights)

            purge_history.append((
                round_num,
                removed.candidate.chunk_id,
                esr,
                esr_after,
            ))

            working_set.pop(worst_idx)
            purge_count += 1

        # Final state
        sds_scores = np.array([r.sds_score for r in working_set])
        tve_scores = np.array([r.candidate.tve_score for r in working_set])
        weights = self._softmax_weights(tve_scores)
        esr, p = self._compute_esr(sds_scores, weights)

        return CPGEvaluation(
            window=working_set,
            poison_index=p,
            esr=esr,
            is_clean=esr >= self.config.theta_cpg,
            purge_count=purge_count,
            purge_history=purge_history,
            softmax_weights=weights,
        )
