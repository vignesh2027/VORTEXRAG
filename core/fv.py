"""
Faithfulness Verifier (FV) — Layer 6b of VORTEXRAG

Mathematical Foundation:
  After generation, the FV verifies that the LLM's answer is GROUNDED in the
  context W* and has not hallucinated. It combines two complementary signals:

  DELTA-R SCORE (Hallucination Score):
    ΔR(answer, W*) = 1 − ROUGE_overlap(answer, W*) · NLI_entailment(answer, W*)

    where:
      ROUGE_overlap(a, W*)    — lexical overlap: measures whether the answer
                                contains words/phrases that actually appear in W*
                                (uses ROUGE-L for longest common subsequence)
      NLI_entailment(a, W*)   — semantic entailment: measures whether W* logically
                                entails the answer via a natural language inference
                                model (e.g., DeBERTa-v3 trained on NLI tasks)
      Product of both:        — a high ΔR requires BOTH conditions:
                                1. Lexical presence in W* (not fabricated words)
                                2. Logical entailment from W* (not contradicted)

  ACCEPTANCE CONDITION:
    Answer is ACCEPTED iff ΔR(answer, W*) ≤ δ_FV   (default: 0.15)

  REGENERATION LOOP:
    if ΔR > δ_FV:
        re-rank context by Φ̃ (different top-m selection)
        regenerate (max 3 iterations)

WHY MULTIPLY ROUGE × NLI (not add)?
  Adding would allow a hallucinated answer with high ROUGE (copies words from W*)
  but contradicts W*'s meaning to pass. Multiplication enforces:
    - High ROUGE alone (0.9 × 0.1 = 0.09 → ΔR = 0.91 → REJECTED if copies
      words but contradicts logic)
    - High NLI alone  (0.1 × 0.9 = 0.09 → REJECTED if entailed but uses
      different vocabulary, which suggests paraphrasing unsupported claims)
    - Both high       (0.8 × 0.85 = 0.68 → ΔR = 0.32 → borderline)
    - Both very high  (0.95 × 0.92 = 0.874 → ΔR = 0.126 → ACCEPTED ✓)

  The product creates a joint faithfulness criterion where both lexical
  fidelity and logical grounding must simultaneously hold.

WHY ΔR = 1 − product (not just product)?
  ΔR is a HALLUCINATION SCORE (higher = more hallucinated). A perfectly
  grounded answer has ΔR→0. The threshold δ_FV=0.15 means: "the answer
  is acceptable if at most 15% of its content is unverifiable."

WHY MAX 3 ITERATIONS?
  Empirically, if an answer doesn't pass after 3 regenerations with different
  context configurations, the problem is in retrieval, not generation. Further
  iterations either converge on the same answer or produce increasingly degraded
  outputs. The 3-iteration limit caps latency while catching the most common
  hallucination patterns (typically fixed in the first re-generation).

WHY ROUGE-L SPECIFICALLY?
  ROUGE-L uses longest common subsequence (LCS) rather than unigram or bigram
  overlap. This makes it robust to paraphrasing (different word order but same
  content) while still penalizing invented facts. ROUGE-1/2 would penalize
  legitimate paraphrases; ROUGE-L correctly identifies them as faithful.
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .ccb import OrderedContextSlot


@dataclass
class FVConfig:
    """Configuration for Faithfulness Verifier."""
    delta_fv: float = 0.15       # acceptance threshold (ΔR ≤ δ_FV)
    max_iterations: int = 3      # max regeneration attempts
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    use_nli: bool = False        # set True when NLI model is available


class FVResult(NamedTuple):
    """Result of faithfulness verification for a single answer."""
    answer: str
    delta_r: float          # ΔR hallucination score ∈ [0, 1]
    rouge_score: float      # ROUGE-L overlap ∈ [0, 1]
    nli_score: float        # NLI entailment score ∈ [0, 1]
    accepted: bool          # ΔR ≤ δ_FV
    iteration: int          # which iteration produced this answer


class FaithfulnessVerifier:
    """
    Post-generation hallucination detector and regeneration controller.

    The FV is the final quality gate in VORTEXRAG. It measures whether the
    LLM's answer is grounded in the retrieved context W* using a dual-signal
    metric: ROUGE-L for lexical grounding and NLI entailment for logical
    grounding. Both signals must be high for the answer to be accepted.

    The FV closes the VORTEXRAG feedback loop: if an answer fails, it
    triggers re-ranking (different Φ̃ selection) and regeneration, giving
    the system up to 3 attempts to produce a faithful answer. This is the
    "last line of defense" against hallucination after all retrieval
    purification has been applied.

    In v0.1, ROUGE-L is implemented from scratch. NLI requires an optional
    cross-encoder dependency (set use_nli=True when available).

    Usage:
        fv = FaithfulnessVerifier()
        result = fv.verify(answer, context_string)
        if not result.accepted:
            # trigger re-generation
    """

    def __init__(self, config: FVConfig | None = None):
        self.config = config or FVConfig()
        self._nli_model = None
        if self.config.use_nli:
            self._load_nli()

    def _load_nli(self):
        try:
            from sentence_transformers import CrossEncoder
            self._nli_model = CrossEncoder(self.config.nli_model)
        except ImportError:
            pass

    # ──── ROUGE-L (from scratch) ────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r'\b\w+\b', text.lower())

    def _lcs_length(self, a: list[str], b: list[str]) -> int:
        """Compute length of Longest Common Subsequence via DP."""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        # Space-optimized DP (O(min(m,n)) space)
        if m < n:
            a, b = b, a
            m, n = n, m
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(curr[j-1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def rouge_l(self, hypothesis: str, reference: str) -> float:
        """
        Compute ROUGE-L (F1) between hypothesis (answer) and reference (context).

        ROUGE-L F1 = (2 · P · R) / (P + R + ε)
          P = LCS_length / |hypothesis|   (precision: fraction of answer in context)
          R = LCS_length / |reference|    (recall: fraction of context covered)
        """
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)
        if not hyp_tokens or not ref_tokens:
            return 0.0
        lcs = self._lcs_length(hyp_tokens, ref_tokens)
        p = lcs / len(hyp_tokens)
        r = lcs / len(ref_tokens)
        if p + r < 1e-10:
            return 0.0
        return 2 * p * r / (p + r)

    # ──── NLI Entailment ────────────────────────────────────────────────────

    def nli_entailment(self, hypothesis: str, premise: str) -> float:
        """
        Compute NLI entailment score: P(premise entails hypothesis).

        Uses a cross-encoder NLI model (DeBERTa-v3) if available.
        Falls back to a ROUGE-L approximation if not available.

        The entailment direction matters: we check whether the CONTEXT (premise)
        entails the ANSWER (hypothesis) — not the other way around.
        """
        if self._nli_model is not None:
            scores = self._nli_model.predict([(premise, hypothesis)])
            # DeBERTa NLI labels: [contradiction, neutral, entailment]
            probs = np.exp(scores[0]) / np.exp(scores[0]).sum()
            return float(probs[2])  # entailment probability
        # Fallback: use ROUGE-L as weak entailment proxy
        return self.rouge_l(hypothesis, premise)

    # ──── Delta-R (Hallucination Score) ─────────────────────────────────────

    def delta_r(self, answer: str, context: str) -> tuple[float, float, float]:
        """
        Compute ΔR(answer, W*) = 1 − ROUGE_overlap · NLI_entailment

        Returns (delta_r, rouge_score, nli_score).

        The product ROUGE·NLI is the GROUNDING SCORE (1−ΔR):
          Grounding → 1.0: perfectly grounded answer
          Grounding → 0.0: completely hallucinated answer
        ΔR = 1 − Grounding: hallucination score.
        """
        rouge = self.rouge_l(answer, context)
        nli   = self.nli_entailment(answer, context)
        grounding = rouge * nli
        return 1.0 - grounding, rouge, nli

    def verify(
        self,
        answer: str,
        context: str,
        iteration: int = 1,
    ) -> FVResult:
        """Verify a single answer against its context."""
        dr, rouge, nli = self.delta_r(answer, context)
        return FVResult(
            answer=answer,
            delta_r=dr,
            rouge_score=rouge,
            nli_score=nli,
            accepted=dr <= self.config.delta_fv,
            iteration=iteration,
        )

    def verify_with_retry(
        self,
        context: str,
        generate_fn,
    ) -> FVResult:
        """
        Verify with up to max_iterations regeneration attempts.

        generate_fn: callable(context, attempt) → str
          Called with the same context each time (re-ranking is handled upstream).
          The attempt number allows generate_fn to vary temperature or sampling.

        Returns the best result by ΔR (lowest hallucination score).
        """
        best_result = None
        for attempt in range(1, self.config.max_iterations + 1):
            answer = generate_fn(context, attempt)
            result = self.verify(answer, context, iteration=attempt)
            if best_result is None or result.delta_r < best_result.delta_r:
                best_result = result
            if result.accepted:
                return result
        return best_result
