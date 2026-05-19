"""
Causal Context Builder (CCB) — Layer 5b of VORTEXRAG

Mathematical Foundation:
  After RFG ranks chunks by Φ̃, the CCB determines the OPTIMAL ORDERING of
  chunks within the final context window W*. Order matters critically: LLM
  attention is position-sensitive, and generation quality depends on whether
  causal prerequisites appear before their dependents.

  ORDERED SLOT INJECTION FORMULA:
    W* = sort_by(Φ̃) ∩ causal_dependency_graph(q)

  SLOT POSITION FORMULA:
    pos(c_i) = rank(Φ̃(c_i)) × causal_depth(c_i)

    where:
      rank(Φ̃(c_i))       — position in Φ̃ ranking (1 = highest phi, ascending)
      causal_depth(c_i)   — depth of chunk in the causal dependency graph
                           (0 = root cause, 1 = immediate effect, 2+ = downstream)

  WHY THIS FORMULA?
    The product rank × causal_depth balances two competing objectives:
      1. Put high-Φ̃ chunks first (rank factor)
      2. Put root causes before effects (causal_depth factor)

    Example:
      Chunk A: rank=3, causal_depth=0 (root cause, rank 3) → pos = 0
      Chunk B: rank=1, causal_depth=3 (downstream effect, rank 1) → pos = 3

    Chunk A (root cause) appears BEFORE chunk B (effect), even though B has a
    higher Φ̃ rank. This is correct: "CDO derivatives were over-leveraged" must
    appear before "Lehman Brothers filed for bankruptcy" in the context.

    Without this ordering, the LLM encounters effects before causes, generating
    answers that treat causation as unexplained coincidence.

THE "LOST IN THE MIDDLE" PROBLEM (Liu et al., 2023):
  LLMs attend most strongly to text at the BEGINNING and END of context windows.
  Content in the middle receives systematically lower attention, often being
  ignored or poorly integrated. This is the "Lost in the Middle" problem.

  CCB solves this by structural placement:
    - causal_depth=0 chunks (root causes) → pos=0 → placed FIRST in context
    - LLMs attend maximally to root causes, generating causally grounded answers
    - Supporting/downstream chunks are placed later (still attended but with less weight)

  This is better than just ordering by Φ̃ rank because root causes may have
  slightly lower Φ̃ than immediately-relevant effect chunks, but they are
  semantically required for a complete causal explanation.

CAUSAL DEPENDENCY GRAPH CONSTRUCTION:
  The graph is built per-query using:
    1. Entity extraction: key entities from query → depth-0 "root" nodes
    2. Entity overlap BFS: chunks sharing query entities → depth 0
       Chunks sharing entities with depth-0 chunks → depth 1, etc.
    3. Causal verb detection: sentences with CAUSAL_VERB tokens are given
       lower depth (more causal) than associative sentences
    4. Chunks not in the graph → fallback_depth (placed last)

  In production with spaCy: dependency parsing extracts directed
  cause→effect edges for more precise depth assignment.

DEDUPLICATION:
  Before building slots, CCB deduplicates near-identical chunks using
  cosine similarity of semantic TVE vectors. Chunks with similarity >
  dedup_threshold are merged (keep highest Φ̃, discard the rest).
  This prevents the LLM from seeing the same information twice, which
  wastes context budget and may bias generation toward repeated facts.

TOKEN BUDGET:
  CCB enforces a soft token budget via max_tokens. Chunks are added in
  causal order until the estimated token count is exhausted. Estimation:
  word_count × 1.3 (accounts for subword tokenization overhead).
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
from collections import defaultdict

from .rfg import RankedChunk


# ── Causal vocabulary for entity-relation extraction ──────────────────────────
CAUSAL_VERBS = frozenset({
    "cause", "causes", "caused", "causing",
    "lead", "leads", "led", "leading",
    "result", "results", "resulted", "resulting",
    "trigger", "triggers", "triggered", "triggering",
    "produce", "produces", "produced", "producing",
    "generate", "generates", "generated",
    "create", "creates", "created",
    "force", "forces", "forced",
    "prevent", "prevents", "prevented",
    "inhibit", "inhibits", "inhibited",
    "enable", "enables", "enabled",
    "drive", "drives", "drove",
    "induce", "induces", "induced",
    "provoke", "provokes", "provoked",
    "stimulate", "stimulates", "stimulated",
    "suppress", "suppresses", "suppressed",
    "amplify", "amplifies", "amplified",
    "initiate", "initiates", "initiated",
    "propagate", "propagates", "propagated",
    "accelerate", "accelerates", "accelerated",
    "decelerate", "decelerates", "decelerated",
})

# Temporal markers that indicate causal sequence
TEMPORAL_MARKERS = frozenset({
    "because", "therefore", "hence", "thus", "consequently",
    "so", "since", "due", "following", "after", "before",
    "when", "once", "until", "as a result", "which led",
    "this caused", "resulting in", "leading to",
})

# Words that indicate observational / effect (downstream) content
EFFECT_MARKERS = frozenset({
    "affected", "experienced", "observed", "found", "reported",
    "noted", "demonstrated", "showed", "revealed", "indicated",
    "evidenced", "confirmed", "established",
})


@dataclass
class CCBConfig:
    """
    Configuration for Causal Context Builder.

    max_causal_depth: BFS stops at this depth. Chunks requiring deeper traversal
    get fallback_depth and are placed at the end of W*.

    dedup_threshold: cosine similarity threshold above which two chunks are
    considered near-duplicates. 0.90 is conservative (catches only very close
    duplicates); 0.80 is more aggressive.

    max_tokens: soft token budget for the assembled context string.
    Helps control LLM input cost and stays within model context limits.
    """
    max_causal_depth: int = 5       # BFS depth limit for causal graph
    fallback_depth: int = 10        # depth for chunks not in causal graph
    max_tokens: int = 4096          # approximate token budget for W*
    avg_words_per_token: float = 0.75  # word-to-token ratio (approx)
    dedup_threshold: float = 0.92   # cosine sim threshold for deduplication
    enable_dedup: bool = True       # whether to deduplicate before ordering
    causal_depth_bonus: int = 2     # extra priority boost for causal-verb chunks


class CausalNode(NamedTuple):
    """A node in the causal dependency graph."""
    entity:    str
    chunk_ids: list[int]
    depth:     int


class OrderedContextSlot(NamedTuple):
    """
    A chunk placed in an ordered slot in the final context window W*.

    slot_position: pos(c_i) = rank × causal_depth. Lower = appears earlier.
    causal_depth:  depth in the causal dependency graph.
    phi_rank:      1-indexed Φ̃ rank (1 = highest phi_norm score).
    chunk:         the RankedChunk with all quality scores.
    """
    slot_position: float
    causal_depth:  int
    phi_rank:      int
    chunk:         RankedChunk


class CausalContextBuilder:
    """
    Orders the Φ̃-ranked chunks into a causally coherent context window W*.

    The CCB is the "narrative constructor" of VORTEXRAG — it ensures the
    final context tells a causally consistent story rather than presenting
    a bag of retrieved facts. Coherent causal ordering improves generation
    quality because LLMs are trained on coherent text, and causally ordered
    context more closely matches the pre-training distribution.

    Processing pipeline:
      1. Deduplicate near-identical chunks (optional, config.enable_dedup)
      2. Extract entities and causal relations from each chunk
      3. BFS over entity overlap to assign causal depths
      4. Apply slot_position formula: pos = rank × depth
      5. Sort by slot_position ascending (lowest pos = first in context)
      6. Assemble context string within token budget

    Usage:
        ccb = CausalContextBuilder()
        slots = ccb.build("Why did Lehman Brothers collapse?", ranked_chunks)
        context_text = ccb.to_context_string(slots)
        structured = ccb.to_structured_context(slots)
    """

    def __init__(self, config: CCBConfig | None = None):
        self.config = config or CCBConfig()

    # ──── Entity and Causal Feature Extraction ────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Simple lowercase word tokenizer."""
        return re.findall(r'\b[a-zA-Z]\w+\b', text.lower())

    def _extract_entities(self, text: str) -> set[str]:
        """
        Extract key entities from text.

        Strategy (without spaCy):
          1. Capitalized sequences (proper nouns) — length ≥ 3
          2. Long content words (length ≥ 9) — likely domain-specific terms
          3. Exclude common stop words

        In production with spaCy: uses NER + noun chunk extraction for
        significantly higher precision on domain entities.
        """
        stop_words = frozenset({
            "the", "and", "for", "that", "this", "with", "from", "have",
            "they", "were", "are", "was", "been", "has", "had", "not",
            "but", "their", "also", "which", "when", "more", "about",
        })

        words = re.findall(r'\b[A-Za-z]\w+\b', text)
        entities: set[str] = set()
        for w in words:
            lower = w.lower()
            if lower in stop_words:
                continue
            if w[0].isupper() and len(w) >= 3:
                entities.add(lower)
            elif len(w) >= 9:
                entities.add(lower)
        return entities

    def _causal_score(self, text: str) -> float:
        """
        Compute a causal verb density score for a chunk.

        Higher score = more causal verbs = chunk is more likely to be
        a root cause description rather than an observable effect description.

        Returns a score ∈ [0, 1] where 1 = very causally dense text.
        """
        words = self._tokenize(text)
        if not words:
            return 0.0
        causal_count = sum(1 for w in words if w in CAUSAL_VERBS)
        temporal_count = sum(1 for w in words if w in TEMPORAL_MARKERS)
        effect_count = sum(1 for w in words if w in EFFECT_MARKERS)
        # Causal and temporal markers push depth down; effect markers push up
        raw = (causal_count + temporal_count * 0.5 - effect_count * 0.3) / len(words)
        return float(np.clip(raw * 10, 0.0, 1.0))

    def _build_causal_graph(
        self,
        query: str,
        chunks: list[RankedChunk],
    ) -> dict[int, int]:
        """
        Build a causal depth map: chunk_id → causal_depth.

        Algorithm:
          1. Extract query entities → seed depth-0 entity set
          2. BFS: chunks overlapping with depth-k entities → depth k
          3. Propagate chunk entities to next BFS level
          4. Apply causal verb bonus: if chunk has high causal density,
             reduce its depth by causal_depth_bonus (makes it appear earlier)
          5. Assign fallback_depth to unconnected chunks

        Entity overlap is used as a causal proximity proxy — entities shared
        with the query are more causally relevant than entities two hops away.
        """
        query_entities = self._extract_entities(query)
        if not query_entities:
            return {chunk.chunk_id: self.config.fallback_depth for chunk in chunks}

        depth_map: dict[int, int] = {}
        entity_to_depth: dict[str, int] = {e: 0 for e in query_entities}
        chunk_entities: dict[int, set[str]] = {}
        chunk_causal_scores: dict[int, float] = {}

        for chunk in chunks:
            chunk_entities[chunk.chunk_id] = self._extract_entities(chunk.chunk_text)
            chunk_causal_scores[chunk.chunk_id] = self._causal_score(chunk.chunk_text)

        # BFS over entity overlap
        for depth in range(self.config.max_causal_depth):
            current_level = {e for e, d in entity_to_depth.items() if d == depth}
            if not current_level:
                break

            for chunk in chunks:
                if chunk.chunk_id in depth_map:
                    continue
                overlap = chunk_entities[chunk.chunk_id] & current_level
                if overlap:
                    # Base depth from BFS level
                    base_depth = depth
                    # Causal verb bonus: high causal density → lower depth (appears earlier)
                    causal_bonus = int(
                        chunk_causal_scores[chunk.chunk_id] > 0.3
                    ) * self.config.causal_depth_bonus
                    assigned_depth = max(0, base_depth - causal_bonus)
                    depth_map[chunk.chunk_id] = assigned_depth

                    # Propagate chunk's entities to enable next BFS level
                    for e in chunk_entities[chunk.chunk_id]:
                        if e not in entity_to_depth:
                            entity_to_depth[e] = depth + 1

        # Assign fallback to disconnected chunks
        for chunk in chunks:
            if chunk.chunk_id not in depth_map:
                depth_map[chunk.chunk_id] = self.config.fallback_depth

        return depth_map

    # ──── Deduplication ───────────────────────────────────────────────────────

    def deduplicate(
        self,
        ranked_chunks: list[RankedChunk],
        threshold: float | None = None,
    ) -> list[RankedChunk]:
        """
        Remove near-duplicate chunks using cosine similarity of semantic vectors.

        For each pair of chunks with cosine similarity > threshold, keep the
        one with the higher Φ̃ score and discard the other.

        The deduplication order matters: chunks are processed in Φ̃ rank order
        (highest first), so the most relevant representative is always kept.

        Returns the deduplicated list, still sorted by phi_norm descending.
        """
        thresh = threshold if threshold is not None else self.config.dedup_threshold
        if len(ranked_chunks) <= 1:
            return ranked_chunks

        kept: list[RankedChunk] = []
        kept_vecs: list[np.ndarray] = []

        for chunk in ranked_chunks:  # already sorted by phi_norm desc
            sem_vec = chunk.sdc_result.candidate.tve_vec.semantic

            if not kept_vecs:
                kept.append(chunk)
                kept_vecs.append(sem_vec)
                continue

            # Check similarity to all kept chunks
            kept_matrix = np.stack(kept_vecs)        # (n_kept, d)
            sims = kept_matrix @ sem_vec              # (n_kept,)
            max_sim = float(np.max(sims))

            if max_sim < thresh:
                kept.append(chunk)
                kept_vecs.append(sem_vec)
            # else: near-duplicate → discard (lower Φ̃ chunk)

        return kept

    # ──── Slot Assignment and Ordering ────────────────────────────────────────

    def _slot_position(self, phi_rank: int, causal_depth: int) -> float:
        """
        Compute pos(c_i) = rank(Φ̃(c_i)) × causal_depth(c_i).

        Lower pos = appears earlier in context window.

        Special cases:
          causal_depth=0: pos=0 for all ranks (root causes always first)
          fallback_depth: pos is very large (always last)
          phi_rank=1, causal_depth=1: pos=1 (second most important)

        Tie-breaking: chunks with same slot_position are broken by phi_rank
        (lower phi_rank = higher Φ̃ = better chunk).
        """
        return float(phi_rank * causal_depth)

    def build(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
    ) -> list[OrderedContextSlot]:
        """
        Build the ordered context window W* from Φ̃-ranked chunks.

        Steps:
          1. Optionally deduplicate (config.enable_dedup)
          2. Build causal depth map via entity-overlap BFS
          3. Assign slot positions: pos = rank × causal_depth
          4. Sort by (slot_position, phi_rank) ascending
          5. Return ordered list of OrderedContextSlot

        Returns OrderedContextSlot list sorted by (slot_position, phi_rank).
        Chunks with slot_position=0 (root causes) appear first.
        """
        if not ranked_chunks:
            return []

        # Deduplication
        chunks = self.deduplicate(ranked_chunks) if self.config.enable_dedup else ranked_chunks

        # Causal graph
        depth_map = self._build_causal_graph(query, chunks)

        # Slot assignment
        slots: list[OrderedContextSlot] = []
        for phi_rank, chunk in enumerate(chunks, start=1):
            depth = depth_map.get(chunk.chunk_id, self.config.fallback_depth)
            pos = self._slot_position(phi_rank, depth)
            slots.append(OrderedContextSlot(
                slot_position=pos,
                causal_depth=depth,
                phi_rank=phi_rank,
                chunk=chunk,
            ))

        # Sort: primary = slot_position asc, secondary = phi_rank asc
        slots.sort(key=lambda s: (s.slot_position, s.phi_rank))
        return slots

    # ──── Context Assembly ────────────────────────────────────────────────────

    def to_context_string(
        self,
        slots: list[OrderedContextSlot],
        include_citations: bool = True,
        include_metadata: bool = False,
    ) -> str:
        """
        Assemble the final context string W* from ordered slots.

        Citation markers [C1], [C2], ... are injected at the start of each
        chunk to enable the Faithfulness Verifier (FV) to trace generated
        claims back to their source chunks via ROUGE-L matching.

        include_metadata: if True, prepends causal depth and Φ̃ score to each
        chunk for debugging (not recommended for production use with LLMs).

        Token budget is enforced using word count × (1/avg_words_per_token)
        as a proxy for subword token count (approximately correct for English).
        """
        parts: list[str] = []
        estimated_tokens = 0
        token_limit = self.config.max_tokens

        for i, slot in enumerate(slots, start=1):
            chunk_text = slot.chunk.chunk_text.strip()
            word_count = len(chunk_text.split())
            chunk_tokens = int(word_count / self.config.avg_words_per_token)

            if estimated_tokens + chunk_tokens > token_limit:
                break

            components = []
            if include_citations:
                components.append(f"[C{i}]")
            if include_metadata:
                components.append(
                    f"[depth={slot.causal_depth} Φ̃={slot.chunk.phi_norm:.3f}]"
                )
            components.append(chunk_text)
            parts.append(" ".join(components))
            estimated_tokens += chunk_tokens

        return "\n\n".join(parts)

    def to_structured_context(
        self,
        slots: list[OrderedContextSlot],
    ) -> list[dict]:
        """
        Return the ordered context as a list of structured dicts.

        Each dict contains:
          {
            'citation':       str   — [C1], [C2], etc.
            'slot_position':  float — pos(c_i) = rank × depth
            'causal_depth':   int   — depth in causal graph
            'phi_rank':       int   — rank by Φ̃ score
            'phi_norm':       float — normalized Φ̃ score
            'tve_score':      float — tri-vector relevance
            'sds_score':      float — semantic drift alignment
            'esr_contrib':    float — ESR signal contribution
            'chunk_text':     str   — full chunk text
            'word_count':     int   — approximate length
            'causal_label':   str   — 'root_cause' | 'effect' | 'supporting' | 'fallback'
          }

        Useful for structured logging, debugging, and UI rendering.
        """
        structured = []
        for i, slot in enumerate(slots, start=1):
            depth = slot.causal_depth
            if depth == 0:
                label = "root_cause"
            elif depth <= 2:
                label = "effect"
            elif depth < self.config.fallback_depth:
                label = "supporting"
            else:
                label = "fallback"

            structured.append({
                "citation":      f"[C{i}]",
                "slot_position": round(slot.slot_position, 2),
                "causal_depth":  depth,
                "phi_rank":      slot.phi_rank,
                "phi_norm":      round(slot.chunk.phi_norm, 4),
                "tve_score":     round(slot.chunk.tve_score, 4),
                "sds_score":     round(slot.chunk.sds_score, 4),
                "esr_contrib":   round(slot.chunk.esr_contribution, 4),
                "chunk_text":    slot.chunk.chunk_text,
                "word_count":    len(slot.chunk.chunk_text.split()),
                "causal_label":  label,
            })
        return structured

    # ──── Analysis and Interpretability ───────────────────────────────────────

    def explain_ordering(self, slots: list[OrderedContextSlot]) -> list[str]:
        """
        Human-readable explanation of why each chunk was placed at its position.

        Returns a list of strings, one per slot, explaining the slot_position
        formula result and what it means for LLM attention.
        """
        explanations = []
        for i, slot in enumerate(slots, start=1):
            depth = slot.causal_depth
            phi_rank = slot.phi_rank
            pos = slot.slot_position

            if depth == 0:
                role = "ROOT CAUSE — placed first for maximum LLM attention"
            elif depth == 1:
                role = "IMMEDIATE EFFECT — placed early, causally derived from root"
            elif depth < self.config.fallback_depth:
                role = f"SUPPORTING CONTEXT (depth={depth}) — downstream context"
            else:
                role = "FALLBACK — not in causal graph, placed last as background"

            explanations.append(
                f"[C{i}] pos={pos:.0f} = rank({phi_rank}) × depth({depth}) | "
                f"Φ̃={slot.chunk.phi_norm:.3f} | {role}"
            )
        return explanations

    def causal_chain_summary(self, slots: list[OrderedContextSlot]) -> str:
        """
        Generate a narrative description of the causal chain in W*.

        Groups chunks by causal depth and describes the flow:
          Root causes (depth=0) → Effects (depth=1–2) → Background (depth 3+)
        """
        depth_groups: dict[int, list[OrderedContextSlot]] = defaultdict(list)
        for slot in slots:
            depth_groups[slot.causal_depth].append(slot)

        lines = ["Causal Chain in W*:"]
        for depth in sorted(depth_groups.keys()):
            group = depth_groups[depth]
            if depth == 0:
                label = "🔴 Root Causes"
            elif depth <= 2:
                label = f"🟡 Effects (depth={depth})"
            elif depth < self.config.fallback_depth:
                label = f"🟢 Supporting Context (depth={depth})"
            else:
                label = "⚪ Background / Fallback"

            lines.append(f"\n{label}:")
            for slot in group:
                preview = slot.chunk.chunk_text[:90].replace("\n", " ")
                lines.append(f"  [C{slot.phi_rank}] Φ̃={slot.chunk.phi_norm:.3f} → {preview}...")

        return "\n".join(lines)

    def depth_distribution(self, slots: list[OrderedContextSlot]) -> dict[int, int]:
        """Count how many chunks are at each causal depth level."""
        dist: dict[int, int] = {}
        for slot in slots:
            dist[slot.causal_depth] = dist.get(slot.causal_depth, 0) + 1
        return dict(sorted(dist.items()))

    def token_budget_usage(self, slots: list[OrderedContextSlot]) -> dict:
        """
        Report estimated token usage for the assembled context.

        Returns {total_words, estimated_tokens, budget, utilization_pct}.
        Helps tune max_tokens to stay within LLM context limits.
        """
        total_words = sum(len(s.chunk.chunk_text.split()) for s in slots)
        estimated_tokens = int(total_words / self.config.avg_words_per_token)
        return {
            "n_slots":          len(slots),
            "total_words":      total_words,
            "estimated_tokens": estimated_tokens,
            "token_budget":     self.config.max_tokens,
            "utilization_pct":  round(100 * estimated_tokens / max(self.config.max_tokens, 1), 1),
        }
