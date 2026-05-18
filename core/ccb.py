"""
Causal Context Builder (CCB) — Layer 5b of VORTEXRAG

Mathematical Foundation:
  After RFG ranks chunks by Φ̃, the CCB determines the OPTIMAL ORDERING of
  chunks within the context window. Order matters: transformer attention is
  position-sensitive, and the LLM's generation quality depends on whether
  causal prerequisites appear before their dependents in the context.

  ORDERED SLOT INJECTION FORMULA:
    W* = sort_by(Φ̃) ∩ causal_dependency_graph(q)

  SLOT POSITION FORMULA:
    pos(c_i) = rank(Φ̃(c_i)) × causal_depth(c_i)

    where:
      rank(Φ̃(c_i))       — position in Φ̃ ranking (1 = highest phi, ascending)
      causal_depth(c_i)   — depth of chunk in the causal dependency graph
                           (0 = root cause, 1 = immediate effect, 2 = downstream effect)

  WHY THIS FORMULA?
    The product rank × causal_depth balances two competing objectives:
      1. Put high-Φ̃ chunks first (rank factor)
      2. Put root causes before effects (causal_depth factor)

    A chunk with rank=1 and causal_depth=3 (a highly relevant downstream effect)
    gets pos=3. A chunk with rank=3 and causal_depth=0 (a less-ranked root cause)
    gets pos=0. This means the root cause will appear first in the context.

    Without this ordering, the LLM might encounter "Lehman Brothers collapsed"
    before "CDO derivatives were over-leveraged" — generating an answer that
    treats the collapse as unexplained rather than causally derived.

CAUSAL DEPENDENCY GRAPH:
  The graph is built per-query using entity-relation triplets extracted from
  chunks. Nodes = entities, edges = causal relations (causes, leads_to, triggers,
  results_in, prevents, enables). The query entity is the target node; chunks
  are ranked by their distance from the target node (causal_depth = shortest path).

  Chunks NOT in the dependency graph (causal_depth = ∞) are placed at the END
  of the context window (they passed SDC/CPG/RFG but lack an explicit causal
  path — they provide supporting context, not direct causal evidence).

WHY DOES ORDERING MATTER FOR LLMS?
  Experimental finding (Liu et al., 2023): LLMs attend most strongly to text
  at the BEGINNING and END of the context window, with reduced attention to
  the middle. The CCB exploits this by placing root causes and key definitions
  at the beginning (causal_depth=0) and supporting context at the end.
  This is called the "Lost in the Middle" problem — CCB solves it by structural
  placement, not just content filtering.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
from collections import defaultdict

from .rfg import RankedChunk


@dataclass
class CCBConfig:
    """Configuration for Causal Context Builder."""
    max_causal_depth: int = 5    # max depth in causal graph to consider
    fallback_depth: int = 10     # depth assigned to non-graph chunks (placed last)
    max_tokens: int = 4096       # approximate token budget for W*
    avg_tokens_per_chunk: int = 200  # approximate chunk size for budget estimation


class CausalNode(NamedTuple):
    """A node in the causal dependency graph."""
    entity: str
    chunk_ids: list[int]
    depth: int


class OrderedContextSlot(NamedTuple):
    """A chunk placed in an ordered slot in the final context W*."""
    slot_position: float    # pos(c_i) = rank × causal_depth
    causal_depth: int
    phi_rank: int           # 1-indexed rank by Φ̃
    chunk: RankedChunk


class CausalContextBuilder:
    """
    Orders the Φ̃-ranked chunks into a causally coherent context window W*.

    The CCB is the "narrative constructor" of VORTEXRAG — it ensures that
    the final context tells a causally consistent story rather than presenting
    a bag of retrieved facts. This directly improves generation quality because
    transformer language models are trained on coherent text, and coherently
    ordered context more closely matches the training distribution.

    The CCB builds a lightweight causal dependency graph from chunk content,
    assigns depths, and sorts chunks by the pos() formula. Chunks without
    causal graph membership get fallback_depth=10 and appear at the end.

    Usage:
        ccb = CausalContextBuilder()
        ordered_slots = ccb.build(query, ranked_chunks)
        final_context_text = ccb.to_context_string(ordered_slots)
    """

    def __init__(self, config: CCBConfig | None = None):
        self.config = config or CCBConfig()

    def _extract_entities(self, text: str) -> set[str]:
        """
        Extract key entities (nouns, proper nouns) from text.
        In production: uses spaCy NER + noun chunk extraction.
        Here: simple capitalized word extraction as scaffold.
        """
        words = text.split()
        entities = set()
        for w in words:
            clean = w.strip(".,;:!?()")
            if clean and (clean[0].isupper() or len(clean) > 8):
                entities.add(clean.lower())
        return entities

    def _build_causal_graph(
        self,
        query: str,
        chunks: list[RankedChunk],
    ) -> dict[int, int]:
        """
        Build causal depth map: chunk_id → causal_depth.

        Algorithm:
          1. Extract query entities (the "target" entities)
          2. For each chunk, extract entities and causal relations
          3. Build adjacency: query_entities → depth=0
          4. BFS from query entities through causal edges to assign depths
          5. Unconnected chunks get fallback_depth

        In production: uses spaCy dependency parsing + causal verb detection
        to build directed edges (cause → effect). This scaffold uses entity
        overlap as a proxy for causal proximity.
        """
        query_entities = self._extract_entities(query)
        depth_map: dict[int, int] = {}

        # BFS levels: chunks sharing query entities = depth 0
        # Chunks sharing entities with depth-0 chunks = depth 1, etc.
        entity_to_depth: dict[str, int] = {e: 0 for e in query_entities}
        chunk_entities: dict[int, set[str]] = {}

        for chunk in chunks:
            chunk_entities[chunk.chunk_id] = self._extract_entities(chunk.chunk_text)

        # BFS
        for depth in range(self.config.max_causal_depth):
            current_level = {e for e, d in entity_to_depth.items() if d == depth}
            for chunk in chunks:
                if chunk.chunk_id in depth_map:
                    continue
                overlap = chunk_entities[chunk.chunk_id] & current_level
                if overlap:
                    depth_map[chunk.chunk_id] = depth
                    # Propagate chunk entities to next level
                    for e in chunk_entities[chunk.chunk_id]:
                        if e not in entity_to_depth:
                            entity_to_depth[e] = depth + 1

        # Assign fallback depth to chunks not in causal graph
        for chunk in chunks:
            if chunk.chunk_id not in depth_map:
                depth_map[chunk.chunk_id] = self.config.fallback_depth

        return depth_map

    def _slot_position(self, phi_rank: int, causal_depth: int) -> float:
        """
        Compute pos(c_i) = rank(Φ̃(c_i)) × causal_depth(c_i)

        Lower pos = appears earlier in context window.
        Tie-breaking: by phi_rank (lower phi_rank = higher Φ̃ score).
        """
        return phi_rank * causal_depth

    def build(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
    ) -> list[OrderedContextSlot]:
        """
        Build the ordered context window W* from Φ̃-ranked chunks.

        Returns OrderedContextSlot list sorted by slot_position ascending
        (most causally relevant + highest Φ̃ = position 0 = first in context).
        """
        if not ranked_chunks:
            return []

        depth_map = self._build_causal_graph(query, ranked_chunks)

        slots: list[OrderedContextSlot] = []
        for phi_rank, chunk in enumerate(ranked_chunks, start=1):
            depth = depth_map.get(chunk.chunk_id, self.config.fallback_depth)
            pos = self._slot_position(phi_rank, depth)
            slots.append(OrderedContextSlot(
                slot_position=pos,
                causal_depth=depth,
                phi_rank=phi_rank,
                chunk=chunk,
            ))

        # Primary sort: slot_position ascending; secondary: phi_rank ascending
        slots.sort(key=lambda s: (s.slot_position, s.phi_rank))
        return slots

    def to_context_string(
        self,
        slots: list[OrderedContextSlot],
        include_citations: bool = True,
    ) -> str:
        """
        Assemble final context string W* with optional citation markers.

        Citation markers [C1], [C2], ... are injected at the START of each
        chunk to enable the Faithfulness Verifier's ROUGE and NLI checks
        to trace each claim in the generated answer back to its source chunk.
        """
        parts = []
        total_tokens = 0
        for i, slot in enumerate(slots, start=1):
            chunk_tokens = len(slot.chunk.chunk_text.split()) * 1.3  # approx
            if total_tokens + chunk_tokens > self.config.max_tokens:
                break
            prefix = f"[C{i}] " if include_citations else ""
            parts.append(f"{prefix}{slot.chunk.chunk_text}")
            total_tokens += chunk_tokens
        return "\n\n".join(parts)
