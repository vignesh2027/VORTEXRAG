"""
VORTEXRAG — Main Pipeline
Vector Orthogonal Resonance-Tuned EXtraction Retrieval-Augmented Generation

"The only RAG that kills semantic drift and context poisoning simultaneously."

Full 7-Layer Pipeline:
  Layer 0: Corpus preprocessing (chunking + causal graph extraction)
  Layer 1: Query decomposition (intent + sub-queries + entity extraction)
  Layer 2: Tri-Vector Encoding (semantic + syntactic + causal)
  Layer 3: Vortex Retrieval Cone (spiral topology ranking)
  Layer 4: Dual Correction (SDC ∥ CPG — run in parallel)
  Layer 5: Rank Fusion + CCB (Φ-score + causal ordering)
  Layer 6: Grounded Generation + Faithfulness Verification (ΔR loop)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.tve import TriVectorEncoder, TVEConfig, TVEVector
from core.vrc import VortexRetrievalCone, VRCConfig
from core.sdc import SemanticDriftCorrector, SDCConfig
from core.cpg import ContextPoisonGuard, CPGConfig
from core.rfg import RankFusionGate, RFGConfig
from core.ccb import CausalContextBuilder, CCBConfig
from core.fv import FaithfulnessVerifier, FVConfig


@dataclass
class VortexRAGConfig:
    """Master configuration for the VORTEXRAG pipeline."""
    domain: str = "general"
    tve: TVEConfig = field(default_factory=TVEConfig)
    vrc: VRCConfig = field(default_factory=VRCConfig)
    sdc: SDCConfig = field(default_factory=SDCConfig)
    cpg: CPGConfig = field(default_factory=CPGConfig)
    rfg: RFGConfig = field(default_factory=RFGConfig)
    ccb: CCBConfig = field(default_factory=CCBConfig)
    fv:  FVConfig  = field(default_factory=FVConfig)
    chunk_size: int = 512        # tokens per chunk (approx)
    chunk_overlap: int = 64      # overlap between consecutive chunks
    verbose: bool = False

    def __post_init__(self):
        # Propagate domain to domain-aware components
        self.sdc.domain = self.domain


@dataclass
class VortexRAGResult:
    """Complete result from a VORTEXRAG query."""
    query: str
    answer: str
    context_window: list[str]   # ordered chunks in W*
    phi_scores: list[float]     # Φ̃ scores for each chunk
    delta_r: float              # final ΔR hallucination score
    esr: float                  # Effective Signal Ratio of W*
    spiral_pool_size: int       # VRC pool size before SDC/CPG
    purge_count: int            # chunks removed by CPG
    latency_ms: float           # total pipeline latency
    iterations: int             # FV regeneration iterations used
    accepted: bool              # ΔR ≤ δ_FV


class VortexRAG:
    """
    The complete VORTEXRAG pipeline.

    Solves semantic drift and context window poisoning simultaneously via
    a 7-layer pipeline that combines tri-vector encoding, vortex topology
    retrieval, dual-correction (SDC + CPG), phi-score rank fusion, causal
    context ordering, and faithfulness-verified generation.

    Quick start:
        rag = VortexRAG(corpus="your_docs/")
        rag.index()
        answer = rag.query("What caused the 2008 financial crisis?")
        print(answer.answer)

    Custom config:
        config = VortexRAGConfig(domain="legal")
        rag = VortexRAG(corpus="case_files/", config=config)
    """

    def __init__(
        self,
        corpus: str | list[str] | None = None,
        config: VortexRAGConfig | None = None,
        llm_fn: Callable[[str, str], str] | None = None,
    ):
        """
        Args:
            corpus:  Path to document folder, list of texts, or None (call index() later)
            config:  Full pipeline configuration
            llm_fn:  LLM callable: (context_string, query) → answer.
                     If None, uses a stub that returns the top context chunk.
        """
        self.config = config or VortexRAGConfig()
        self.llm_fn = llm_fn or self._default_llm

        # Initialize all pipeline components
        self.encoder  = TriVectorEncoder(self.config.tve)
        self.vrc      = VortexRetrievalCone(self.encoder, self.config.vrc)
        self.sdc      = SemanticDriftCorrector(self.config.sdc)
        self.cpg      = ContextPoisonGuard(self.config.cpg)
        self.rfg      = RankFusionGate(self.config.rfg)
        self.ccb      = CausalContextBuilder(self.config.ccb)
        self.fv       = FaithfulnessVerifier(self.config.fv)

        # Corpus state
        self._corpus_texts: list[str] = []
        self._corpus_vecs: list[TVEVector] = []
        self._indexed: bool = False

        if corpus is not None:
            self._load_corpus(corpus)

    # ──── Layer 0: Corpus Loading & Chunking ────────────────────────────────

    def _load_corpus(self, corpus: str | list[str]):
        if isinstance(corpus, list):
            # Direct text list
            for text in corpus:
                self._corpus_texts.extend(self._chunk_text(text))
        elif isinstance(corpus, str):
            path = Path(corpus)
            if path.is_file():
                self._corpus_texts.extend(self._chunk_text(path.read_text()))
            elif path.is_dir():
                for fpath in sorted(path.rglob("*.txt")) + sorted(path.rglob("*.md")):
                    self._corpus_texts.extend(self._chunk_text(fpath.read_text()))

    def _chunk_text(self, text: str) -> list[str]:
        """
        Semantic boundary chunking with overlap.

        Splits on paragraph boundaries (double newlines), then groups
        paragraphs into chunks of approximately config.chunk_size tokens.
        Overlap of config.chunk_overlap tokens between consecutive chunks
        preserves cross-boundary context.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk_words: list[str] = []
        overlap_buffer: list[str] = []

        for para in paragraphs:
            words = para.split()
            current_chunk_words.extend(words)
            if len(current_chunk_words) >= self.config.chunk_size:
                chunks.append(" ".join(current_chunk_words))
                # Keep last chunk_overlap words as overlap for next chunk
                overlap_buffer = current_chunk_words[-self.config.chunk_overlap:]
                current_chunk_words = list(overlap_buffer)

        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        return chunks if chunks else [text[:2000]]

    # ──── Layer 2: Indexing (TVE encoding of corpus) ────────────────────────

    def index(self, additional_texts: list[str] | None = None) -> "VortexRAG":
        """
        Encode all corpus chunks into TVE vectors.

        This is the O(N) preprocessing step that must be called before query().
        In production, vectors would be persisted to FAISS + NetworkX indexes.

        Returns self for chaining: rag = VortexRAG().index()
        """
        if additional_texts:
            for text in additional_texts:
                self._corpus_texts.extend(self._chunk_text(text))

        if self.config.verbose:
            print(f"[VORTEXRAG] Indexing {len(self._corpus_texts)} chunks...")

        self._corpus_vecs = [self.encoder.encode_chunk(t) for t in self._corpus_texts]
        self._indexed = True

        if self.config.verbose:
            print(f"[VORTEXRAG] Index complete. {len(self._corpus_vecs)} vectors ({self.config.tve.embedding_dim * 3}d each)")

        return self

    # ──── Main Query Pipeline ────────────────────────────────────────────────

    def query(self, query: str) -> VortexRAGResult:
        """
        Run the full 7-layer VORTEXRAG pipeline for a query.

        Raises ValueError if corpus is not indexed yet.
        """
        if not self._indexed:
            raise ValueError("Call .index() before .query(). Example: rag.index().query('...')")

        t0 = time.perf_counter()

        # Layer 1: Query encoding
        query_vec = self.encoder.encode_query(query)
        if self.config.verbose:
            print(f"[TVE] Query encoded to {query_vec.dim}d tri-vector")

        # Layer 3: VRC spiral retrieval
        spiral_pool = self.vrc.retrieve(query_vec, self._corpus_vecs, self._corpus_texts)
        if self.config.verbose:
            print(f"[VRC] Spiral pool: {len(spiral_pool)} candidates")

        # Layer 4a: SDC — Semantic Drift Correction
        sdc_results = self.sdc.filter(query_vec, spiral_pool)
        accepted_sdc = self.sdc.accepted_only(sdc_results)
        if self.config.verbose:
            print(f"[SDC] {len(accepted_sdc)}/{len(spiral_pool)} chunks passed drift gate")

        # If SDC filters too aggressively, fall back to all SDC results
        if len(accepted_sdc) < self.config.cpg.min_chunks:
            accepted_sdc = sdc_results[:max(self.config.cpg.min_chunks, 5)]

        # Layer 4b: CPG — Context Poison Guard
        cpg_eval = self.cpg.evaluate(query_vec, accepted_sdc)
        if self.config.verbose:
            print(f"[CPG] ESR={cpg_eval.esr:.3f} | Purged={cpg_eval.purge_count} | Clean={cpg_eval.is_clean}")

        # Layer 5a: RFG — Rank Fusion Gate (Φ-score)
        ranked = self.rfg.rank(query_vec, cpg_eval)
        final_ranked = self.rfg.select_top_m(ranked)
        if self.config.verbose:
            print(f"[RFG] Top-{len(final_ranked)} by Φ̃ | max_Φ̃={final_ranked[0].phi_norm:.4f}")

        # Layer 5b: CCB — Causal Context Builder
        ordered_slots = self.ccb.build(query, final_ranked)
        context_string = self.ccb.to_context_string(ordered_slots)

        # Layer 6: Generation + Faithfulness Verification
        fv_result = self.fv.verify_with_retry(
            context=context_string,
            generate_fn=lambda ctx, attempt: self.llm_fn(ctx, query),
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        if self.config.verbose:
            print(f"[FV] ΔR={fv_result.delta_r:.4f} | Accepted={fv_result.accepted} | Iter={fv_result.iteration}")
            print(f"[VORTEXRAG] Total latency: {latency_ms:.1f}ms")

        return VortexRAGResult(
            query=query,
            answer=fv_result.answer,
            context_window=[s.chunk.chunk_text for s in ordered_slots],
            phi_scores=[s.chunk.phi_norm for s in ordered_slots],
            delta_r=fv_result.delta_r,
            esr=cpg_eval.esr,
            spiral_pool_size=len(spiral_pool),
            purge_count=cpg_eval.purge_count,
            latency_ms=latency_ms,
            iterations=fv_result.iteration,
            accepted=fv_result.accepted,
        )

    @staticmethod
    def _default_llm(context: str, query: str) -> str:
        """
        Default LLM stub — returns the most relevant context passage.
        Replace with your preferred LLM:
            rag = VortexRAG(llm_fn=lambda ctx, q: openai_client.chat(...))
        """
        if not context:
            return "No relevant context found."
        # Return first paragraph of context as the "answer"
        first_block = context.split("\n\n")[0]
        return f"[Stub answer — wire up your LLM]\n\nBased on context:\n{first_block[:500]}"
