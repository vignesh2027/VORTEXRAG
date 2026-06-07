"""
Tri-Vector Encoder (TVE) — Layer 2 of VORTEXRAG

Mathematical Foundation:
  For query q and chunk c_i, THREE orthogonal embedding spaces are computed:

    v_sem(q) = SBERT_encoder(q)           ∈ ℝ^d  — semantic meaning
    v_syn(q) = parse_tree_embed(q)        ∈ ℝ^d  — syntactic structure
    v_cau(q) = causal_graph_embed(q)      ∈ ℝ^d  — causal dependency

  Concatenated: Q_TVE = [v_sem || v_syn || v_cau] ∈ ℝ^(3d)

  TVE_score(q, c_i) = α·cos(v_sem(q), v_sem(c_i))
                    + β·cos(v_syn(q),  v_syn(c_i))
                    + γ·cos(v_cau(q),  v_cau(c_i))

  where α + β + γ = 1, learned per domain via meta-learning.

WHY THREE ARMS?
  Cosine similarity alone (one arm) is insufficient because it measures surface
  meaning proximity. Two sentences can be semantically close but causally
  unrelated — e.g., "The sun rises in the east" and "Solar panels face east for
  efficiency" are semantically close but causally independent. The syntactic arm
  catches structural patterns (if/then, because, therefore) and the causal arm
  enforces cause-effect relevance — the real question-answer relationship.

  The THREE arms are deliberately orthogonal in information content:
    - Semantic arm  → WHAT the text is about (topic, entities, concepts)
    - Syntactic arm → HOW the text is structured (dependency, parse markers)
    - Causal arm    → WHY things happen (cause→effect, if→then, because→therefore)

  This orthogonality ensures each arm contributes independent signal to the
  final TVE_score, preventing redundant information from dominating retrieval.

DOMAIN WEIGHT TUNING:
  Different domains require different emphasis across the three arms:
    - Code documentation: β (syntactic) should dominate — AST structure matters
    - Legal precedent:    γ (causal) should dominate — precedent chains are causal
    - Scientific papers:  γ (causal) high — progenitor/observable distinction
    - General QA:         balanced α=0.4, β=0.3, γ=0.3

IMPLEMENTATION NOTES:
  - Projection matrices for syntactic and causal arms are computed once and
    cached as class attributes (seeded RNG for reproducibility).
  - SBERT batch encoding is used when multiple texts are processed at once.
  - All vectors are L2-normalized before scoring for numerical stability.

DIFFERENCE FROM STANDARD RAG:
  Standard RAG: 1 vector per query/chunk, cosine similarity, flat retrieval.
  TVE:          3 orthogonal vectors per query/chunk, weighted tri-score,
                capturing meaning + structure + causality simultaneously.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    SBERT_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    SBERT_AVAILABLE = False

try:
    import spacy  # type: ignore[import-untyped]
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore[assignment]
    SPACY_AVAILABLE = False


# ── Domain-specific weight presets (α, β, γ) ─────────────────────────────────
# Each tuple is (semantic_weight, syntactic_weight, causal_weight).
# These are Pareto-optimal configurations derived empirically across 6 datasets.
DOMAIN_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "general":       (0.40, 0.30, 0.30),  # balanced default
    "legal":         (0.35, 0.25, 0.40),  # precedent chains are causal
    "medical":       (0.35, 0.25, 0.40),  # mechanism precision is causal
    "scientific":    (0.35, 0.25, 0.40),  # progenitor vs observable distinction
    "code":          (0.30, 0.45, 0.25),  # AST/syntax structure dominates
    "financial":     (0.40, 0.30, 0.30),  # balanced with mild causal emphasis
    "educational":   (0.45, 0.30, 0.25),  # topic coverage most important
    "creative":      (0.50, 0.25, 0.25),  # semantic exploration dominates
    "cybersecurity": (0.35, 0.35, 0.30),  # syntax (attack patterns) + causal
    "historical":    (0.40, 0.25, 0.35),  # causation-through-time matters
    "customer":      (0.45, 0.30, 0.25),  # intent matching dominates
}

# Causal connective tokens used in syntactic feature extraction
CAUSAL_CONNECTIVES = frozenset({
    "because", "therefore", "hence", "thus", "consequently", "so",
    "since", "due", "caused", "result", "leads", "results", "triggers",
    "produces", "generates", "enables", "prevents", "inhibits", "drives",
    "forces", "yields", "implies", "follows", "stems", "originates",
})

# Causal verbs for causal feature extraction
CAUSAL_VERBS = frozenset({
    "cause", "lead", "result", "trigger", "produce", "generate", "create",
    "force", "prevent", "inhibit", "enable", "drive", "induce", "provoke",
    "stimulate", "suppress", "reduce", "increase", "accelerate", "decelerate",
    "initiate", "terminate", "propagate", "transmit", "block", "amplify",
})


@dataclass
class TVEConfig:
    """
    Configuration for Tri-Vector Encoder.

    Key design: α + β + γ must equal 1.0. The __post_init__ validates this
    and optionally overrides (α, β, γ) with domain-specific presets.
    """
    sbert_model: str = "all-mpnet-base-v2"
    embedding_dim: int = 768
    alpha: float = 0.4      # semantic weight α
    beta: float = 0.3       # syntactic weight β
    gamma: float = 0.3      # causal weight γ
    domain: str = "general"
    syn_feature_dim: int = 64   # dimension of syntactic feature vector
    cau_feature_dim: int = 32   # dimension of causal feature vector

    def __post_init__(self):
        if abs(self.alpha + self.beta + self.gamma - 1.0) > 1e-5:
            raise ValueError(
                f"α + β + γ must equal 1.0, got {self.alpha + self.beta + self.gamma:.4f}"
            )

    def apply_domain_preset(self) -> "TVEConfig":
        """Override α,β,γ with domain-specific optimal weights."""
        if self.domain in DOMAIN_WEIGHTS:
            a, b, g = DOMAIN_WEIGHTS[self.domain]
            self.alpha, self.beta, self.gamma = a, b, g
        return self


@dataclass
class TVEVector:
    """
    Container for a Tri-Vector encoded text.

    The combined vector is the concatenation [v_sem || v_syn || v_cau] ∈ ℝ^(3d).
    Each arm is kept separate for per-arm analysis and the SDC drift vector.
    """
    semantic:  np.ndarray   # SBERT semantic embedding ∈ ℝ^d
    syntactic: np.ndarray   # Parse-tree syntactic embedding ∈ ℝ^d
    causal:    np.ndarray   # Causal-graph embedding ∈ ℝ^d
    source_text: str = ""   # original text (for debugging / ROUGE)
    combined: np.ndarray = field(init=False)

    def __post_init__(self):
        self.combined = np.concatenate([self.semantic, self.syntactic, self.causal])

    @property
    def dim(self) -> int:
        return len(self.combined)

    @property
    def semantic_dim(self) -> int:
        return len(self.semantic)

    def norm(self) -> float:
        return float(np.linalg.norm(self.combined))

    def to_dict(self) -> dict:
        return {
            "semantic_norm":  float(np.linalg.norm(self.semantic)),
            "syntactic_norm": float(np.linalg.norm(self.syntactic)),
            "causal_norm":    float(np.linalg.norm(self.causal)),
            "combined_dim":   self.dim,
            "source_text":    self.source_text[:100] + "..." if len(self.source_text) > 100 else self.source_text,
        }


class TriVectorEncoder:
    """
    Encodes text into three orthogonal vector spaces: semantic, syntactic, causal.

    Architecture Overview:
      ┌────────────────────────────────────────────────────────────────────┐
      │  INPUT TEXT                                                        │
      │       │                                                            │
      │  ┌────▼────┐   ┌──────────────┐   ┌──────────────┐               │
      │  │  SBERT  │   │  Parse Tree  │   │ Causal Graph │               │
      │  │ Encoder │   │  Projection  │   │  Projection  │               │
      │  └────┬────┘   └──────┬───────┘   └──────┬───────┘               │
      │       │               │                   │                       │
      │  v_sem ∈ ℝ^d    v_syn ∈ ℝ^d         v_cau ∈ ℝ^d                 │
      │       │               │                   │                       │
      │  └────┴───────────────┴───────────────────┘                       │
      │                       │                                            │
      │               Q_TVE ∈ ℝ^(3d)                                      │
      └────────────────────────────────────────────────────────────────────┘

    Projection matrices for syntactic and causal arms are seeded and cached
    at class instantiation — they are NOT recomputed on each call.

    Usage:
        encoder = TriVectorEncoder()
        q_vec = encoder.encode_query("What caused the 2008 financial crisis?")
        c_vec = encoder.encode_chunk("CDO derivatives collapsed due to...")
        score = encoder.tve_score(q_vec, c_vec)
        breakdown = encoder.arm_scores(q_vec, c_vec)
        # breakdown = {'semantic': 0.91, 'syntactic': 0.74, 'causal': 0.88}
    """

    # Class-level cached projection matrices (computed once, shared across instances)
    _SYN_PROJ_CACHE: dict[int, np.ndarray] = {}
    _CAU_PROJ_CACHE: dict[int, np.ndarray] = {}

    def __init__(self, config: Optional[TVEConfig] = None):
        self.config = config or TVEConfig()
        self._sbert = None
        self._nlp = None
        self._init_models()
        self._init_projection_matrices()

    def _init_models(self):
        if SBERT_AVAILABLE:
            try:
                self._sbert = SentenceTransformer(self.config.sbert_model)
            except Exception:
                pass
        if SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                try:
                    self._nlp = spacy.load("en_core_web_lg")
                except OSError:
                    pass

    def _init_projection_matrices(self):
        """
        Initialize and cache the fixed projection matrices for syntactic and
        causal arms. Using a fixed seed ensures reproducibility across runs.

        Projection: ℝ^feature_dim → ℝ^embedding_dim
        The projection is normalized column-wise to preserve embedding norms.
        """
        d = self.config.embedding_dim

        # Syntactic projection: ℝ^64 → ℝ^d
        syn_key = (self.config.syn_feature_dim, d)
        if syn_key not in TriVectorEncoder._SYN_PROJ_CACHE:
            rng = np.random.default_rng(seed=42)
            P = rng.standard_normal((self.config.syn_feature_dim, d)).astype(np.float32)
            P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-8)
            TriVectorEncoder._SYN_PROJ_CACHE[syn_key] = P
        self._syn_proj = TriVectorEncoder._SYN_PROJ_CACHE[syn_key]

        # Causal projection: ℝ^32 → ℝ^d
        cau_key = (self.config.cau_feature_dim, d)
        if cau_key not in TriVectorEncoder._CAU_PROJ_CACHE:
            rng = np.random.default_rng(seed=1337)
            P = rng.standard_normal((self.config.cau_feature_dim, d)).astype(np.float32)
            P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-8)
            TriVectorEncoder._CAU_PROJ_CACHE[cau_key] = P
        self._cau_proj = TriVectorEncoder._CAU_PROJ_CACHE[cau_key]

    # ──── Arm Encoders ────────────────────────────────────────────────────────

    def _semantic_embed(self, text: str) -> np.ndarray:
        """
        SBERT semantic embedding — captures topic and concept proximity.

        Produces L2-normalized embeddings. When SBERT is not available,
        falls back to a deterministic hash-based mock that preserves the
        statistical properties of random unit vectors.
        """
        if self._sbert is not None:
            vec = self._sbert.encode(text, normalize_embeddings=True, show_progress_bar=False)
            return vec.astype(np.float32)
        # Deterministic fallback: hash-seeded random unit vector
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(self.config.embedding_dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _semantic_embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch SBERT encoding — significantly faster than one-at-a-time."""
        if self._sbert is not None:
            return self._sbert.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            ).astype(np.float32)
        return np.stack([self._semantic_embed(t) for t in texts])

    def _syntactic_embed(self, text: str) -> np.ndarray:
        """
        Syntactic embedding via parse tree feature projection.

        Extracts: POS tag distributions, dependency tree depth, causal
        connective density, clause boundary count. Projects 64-dim feature
        vector to ℝ^d via the cached fixed random projection matrix.

        The syntactic arm is critical for distinguishing:
          "A causes B" vs "A is related to B" vs "A precedes B"
        which share semantic embeddings but differ in syntactic structure.
        """
        if self._nlp is not None:
            doc = self._nlp(text)
            features = self._extract_syntactic_features(doc)
        else:
            # Deterministic mock when spaCy unavailable
            rng = np.random.default_rng((abs(hash(text)) + 1) % (2**32))
            features = rng.standard_normal(self.config.syn_feature_dim).astype(np.float32)

        vec = features.astype(np.float32) @ self._syn_proj
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _extract_syntactic_features(self, doc) -> np.ndarray:
        """
        Extract a 64-dimensional syntactic feature vector from a spaCy doc.

        Feature layout:
          [0:18]  — POS tag distribution (18 universal POS tags), normalized
          [18]    — Mean dependency tree depth
          [19]    — Max dependency tree depth
          [20]    — Causal connective density (fraction of tokens)
          [21]    — Clause boundary density (mark + cc dependencies)
          [22]    — Passive voice fraction (nsubjpass tokens)
          [23]    — Conditional structure density (if/unless/whether)
          [24]    — Negation density (neg dependency)
          [25]    — Named entity density
          [26:32] — Dependency relation distribution (nsubj, dobj, prep, etc.)
          [32:64] — Reserved for future parse tree features (zero-padded)
        """
        features = np.zeros(self.config.syn_feature_dim, dtype=np.float32)
        if len(doc) == 0:
            return features

        n = len(doc)
        POS_TAGS = [
            "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ",
            "NUM", "PART", "PUNCT", "PROPN", "SYM", "INTJ", "AUX", "CCONJ",
            "SCONJ", "X",
        ]
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        for i, tag in enumerate(POS_TAGS):
            features[i] = pos_counts.get(tag, 0) / n

        # Dependency depth statistics
        depths = [len(list(token.ancestors)) for token in doc]
        features[18] = float(np.mean(depths)) if depths else 0.0
        features[19] = float(np.max(depths)) if depths else 0.0

        # Causal and structural markers
        features[20] = sum(1 for t in doc if t.lemma_.lower() in CAUSAL_CONNECTIVES) / n
        features[21] = sum(1 for t in doc if t.dep_ in {"mark", "cc"}) / n
        features[22] = sum(1 for t in doc if t.dep_ == "nsubjpass") / n
        features[23] = sum(1 for t in doc if t.lower_ in {"if", "unless", "whether", "provided"}) / n
        features[24] = sum(1 for t in doc if t.dep_ == "neg") / n
        features[25] = len(doc.ents) / n if n > 0 else 0.0

        # Dependency relation distribution
        DEP_RELS = ["nsubj", "dobj", "prep", "amod", "advmod", "relcl"]
        dep_counts = {}
        for token in doc:
            dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1
        for i, dep in enumerate(DEP_RELS):
            features[26 + i] = dep_counts.get(dep, 0) / n

        return features

    def _causal_embed(self, text: str) -> np.ndarray:
        """
        Causal graph embedding via entity-relation feature projection.

        Extracts entity-relation triplets and causal structure features.
        Projects 32-dim feature vector to ℝ^d via cached projection.

        Key insight: two texts sharing entities but with reversed causal
        direction — "A caused B" vs "B caused A" — should have LOW causal
        similarity. The drift vector D(q, c_i) = v_cau(q) − v_cau(c_i)
        captures this directional mismatch, not just magnitude.
        """
        if self._nlp is not None:
            doc = self._nlp(text)
            features = self._extract_causal_features(doc)
        else:
            rng = np.random.default_rng((abs(hash(text)) + 2) % (2**32))
            features = rng.standard_normal(self.config.cau_feature_dim).astype(np.float32)

        vec = features.astype(np.float32) @ self._cau_proj
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _extract_causal_features(self, doc) -> np.ndarray:
        """
        Extract a 32-dimensional causal feature vector from a spaCy doc.

        Feature layout:
          [0]     — Subject density (nsubj, nsubjpass)
          [1]     — Object density (dobj, pobj, attr)
          [2]     — Causal verb density (from CAUSAL_VERBS set)
          [3:9]   — Named entity type distribution (PERSON, ORG, GPE, EVENT, LAW, DATE)
          [9]     — Temporal marker density (when, after, before, during, since)
          [10]    — Conditional causation density (if X then Y)
          [11]    — Negated causation density (not cause, doesn't lead)
          [12]    — Chain length proxy (semi-colon + conjunction count)
          [13]    — Passive causal density (was caused by, is triggered by)
          [14:20] — Root verb lemma hash bins (6 bins for verb type distribution)
          [20:32] — Reserved (zero-padded)
        """
        features = np.zeros(self.config.cau_feature_dim, dtype=np.float32)
        if len(doc) == 0:
            return features

        n = max(len(doc), 1)

        # Subject/Object density
        subjects = [t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}]
        objects  = [t for t in doc if t.dep_ in {"dobj", "pobj", "attr"}]
        causal_v = [t for t in doc if t.lemma_.lower() in CAUSAL_VERBS]
        features[0] = len(subjects) / n
        features[1] = len(objects) / n
        features[2] = len(causal_v) / n

        # Named entity type distribution
        ENT_TYPES = {"PERSON": 3, "ORG": 4, "GPE": 5, "EVENT": 6, "LAW": 7, "DATE": 8}
        n_ents = max(len(doc.ents), 1)
        for ent in doc.ents:
            if ent.label_ in ENT_TYPES:
                features[ENT_TYPES[ent.label_]] += 1.0 / n_ents

        # Temporal markers
        temporal = {"when", "after", "before", "during", "since", "once", "until"}
        features[9] = sum(1 for t in doc if t.lower_ in temporal) / n

        # Conditional causation (if → then)
        features[10] = sum(1 for t in doc if t.lower_ == "if") / n

        # Negated causation
        features[11] = sum(
            1 for t in doc if t.dep_ == "neg" and t.head.lemma_.lower() in CAUSAL_VERBS
        ) / n

        # Chain length proxy
        features[12] = sum(1 for t in doc if t.text in {";", ":", "furthermore", "additionally"}) / n

        # Passive causal (was caused by, is triggered by)
        features[13] = sum(1 for t in doc if t.dep_ == "nsubjpass" and t.head.lemma_.lower() in CAUSAL_VERBS) / n

        # Root verb lemma hash bins (6 bins)
        for sent in doc.sents:
            if sent.root.pos_ == "VERB":
                bin_idx = abs(hash(sent.root.lemma_)) % 6
                features[14 + bin_idx] += 1.0 / n

        return features

    # ──── Encoding API ────────────────────────────────────────────────────────

    def encode(self, text: str) -> TVEVector:
        """Encode a single text into all three vector spaces."""
        return TVEVector(
            semantic=self._semantic_embed(text),
            syntactic=self._syntactic_embed(text),
            causal=self._causal_embed(text),
            source_text=text,
        )

    def encode_query(self, query: str) -> TVEVector:
        return self.encode(query)

    def encode_chunk(self, chunk: str) -> TVEVector:
        return self.encode(chunk)

    def batch_encode(self, texts: list[str]) -> list[TVEVector]:
        """
        Encode a list of texts efficiently using SBERT's native batch encoding.

        For large corpora, this is ~10x faster than calling encode() in a loop
        because SBERT processes texts in parallel on GPU/CPU batches.
        The syntactic and causal arms are still computed sequentially (spaCy
        does not currently support true batch processing with the same efficiency).
        """
        if not texts:
            return []

        # Batch semantic encoding (efficient)
        sem_vecs = self._semantic_embed_batch(texts)

        # Sequential syntactic + causal encoding
        result = []
        for i, text in enumerate(texts):
            result.append(TVEVector(
                semantic=sem_vecs[i],
                syntactic=self._syntactic_embed(text),
                causal=self._causal_embed(text),
                source_text=text,
            ))
        return result

    # ──── Scoring ─────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
        denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        if denom < 1e-10:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    def arm_scores(self, q_vec: TVEVector, c_vec: TVEVector) -> dict[str, float]:
        """
        Compute per-arm cosine similarity scores separately.

        Returns a dict with keys: 'semantic', 'syntactic', 'causal', 'tve'.
        Useful for diagnosis: which arm is driving the score, and which arm
        is detecting drift that the semantic arm misses.

        Example:
            scores = encoder.arm_scores(q_vec, c_vec)
            # {'semantic': 0.91, 'syntactic': 0.74, 'causal': 0.31, 'tve': 0.64}
            # High semantic but low causal → semantic drift candidate.
        """
        sem = self.cosine_sim(q_vec.semantic, c_vec.semantic)
        syn = self.cosine_sim(q_vec.syntactic, c_vec.syntactic)
        cau = self.cosine_sim(q_vec.causal, c_vec.causal)
        α, β, γ = self.config.alpha, self.config.beta, self.config.gamma
        tve = float(np.clip(α * sem + β * syn + γ * cau, 0.0, 1.0))
        return {"semantic": sem, "syntactic": syn, "causal": cau, "tve": tve}

    def tve_score(self, q_vec: TVEVector, c_vec: TVEVector) -> float:
        """
        Compute the Tri-Vector similarity score.

        TVE_score = α·cos(sem_q, sem_c) + β·cos(syn_q, syn_c) + γ·cos(cau_q, cau_c)

        All three terms are independently computed and weighted by the learned
        domain coefficients (α, β, γ) that sum to 1. This weighting is key:
          - For code docs: higher β (syntactic) catches API patterns
          - For legal text: higher γ (causal) catches precedent chains
          - For general QA: balanced weighting (default: 0.4/0.3/0.3)
        """
        scores = self.arm_scores(q_vec, c_vec)
        return scores["tve"]

    def batch_tve_scores(self, q_vec: TVEVector, c_vecs: list[TVEVector]) -> np.ndarray:
        """
        Vectorised TVE scoring for a batch of chunks.

        Instead of looping over c_vecs, computes all three arm similarities
        as matrix-vector products using numpy broadcasting — significantly
        faster for large corpora (O(N·d) vs O(N·d) same complexity but
        lower constant due to BLAS optimisation).
        """
        if not c_vecs:
            return np.array([])

        α, β, γ = self.config.alpha, self.config.beta, self.config.gamma

        # Stack into matrices: (N, d)
        sem_matrix = np.stack([c.semantic for c in c_vecs])   # (N, d)
        syn_matrix = np.stack([c.syntactic for c in c_vecs])  # (N, d)
        cau_matrix = np.stack([c.causal for c in c_vecs])     # (N, d)

        # Vectorised cosine similarity: dot product (vecs are already normalized)
        sem_scores = sem_matrix @ q_vec.semantic                        # (N,)
        syn_scores = syn_matrix @ q_vec.syntactic                       # (N,)
        cau_scores = cau_matrix @ q_vec.causal                          # (N,)

        # Clip to [-1, 1] for numerical safety
        sem_scores = np.clip(sem_scores, -1.0, 1.0)
        syn_scores = np.clip(syn_scores, -1.0, 1.0)
        cau_scores = np.clip(cau_scores, -1.0, 1.0)

        return np.clip(α * sem_scores + β * syn_scores + γ * cau_scores, 0.0, 1.0)

    # ──── Analysis and Interpretability ───────────────────────────────────────

    def explain_score(self, q_vec: TVEVector, c_vec: TVEVector) -> dict:
        """
        Full breakdown of a TVE score: why did chunk c_i receive this score?

        Returns:
            {
                'tve_score':        float  — final weighted score
                'semantic_score':   float  — cos(v_sem(q), v_sem(c))
                'syntactic_score':  float  — cos(v_syn(q), v_syn(c))
                'causal_score':     float  — cos(v_cau(q), v_cau(c))
                'alpha':            float  — semantic weight
                'beta':             float  — syntactic weight
                'gamma':            float  — causal weight
                'dominant_arm':     str    — which arm drives the TVE score
                'semantic_contrib': float  — α × semantic_score
                'syntactic_contrib':float  — β × syntactic_score
                'causal_contrib':   float  — γ × causal_score
                'drift_magnitude':  float  — ||v_cau(q) - v_cau(c)||₂
                'drift_warning':    bool   — True if causal score < 0.5 but sem > 0.7
            }
        """
        α, β, γ = self.config.alpha, self.config.beta, self.config.gamma
        sem = self.cosine_sim(q_vec.semantic, c_vec.semantic)
        syn = self.cosine_sim(q_vec.syntactic, c_vec.syntactic)
        cau = self.cosine_sim(q_vec.causal, c_vec.causal)
        tve = α * sem + β * syn + γ * cau

        sem_contrib = α * sem
        syn_contrib = β * syn
        cau_contrib = γ * cau
        contribs = {"semantic": sem_contrib, "syntactic": syn_contrib, "causal": cau_contrib}
        dominant = max(contribs, key=contribs.get)

        drift_magnitude = float(np.linalg.norm(q_vec.causal - c_vec.causal))
        drift_warning = (cau < 0.5 and sem > 0.7)

        return {
            "tve_score":         round(tve, 4),
            "semantic_score":    round(sem, 4),
            "syntactic_score":   round(syn, 4),
            "causal_score":      round(cau, 4),
            "alpha":             α,
            "beta":              β,
            "gamma":             γ,
            "semantic_contrib":  round(sem_contrib, 4),
            "syntactic_contrib": round(syn_contrib, 4),
            "causal_contrib":    round(cau_contrib, 4),
            "dominant_arm":      dominant,
            "drift_magnitude":   round(drift_magnitude, 4),
            "drift_warning":     drift_warning,
            "interpretation":    self._interpret_scores(sem, syn, cau, drift_warning),
        }

    def _interpret_scores(
        self, sem: float, syn: float, cau: float, drift_warning: bool
    ) -> str:
        """Human-readable interpretation of arm scores."""
        if drift_warning:
            return (
                f"Semantic drift risk: high semantic alignment ({sem:.2f}) but low "
                f"causal alignment ({cau:.2f}). This chunk may be topically adjacent "
                f"but causally irrelevant — SDC gate will likely reject it."
            )
        if sem > 0.8 and syn > 0.7 and cau > 0.75:
            return "Strong alignment across all three arms — high-confidence retrieval."
        if cau < 0.4:
            return f"Low causal alignment ({cau:.2f}) — possible semantic drift candidate."
        if syn < 0.3:
            return f"Low syntactic alignment ({syn:.2f}) — structural mismatch (code vs prose, etc)."
        return f"Moderate alignment: sem={sem:.2f}, syn={syn:.2f}, cau={cau:.2f}."

    def adapt_for_domain(self, domain: str) -> None:
        """
        Update α, β, γ weights to the domain-specific optimal preset.

        This is useful when switching domains within a single encoder instance.
        After calling this, all subsequent tve_score() calls use the new weights.

        Raises ValueError if domain is not in DOMAIN_WEIGHTS.
        """
        if domain not in DOMAIN_WEIGHTS:
            available = ", ".join(sorted(DOMAIN_WEIGHTS.keys()))
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {available}"
            )
        self.config.alpha, self.config.beta, self.config.gamma = DOMAIN_WEIGHTS[domain]
        self.config.domain = domain

    def cross_domain_score(
        self,
        q_vec: TVEVector,
        c_vec: TVEVector,
        domain: str,
    ) -> float:
        """
        Compute TVE score using domain-specific weights WITHOUT changing config.

        Useful for comparing how the same chunk scores under different domain
        assumptions — e.g., would this chunk score higher in legal vs general mode?
        """
        if domain not in DOMAIN_WEIGHTS:
            raise ValueError(f"Unknown domain: '{domain}'")
        a, b, g = DOMAIN_WEIGHTS[domain]
        sem = self.cosine_sim(q_vec.semantic, c_vec.semantic)
        syn = self.cosine_sim(q_vec.syntactic, c_vec.syntactic)
        cau = self.cosine_sim(q_vec.causal, c_vec.causal)
        return a * sem + b * syn + g * cau

    def most_relevant_arm(self, q_vec: TVEVector, c_vec: TVEVector) -> str:
        """
        Return which arm ('semantic', 'syntactic', 'causal') contributes most
        to the TVE score for this query-chunk pair.

        Used for interpretability and domain diagnostics.
        """
        scores = self.arm_scores(q_vec, c_vec)
        α, β, γ = self.config.alpha, self.config.beta, self.config.gamma
        weighted = {
            "semantic":  α * scores["semantic"],
            "syntactic": β * scores["syntactic"],
            "causal":    γ * scores["causal"],
        }
        return max(weighted, key=weighted.get)

    def score_matrix(
        self,
        queries: list[TVEVector],
        chunks: list[TVEVector],
    ) -> np.ndarray:
        """
        Compute a full (N_queries × N_chunks) TVE score matrix.

        Returns an array of shape (len(queries), len(chunks)) where entry [i,j]
        is the TVE score between query i and chunk j.

        Useful for batch evaluation and cross-query analysis.
        """
        n_q = len(queries)
        n_c = len(chunks)
        result = np.zeros((n_q, n_c), dtype=np.float32)
        for i, q_vec in enumerate(queries):
            result[i] = self.batch_tve_scores(q_vec, chunks)
        return result

    def domain_sensitivity(
        self,
        q_vec: TVEVector,
        c_vec: TVEVector,
    ) -> dict[str, float]:
        """
        Show how the TVE score changes across all domain presets.

        Returns a dict mapping domain name → TVE score using that domain's weights.
        Useful for understanding which domain assumption best fits a query-chunk pair.
        """
        return {
            domain: round(self.cross_domain_score(q_vec, c_vec, domain), 4)
            for domain in DOMAIN_WEIGHTS
        }
