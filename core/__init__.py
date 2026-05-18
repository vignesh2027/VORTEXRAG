"""
VORTEXRAG — Vector Orthogonal Resonance-Tuned EXtraction RAG
A novel RAG framework solving Semantic Drift and Context Poisoning simultaneously.
"""

from .tve import TriVectorEncoder
from .vrc import VortexRetrievalCone
from .sdc import SemanticDriftCorrector
from .cpg import ContextPoisonGuard
from .rfg import RankFusionGate
from .ccb import CausalContextBuilder
from .fv import FaithfulnessVerifier

__version__ = "0.1.0"
__all__ = [
    "TriVectorEncoder",
    "VortexRetrievalCone",
    "SemanticDriftCorrector",
    "ContextPoisonGuard",
    "RankFusionGate",
    "CausalContextBuilder",
    "FaithfulnessVerifier",
]
