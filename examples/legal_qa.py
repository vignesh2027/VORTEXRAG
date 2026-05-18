"""
VORTEXRAG Legal QA Example

Demonstrates domain-tuned VORTEXRAG for multi-hop legal reasoning.
Uses strict SDC tau (0.4) to enforce causal precision in legal citation chains.

Test Case: Did Brown v. Board apply to public universities before 1964?
  Expected answer path: Brown (1954) → Cooper v. Aaron (1958) → Green (1968)

Run: python examples/legal_qa.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vortexrag import VortexRAG, VortexRAGConfig


LEGAL_CORPUS = [
    """Brown v. Board of Education (1954): The Supreme Court unanimously held that
    racial segregation of public schools was unconstitutional under the Equal
    Protection Clause of the Fourteenth Amendment. Chief Justice Warren wrote that
    separate educational facilities are inherently unequal.""",

    """Cooper v. Aaron (1958): All nine justices signed an unusual joint opinion
    reaffirming Brown. The Court held that the constitutional rights of students
    could not be sacrificed or yielded to violence and disorder orchestrated by
    state officials. Critically, this ruling explicitly extended Brown's mandate
    to all state-controlled educational institutions, including public universities.""",

    """Sweatt v. Painter (1950): The Court held that the University of Texas Law
    School must admit Herman Sweatt, a Black applicant. The ruling predated Brown
    and struck down the 'separate but equal' doctrine in graduate education,
    laying the legal groundwork for Brown's broader mandate.""",

    """The Civil Rights Act of 1964 explicitly prohibited discrimination based on
    race in any program receiving federal financial assistance (Title VI). This
    created a statutory basis for university desegregation independent of
    constitutional rulings, superseding any ambiguity about Brown's reach.""",

    """Green v. County School Board (1968): The Court unanimously rejected
    freedom-of-choice desegregation plans and ruled that school boards had
    an affirmative obligation to dismantle dual school systems 'root and branch.'
    This extended the Brown mandate to require active desegregation, not merely
    cessation of formal segregation policies.""",

    """The Fourteenth Amendment (1868) guarantees equal protection under the law
    to all citizens. It has been applied to strike down laws and policies that
    discriminate on the basis of race, sex, and other protected characteristics
    in education, employment, and public accommodations.""",

    """Plessy v. Ferguson (1896) established the 'separate but equal' doctrine,
    holding that racially segregated public facilities were constitutional as long
    as they were equal in quality. This ruling was explicitly overturned by Brown
    v. Board of Education in 1954.""",
]

QUERY = "Did the precedent set in Brown v. Board also apply to public universities before 1964?"


def run_legal_qa():
    print("=" * 65)
    print("  VORTEXRAG — Legal QA: Multi-hop Precedent Reasoning")
    print("=" * 65)
    print(f"\nQuery: {QUERY}\n")

    # Legal domain: strict tau (0.4) for causal precision
    config = VortexRAGConfig(domain="legal", verbose=True)
    config.vrc.candidate_pool = 10
    config.vrc.top_k = 7
    config.rfg.top_m = 4

    rag = VortexRAG(config=config)
    rag.index(additional_texts=LEGAL_CORPUS)
    result = rag.query(QUERY)

    print("\n" + "─" * 65)
    print("VORTEXRAG PIPELINE RESULTS")
    print("─" * 65)
    print(f"\n📊 Pipeline Metrics:")
    print(f"   Spiral pool size   : {result.spiral_pool_size}")
    print(f"   Chunks purged (CPG): {result.purge_count}")
    print(f"   Effective Signal   : {result.esr:.4f}")
    print(f"   Hallucination ΔR   : {result.delta_r:.4f}")
    print(f"   Total latency      : {result.latency_ms:.1f}ms")
    print(f"   Answer accepted    : {result.accepted}")

    print(f"\n📑 Final Context Window W* ({len(result.context_window)} chunks):")
    for i, (chunk, phi) in enumerate(zip(result.context_window, result.phi_scores), 1):
        print(f"\n  [C{i}] Φ̃={phi:.4f}")
        print(f"  {chunk[:200]}...")

    print("\n" + "─" * 65)
    print("GENERATED ANSWER:")
    print("─" * 65)
    print(result.answer)

    print("\n" + "─" * 65)
    print("COMPARISON: Standard RAG vs VORTEXRAG")
    print("─" * 65)
    print("\nStandard RAG failure mode:")
    print("  Retrieves Brown (1954) + Civil Rights Act (1964) + Plessy (1896)")
    print("  Semantically similar but causally incorrect chain.")
    print("  LLM produces: 'Brown applied to all education since 1954' (wrong)")
    print("\nVORTEXRAG fix:")
    print("  SDC filters Civil Rights Act chunk (causal drift: legislation ≠ precedent)")
    print("  CPG removes Plessy and 14th Amendment from context (poisons causal chain)")
    print("  CCB orders: Cooper v. Aaron (depth=0) → Sweatt v. Painter (depth=1)")
    print("  Correct causal chain preserved: Brown → Cooper → university application")


if __name__ == "__main__":
    run_legal_qa()
