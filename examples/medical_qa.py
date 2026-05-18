"""
VORTEXRAG Medical QA Example

Demonstrates VORTEXRAG for medical mechanism synthesis.
The CPG module separates the two causal chains (mRNA path vs vector path)
and CCB orders them correctly in the context window.

Test Case: mRNA vs viral vector vaccine spike protein expression mechanism

Run: python examples/medical_qa.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vortexrag import VortexRAG, VortexRAGConfig


MEDICAL_CORPUS = [
    """mRNA Vaccine Mechanism — Step 1: Lipid Nanoparticle Delivery
    The mRNA sequence encoding the SARS-CoV-2 spike protein is encapsulated
    in lipid nanoparticles (LNPs). LNPs protect the fragile mRNA from degradation
    by extracellular RNases and facilitate cellular uptake via endocytosis.
    The LNP formulation is the key innovation enabling mRNA vaccine stability.""",

    """mRNA Vaccine Mechanism — Step 2: Cytoplasmic Translation
    Once inside the cell, the mRNA escapes the endosome into the cytoplasm.
    Ribosomes directly translate the mRNA into spike protein WITHOUT the
    mRNA ever entering the nucleus. The mRNA is single-stranded RNA and
    cannot integrate into genomic DNA. It is degraded by RNases within 24-72 hours.""",

    """mRNA Vaccine Mechanism — Step 3: Immune Activation
    The translated spike protein fragments are displayed on the cell surface
    via MHC-I molecules, triggering cytotoxic T cell responses. Secreted
    fragments are recognized by B cells, stimulating antibody production.
    Memory B cells and T cells are formed for long-term immunity.""",

    """Viral Vector Vaccine Mechanism — Step 1: Adenovirus Delivery
    Viral vector vaccines (AstraZeneca, J&J) use a modified chimpanzee or
    human adenovirus (Ad26, ChAdOx1) engineered to carry the spike protein
    DNA sequence. The adenovirus has been modified to prevent self-replication.
    It enters cells via surface receptor binding, the same pathway as wild-type adenovirus.""",

    """Viral Vector Vaccine Mechanism — Step 2: Nuclear DNA Transcription
    Unlike mRNA vaccines, the viral vector delivers DNA to the CELL NUCLEUS.
    The spike protein gene is transcribed into mRNA by the cell's RNA polymerase.
    This mRNA then travels to the cytoplasm where it is translated into spike protein.
    The extra nuclear step adds latency but does not cause genomic integration.""",

    """Viral Vector Vaccine Mechanism — Step 3: Immune Response
    The spike protein produced by the viral vector pathway activates the same
    downstream immune responses as mRNA vaccines. However, pre-existing immunity
    to the adenovirus vector can reduce vaccine efficacy — a key limitation
    of viral vector approaches that mRNA vaccines do not face.""",

    # Distractor — semantically similar but conflates mechanisms (CPG test)
    """Clinical Efficacy Comparison:
    Both mRNA (Pfizer-BioNTech, Moderna) and viral vector (AstraZeneca, J&J)
    vaccines demonstrated strong efficacy against COVID-19 in clinical trials.
    Both types produce neutralizing antibodies and cellular immunity.
    The choice between platforms has more to do with logistics (cold chain,
    manufacturing) than fundamental immunological differences.""",

    # Another distractor
    """Spike Protein Structure:
    The SARS-CoV-2 spike protein is a trimeric type I fusion protein.
    It consists of two subunits: S1 (receptor binding domain) and S2 (fusion).
    The receptor-binding domain binds ACE2 on human cells. Both vaccine
    platforms express the prefusion-stabilized spike protein to maximize
    neutralizing antibody responses.""",
]

QUERY = "What is the mechanistic difference between mRNA vaccines and viral vector vaccines in spike protein expression?"


def run_medical_qa():
    print("=" * 65)
    print("  VORTEXRAG — Medical QA: Mechanism Synthesis")
    print("=" * 65)
    print(f"\nQuery: {QUERY}\n")

    config = VortexRAGConfig(domain="medical", verbose=True)
    config.vrc.candidate_pool = 12
    config.vrc.top_k = 8
    config.rfg.top_m = 6

    rag = VortexRAG(config=config)
    rag.index(additional_texts=MEDICAL_CORPUS)
    result = rag.query(QUERY)

    print("\n" + "─" * 65)
    print("VORTEXRAG PIPELINE RESULTS")
    print("─" * 65)
    print(f"\nΦ̃-Ranked Context Window:")
    for i, (chunk, phi) in enumerate(zip(result.context_window, result.phi_scores), 1):
        print(f"\n  [C{i}] Φ̃={phi:.4f} — {chunk[:120]}...")

    print(f"\nESR: {result.esr:.4f} | ΔR: {result.delta_r:.4f} | "
          f"Purged: {result.purge_count} | Latency: {result.latency_ms:.1f}ms")

    print("\n" + "─" * 65)
    print("WHY CPG MATTERS HERE:")
    print("─" * 65)
    print("""
Standard RAG problem (Context Poisoning):
  The 'Clinical Efficacy' and 'Spike Protein Structure' chunks have HIGH
  semantic similarity to the query (they mention spike protein, both vaccine
  types, mechanisms). Top-k RAG would include them. The LLM then conflates
  the pathways because both are in context simultaneously with no ordering.

VORTEXRAG solution:
  1. CPG detects the efficacy/structure chunks have low SDS (they don't
     causally answer 'what is the mechanistic difference')
  2. CPG purges them via iterative ESR maximization
  3. CCB orders: mRNA delivery → mRNA translation → [then] vector delivery → vector transcription
  4. The LLM sees a causally ordered, two-pathway narrative — clean comparison
    """)


if __name__ == "__main__":
    run_medical_qa()
