"""
VORTEXRAG Interactive Demo

Shows each pipeline layer's decisions for a given query.
Run with: python examples/demo_gradio.py

The demo works fully offline with mock/fallback encoders — no API keys required.
Optional: install sentence-transformers for real semantic embeddings.
"""

from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Any

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("[WARNING] gradio not installed. Run: pip install gradio")

from core.tve import TriVectorEncoder, TVEConfig, DOMAIN_WEIGHTS
from core.vrc import VortexRetrievalCone, VRCConfig
from core.sdc import SemanticDriftCorrector, SDCConfig
from core.cpg import ContextPoisonGuard, CPGConfig
from core.rfg import RankFusionGate, RFGConfig
from core.ccb import CausalContextBuilder, CCBConfig
from core.fv import FaithfulnessVerifier, FVConfig


# ── Default example documents (used when user provides nothing) ───────────────

DEFAULT_DOCS = """The 2008 financial crisis was caused by excessive leverage in mortgage-backed securities. Banks issued subprime mortgages to borrowers who could not repay them. When housing prices fell, defaults cascaded through the financial system.
---
Lehman Brothers filed for bankruptcy on September 15, 2008, sending shockwaves through global markets. The bank had massive exposure to toxic mortgage-backed assets and could not meet its debt obligations.
---
The Federal Reserve responded by cutting interest rates to near zero and implementing quantitative easing programs to inject liquidity into frozen credit markets.
---
Credit default swaps amplified systemic risk by creating synthetic exposure to subprime mortgages far exceeding the actual mortgage market. When defaults rose, counterparties could not pay out on these contracts.
---
The housing bubble itself was fueled by low interest rates after the dot-com crash, predatory lending practices, and the belief that house prices would always rise."""

DEFAULT_QUERY = "What caused the 2008 financial crisis?"
DEFAULT_DOMAIN = "financial"


# ── Pipeline trace function ────────────────────────────────────────────────────

def run_pipeline_trace(query: str, domain: str, documents: str) -> tuple[str, str]:
    """
    Run VORTEXRAG and show layer-by-layer trace.

    Returns (trace_output, final_answer).
    """
    # Parse documents from text input
    chunks = [d.strip() for d in documents.split("---") if d.strip()]
    if not chunks:
        return "No documents provided. Please paste some text separated by ---", "N/A"

    trace_lines = []
    trace_lines.append(f"VORTEXRAG Pipeline Trace")
    trace_lines.append(f"Query: {query!r}")
    trace_lines.append(f"Domain: {domain}")
    trace_lines.append(f"Documents: {len(chunks)} chunks")
    trace_lines.append("=" * 70)

    # ── Layer 1: TVE Encoding ─────────────────────────────────────────────────
    trace_lines.append("\nLAYER 2 — Tri-Vector Encoding (TVE)")
    trace_lines.append("-" * 40)

    a, b, g = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])
    tve_config = TVEConfig(alpha=a, beta=b, gamma=g, domain=domain)
    encoder = TriVectorEncoder(tve_config)

    q_vec = encoder.encode_query(query)
    corpus_vecs = [encoder.encode_chunk(c) for c in chunks]

    trace_lines.append(f"  Query TVE vector: {q_vec.dim}d combined [{q_vec.semantic.shape[0]}d sem + {q_vec.syntactic.shape[0]}d syn + {q_vec.causal.shape[0]}d cau]")
    trace_lines.append(f"  Weights: α={a} (semantic), β={b} (syntactic), γ={g} (causal)")
    trace_lines.append(f"  Encoded {len(corpus_vecs)} corpus chunks")

    # Compute TVE scores for each chunk
    tve_scores = encoder.batch_tve_scores(q_vec, corpus_vecs)
    trace_lines.append("\n  Per-chunk TVE scores:")
    for i, (chunk, score) in enumerate(zip(chunks, tve_scores)):
        preview = chunk[:60].replace("\n", " ")
        bar = "█" * int(float(score) * 20)
        trace_lines.append(f"    [{i+1:02d}] TVE={float(score):.3f} {bar} | {preview}...")

    # ── Layer 2: VRC Spiral Retrieval ─────────────────────────────────────────
    trace_lines.append("\nLAYER 3 — Vortex Retrieval Cone (VRC)")
    trace_lines.append("-" * 40)

    vrc_config = VRCConfig(n_spiral=2, lambda_decay=0.5, top_k=min(len(chunks), 8), candidate_pool=len(chunks))
    vrc = VortexRetrievalCone(encoder, vrc_config)
    spiral_pool = vrc.retrieve(q_vec, corpus_vecs, chunks)

    trace_lines.append(f"  Spiral pool: top-{len(spiral_pool)} by spiral_rank")
    trace_lines.append(f"  n_spiral={vrc_config.n_spiral}, λ_decay={vrc_config.lambda_decay}")

    stats = vrc.pool_statistics(spiral_pool)
    if stats:
        trace_lines.append(f"  Mean TVE: {stats['mean_tve']:.3f}")
        trace_lines.append(f"  Mean θ: {stats['mean_theta_degrees']:.1f}°")
        trace_lines.append(f"  Negative rank (suppressed): {stats['n_negative_rank']}/{stats['n_candidates']}")

    trace_lines.append("\n  Top-5 by spiral_rank:")
    for i, cand in enumerate(spiral_pool[:5]):
        preview = cand.chunk_text[:55].replace("\n", " ")
        sign = "+" if cand.spiral_rank >= 0 else "-"
        trace_lines.append(
            f"    [{i+1}] spiral_rank={cand.spiral_rank:+.3f} | TVE={cand.tve_score:.3f} | "
            f"θ={cand.theta*57.3:.1f}° | {preview}..."
        )

    # ── Layer 3: SDC — Semantic Drift Correction ──────────────────────────────
    trace_lines.append("\nLAYER 4a — Semantic Drift Corrector (SDC)")
    trace_lines.append("-" * 40)

    sdc_config = SDCConfig(domain=domain)
    sdc = SemanticDriftCorrector(sdc_config)
    sdc_results = sdc.filter(q_vec, spiral_pool)
    accepted_sdc = sdc.accepted_only(sdc_results)
    rejected_sdc = sdc.rejected_only(sdc_results)

    trace_lines.append(f"  τ (drift temperature): {sdc_config.tau}")
    trace_lines.append(f"  δ_SDC (gate): {sdc_config.delta_sdc}")
    trace_lines.append(f"  Accepted: {len(accepted_sdc)}/{len(sdc_results)} chunks")
    trace_lines.append(f"  Rejected: {len(rejected_sdc)} chunks (SDS < {sdc_config.delta_sdc})")

    if rejected_sdc:
        trace_lines.append("\n  Rejected chunks (semantic drift detected):")
        for r in rejected_sdc[:3]:
            preview = r.candidate.chunk_text[:55].replace("\n", " ")
            trace_lines.append(
                f"    ✗ SDS={r.sds_score:.3f} (drift={r.drift_norm:.2f}, {r.drift_category}) | {preview}..."
            )

    if accepted_sdc:
        trace_lines.append("\n  Accepted chunks:")
        for r in accepted_sdc[:5]:
            preview = r.candidate.chunk_text[:55].replace("\n", " ")
            trace_lines.append(
                f"    ✓ SDS={r.sds_score:.3f} | {preview}..."
            )

    # ── Layer 4: CPG — Context Poison Guard ───────────────────────────────────
    trace_lines.append("\nLAYER 4b — Context Poison Guard (CPG)")
    trace_lines.append("-" * 40)

    cpg_config = CPGConfig(theta_cpg=3.5, min_chunks=2)
    cpg = ContextPoisonGuard(cpg_config)

    working_sdc = accepted_sdc if len(accepted_sdc) >= 2 else sdc_results[:max(3, len(sdc_results))]
    cpg_eval = cpg.evaluate(q_vec, working_sdc)

    trace_lines.append(f"  θ_CPG: {cpg_config.theta_cpg}")
    trace_lines.append(f"  ESR (Effective Signal Ratio): {cpg_eval.esr:.4f}")
    trace_lines.append(f"  Poison Index: {cpg_eval.poison_index:.4f}")
    trace_lines.append(f"  Clean: {'YES' if cpg_eval.is_clean else 'NO (purging applied)'}")
    trace_lines.append(f"  Chunks purged: {cpg_eval.purge_count}")
    trace_lines.append(f"  Window size after CPG: {len(cpg_eval.window)}")

    if cpg_eval.purge_history:
        trace_lines.append("\n  Purge history:")
        for rnd, cid, esr_b, esr_a in cpg_eval.purge_history:
            trace_lines.append(f"    Round {rnd+1}: removed chunk #{cid} | ESR {esr_b:.3f} → {esr_a:.3f}")

    # ── Layer 5: RFG — Rank Fusion Gate ──────────────────────────────────────
    trace_lines.append("\nLAYER 5a — Rank Fusion Gate (RFG)")
    trace_lines.append("-" * 40)

    rfg_config = RFGConfig(domain=domain, top_m=min(5, len(cpg_eval.window)))
    rfg_config.apply_domain_preset()
    rfg = RankFusionGate(rfg_config)
    ranked = rfg.rank(q_vec, cpg_eval)
    final_ranked = rfg.select_top_m(ranked)

    trace_lines.append(f"  Fusion weights: α={rfg_config.alpha} (TVE), β={rfg_config.beta} (SDS), γ={rfg_config.gamma} (ESR)")
    trace_lines.append(f"  Φ formula: TVE^α × SDS^β × ESR_contrib^γ")
    trace_lines.append(f"  Top-{len(final_ranked)} by Φ̃:")
    for i, chunk in enumerate(final_ranked):
        preview = chunk.chunk_text[:50].replace("\n", " ")
        trace_lines.append(
            f"    [{i+1}] Φ̃={chunk.phi_norm:.4f} | TVE={chunk.tve_score:.3f} | "
            f"SDS={chunk.sds_score:.3f} | ESR={chunk.esr_contribution:.3f} | {preview}..."
        )

    # ── Layer 6: CCB — Causal Context Builder ────────────────────────────────
    trace_lines.append("\nLAYER 5b — Causal Context Builder (CCB)")
    trace_lines.append("-" * 40)

    ccb_config = CCBConfig(enable_dedup=True, dedup_threshold=0.92)
    ccb = CausalContextBuilder(ccb_config)
    ordered_slots = ccb.build(query, final_ranked)
    context_string = ccb.to_context_string(ordered_slots, include_citations=True)

    trace_lines.append(f"  Ordered {len(ordered_slots)} chunks by causal depth + Φ̃ rank")
    trace_lines.append("\n  Context ordering:")
    for slot in ordered_slots:
        depth_label = {0: "ROOT_CAUSE", 1: "EFFECT", 2: "SUPPORTING"}.get(slot.causal_depth, f"DEPTH_{slot.causal_depth}")
        preview = slot.chunk.chunk_text[:50].replace("\n", " ")
        trace_lines.append(
            f"    [{depth_label}] pos={slot.slot_position:.0f} | depth={slot.causal_depth} | "
            f"Φ̃={slot.chunk.phi_norm:.3f} | {preview}..."
        )

    # ── Layer 7: FV — Faithfulness Verifier ──────────────────────────────────
    trace_lines.append("\nLAYER 6b — Faithfulness Verifier (FV)")
    trace_lines.append("-" * 40)

    # Generate a stub answer from context
    first_block = context_string.split("\n\n")[0][:300] if context_string else "No context available."
    answer = first_block

    fv_config = FVConfig(delta_fv=0.15, max_iterations=3, use_nli=False)
    fv = FaithfulnessVerifier(fv_config)
    fv_result = fv.verify(answer, context_string)

    trace_lines.append(f"  δ_FV (threshold): {fv_config.delta_fv}")
    trace_lines.append(f"  ΔR (hallucination score): {fv_result.delta_r:.4f}")
    trace_lines.append(f"  ROUGE-L: {fv_result.rouge_l:.4f}")
    trace_lines.append(f"  ROUGE-1: {fv_result.rouge_1:.4f}")
    trace_lines.append(f"  NLI proxy: {fv_result.nli_score:.4f}")
    trace_lines.append(f"  Grounding: {fv_result.grounding:.4f}")
    trace_lines.append(f"  Verdict: {'ACCEPTED' if fv_result.accepted else 'REJECTED (would regenerate)'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    trace_lines.append("\n" + "=" * 70)
    trace_lines.append("PIPELINE SUMMARY")
    trace_lines.append(f"  VRC spiral pool:   {len(spiral_pool)} chunks")
    trace_lines.append(f"  SDC accepted:      {len(accepted_sdc)}/{len(sdc_results)}")
    trace_lines.append(f"  CPG purged:        {cpg_eval.purge_count}")
    trace_lines.append(f"  Final context W*:  {len(ordered_slots)} chunks")
    trace_lines.append(f"  ESR:               {cpg_eval.esr:.4f}")
    trace_lines.append(f"  ΔR:                {fv_result.delta_r:.4f}")
    trace_lines.append(f"  Accepted:          {'YES' if fv_result.accepted else 'NO'}")

    trace_output = "\n".join(trace_lines)
    final_answer = f"[Stub answer from top context chunk]\n\n{answer}"

    return trace_output, final_answer


# ── Gradio interface ──────────────────────────────────────────────────────────

DOMAIN_CHOICES = [
    "general", "medical", "legal", "financial", "scientific",
    "code", "cybersecurity", "educational", "historical", "customer", "creative",
]


def build_gradio_app():
    """Build and return the Gradio app."""
    with gr.Blocks(
        title="VORTEXRAG — Interactive Pipeline Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # VORTEXRAG Interactive Pipeline Demo

        **Vector Orthogonal Resonance-Tuned EXtraction Retrieval-Augmented Generation**

        This demo shows each layer of the VORTEXRAG pipeline in action.
        Paste your documents below (separated by `---`), enter a query, and see
        exactly how each layer processes the input.

        > **Note:** Running with hash-based fallback encoders (no real SBERT/spaCy needed).
        > Install `sentence-transformers` for full semantic embeddings.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                query_input = gr.Textbox(
                    label="Query",
                    placeholder="What caused the 2008 financial crisis?",
                    value=DEFAULT_QUERY,
                    lines=2,
                )

                domain_dropdown = gr.Dropdown(
                    label="Domain",
                    choices=DOMAIN_CHOICES,
                    value=DEFAULT_DOMAIN,
                    info="Domain preset affects TVE weights, SDC strictness, and RFG fusion.",
                )

                docs_input = gr.Textbox(
                    label="Documents (separate chunks with ---)",
                    placeholder="Paste your text chunks here, separated by ---",
                    value=DEFAULT_DOCS,
                    lines=15,
                )

                run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

                gr.Markdown("""
                **Tip:** Each `---` separator creates one document chunk.
                Try 3–8 chunks for best results.
                """)

            with gr.Column(scale=2):
                gr.Markdown("### Pipeline Output")

                trace_output = gr.Textbox(
                    label="Layer-by-Layer Trace",
                    lines=40,
                    max_lines=60,
                    show_copy_button=True,
                )

                answer_output = gr.Textbox(
                    label="Generated Answer (Stub)",
                    lines=5,
                    info="Replace mock_llm with your actual LLM for real generation.",
                )

        run_btn.click(
            fn=run_pipeline_trace,
            inputs=[query_input, domain_dropdown, docs_input],
            outputs=[trace_output, answer_output],
        )

        # Load default on startup
        demo.load(
            fn=run_pipeline_trace,
            inputs=[query_input, domain_dropdown, docs_input],
            outputs=[trace_output, answer_output],
        )

        gr.Markdown("""
        ---
        ### Domain Presets Available

        | Domain | α (semantic) | β (syntactic) | γ (causal) |
        |--------|-------------|----------------|-------------|
        | general | 0.40 | 0.30 | 0.30 |
        | medical | 0.35 | 0.25 | 0.40 |
        | legal | 0.35 | 0.25 | 0.40 |
        | code | 0.30 | 0.45 | 0.25 |
        | financial | 0.40 | 0.30 | 0.30 |
        | creative | 0.50 | 0.25 | 0.25 |

        ### Pipeline Layers
        1. **Layer 2 - TVE**: Tri-Vector Encoding → 3×768d embeddings per chunk
        2. **Layer 3 - VRC**: Vortex spiral topology → top-k by spiral_rank
        3. **Layer 4a - SDC**: Semantic Drift Correction → rejects causally drifted chunks
        4. **Layer 4b - CPG**: Context Poison Guard → ESR-based window purification
        5. **Layer 5a - RFG**: Rank Fusion Gate → Φ-score multiplicative ranking
        6. **Layer 5b - CCB**: Causal Context Builder → causal depth ordering
        7. **Layer 6b - FV**: Faithfulness Verifier → ΔR hallucination detection
        """)

    return demo


# ── CLI trace (no Gradio needed) ───────────────────────────────────────────────

def run_cli_demo():
    """Run the pipeline trace without Gradio (for testing/CI)."""
    print("=" * 70)
    print("VORTEXRAG CLI Demo (no Gradio required)")
    print("=" * 70)

    trace, answer = run_pipeline_trace(DEFAULT_QUERY, DEFAULT_DOMAIN, DEFAULT_DOCS)
    print(trace)
    print("\n--- FINAL ANSWER ---")
    print(answer)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VORTEXRAG Interactive Demo")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no Gradio)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    if args.cli or not GRADIO_AVAILABLE:
        run_cli_demo()
    else:
        print("[VORTEXRAG Demo] Starting Gradio interface...")
        app = build_gradio_app()
        app.launch(server_port=args.port, share=args.share)
