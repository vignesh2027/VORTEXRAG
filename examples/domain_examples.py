"""
Domain-specific VORTEXRAG examples showing all 11 presets.

Demonstrates:
  - Medical: mRNA vaccine mechanism query
  - Legal: precedent chain query
  - Financial: 2008 crisis causal query
  - Scientific: supernova progenitor query
  - Code: debugging exception query
  - Cybersecurity: exploit chain query
  - Educational: concept learning query
  - Historical: event causation query
  - Customer support: bug vs feature query
  - Creative: thematic resonance query
  - General: balanced multi-hop query

Usage:
    python examples/domain_examples.py
    python examples/domain_examples.py --domain medical
    python examples/domain_examples.py --domain all
"""

from __future__ import annotations

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortexrag import VortexRAG, VortexRAGConfig


# ── Domain example definitions ────────────────────────────────────────────────

DOMAIN_EXAMPLES = {

    "medical": {
        "domain": "medical",
        "query": "What is the mechanistic difference between mRNA vaccines and viral vector vaccines in spike protein expression?",
        "corpus": [
            "mRNA vaccines work by delivering messenger RNA encoding the spike protein into host cells. Ribosomes in the cytoplasm translate the mRNA into spike protein, which is then displayed on the cell surface. The immune system recognizes the foreign protein and produces antibodies. The mRNA is degraded within 48–72 hours and never enters the cell nucleus.",
            "Viral vector vaccines use a modified adenovirus to carry DNA encoding the spike protein into host cells. The DNA is transcribed into mRNA in the cell nucleus by RNA polymerase, which is then exported to the cytoplasm and translated into spike protein. The adenovirus vector is engineered to be replication-incompetent and does not integrate into the genome.",
            "Both vaccine types successfully generate neutralizing antibodies against SARS-CoV-2. The key mechanistic difference is that mRNA vaccines act in the cytoplasm only, while viral vector vaccines require nuclear entry for DNA transcription. This affects the speed of immune response onset.",
            "The lipid nanoparticles (LNPs) used in mRNA vaccines protect the mRNA from enzymatic degradation and facilitate cellular uptake via endocytosis. After endosomal escape, the mRNA is released into the cytoplasm for translation.",
            "Traditional inactivated virus vaccines present killed pathogens directly to the immune system. These do not require gene delivery but carry no causal mechanistic connection to spike protein expression via cellular machinery.",
        ],
        "description": "Medical domain: mechanism precision is causal (γ emphasized)",
    },

    "legal": {
        "domain": "legal",
        "query": "How did the Brown v. Board decision establish a legal precedent that was later applied to public universities?",
        "corpus": [
            "Brown v. Board of Education (1954) declared racially segregated public schools unconstitutional, overturning the 'separate but equal' doctrine from Plessy v. Ferguson (1896). The Supreme Court held that separate educational facilities are inherently unequal, violating the Equal Protection Clause of the 14th Amendment.",
            "Cooper v. Aaron (1958) extended the Brown mandate to all state-run institutions. The Supreme Court unanimously held that no state official could nullify or circumvent the desegregation orders. This directly established that the Brown principles applied to university-level segregation.",
            "Sweatt v. Painter (1950), decided just before Brown, required the University of Texas Law School to admit a Black applicant, establishing that separate graduate schools were inherently unequal. This case built the causal precedent chain that Brown would later consolidate.",
            "The Civil Rights Act of 1964 prohibited discrimination in public accommodations but was a legislative response to, not a cause of, the Brown precedent. It post-dates the judicial precedent established by Brown-Cooper-Sweatt.",
            "The First Amendment protects free speech and is a separate constitutional doctrine with no direct causal link to the equal protection precedent chain established in Brown.",
        ],
        "description": "Legal domain: precedent chains are causal (γ emphasized)",
    },

    "financial": {
        "domain": "financial",
        "query": "What was the causal chain that led from subprime mortgage lending to the collapse of Lehman Brothers?",
        "corpus": [
            "Subprime mortgage lenders issued loans to borrowers with poor credit histories, often with adjustable rates that reset to unaffordable levels. These loans were then securitized into mortgage-backed securities (MBS) and sold to investors seeking higher yields.",
            "Collateralized debt obligations (CDOs) were constructed from tranches of MBS, allowing risk to be redistributed but concentrated in lower-rated tranches. Rating agencies assigned high ratings to these instruments, mispricing their actual default risk.",
            "When housing prices peaked and began declining in 2006–2007, subprime borrowers began defaulting at high rates. The underlying value of MBS and CDO instruments collapsed, triggering mark-to-market losses for banks holding these assets.",
            "Lehman Brothers had accumulated massive exposure to toxic mortgage-backed assets. As confidence in its solvency collapsed, counterparties withdrew credit lines. Unable to meet its obligations, Lehman filed for bankruptcy on September 15, 2008.",
            "The Federal Reserve's response included emergency liquidity facilities and eventually quantitative easing. These are downstream policy responses, not part of the causal chain leading to Lehman's collapse.",
        ],
        "description": "Financial domain: balanced with causal emphasis",
    },

    "scientific": {
        "domain": "scientific",
        "query": "What distinguishes Type Ia from Type II supernovae in terms of their progenitor systems and explosion mechanisms?",
        "corpus": [
            "Type Ia supernovae arise from white dwarf stars in binary systems. The white dwarf accretes mass from a companion star until it reaches the Chandrasekhar mass limit (~1.4 solar masses), triggering a thermonuclear runaway. Carbon fusion ignites explosively, completely destroying the white dwarf with no remnant left.",
            "Type II supernovae result from the core collapse of massive stars (greater than 8 solar masses) that have exhausted their nuclear fuel. The iron core collapses under gravity in milliseconds, producing a proto-neutron star. The outer envelope rebounds and is expelled in the explosion.",
            "Type Ia supernovae are used as standard candles in cosmology because their peak luminosity is predictable. This property was exploited in 1998 to discover the accelerating expansion of the universe. This is an observational application, not a description of the progenitor mechanism.",
            "The observation that Type II supernovae produce neutron stars or black holes as remnants while Type Ia leave no compact remnant is a key distinguishing observational consequence of the different progenitor paths.",
            "Both supernova types enrich the interstellar medium with heavy elements, but Type Ia produce primarily iron-peak elements while Type II produce both iron-peak and r-process elements including many heavier elements.",
        ],
        "description": "Scientific domain: progenitor vs observable distinction (γ emphasized)",
    },

    "code": {
        "domain": "code",
        "query": "Why does using await outside an async function cause a SyntaxError rather than a RuntimeError in Python?",
        "corpus": [
            "In Python, the await keyword is syntactically restricted to async def function bodies. The Python parser (implemented in Parser/Python.asdl) enforces this constraint at compile time: if await appears outside an async context, the parser raises a SyntaxError before any bytecode is generated. This is a grammar-level constraint.",
            "Python's asyncio event loop raises RuntimeError when loop.run_until_complete() is called from within a running loop. This is a runtime check because the event loop state can only be known at execution time. Unlike await placement, this cannot be detected at parse time.",
            "The CPython compiler transforms Python source into an Abstract Syntax Tree (AST) in two stages: tokenization and parsing. Grammar rules in Grammar/Grammar enforce that await is only valid inside async def bodies. The constraint is hard-coded in the grammar, not in any runtime check.",
            "The async/await syntax was introduced in Python 3.5 via PEP 492. The decision to make await a syntactic constraint was deliberate: it allows static analysis tools, linters, and IDEs to detect incorrect usage without running the code.",
            "Generators in Python use the yield keyword which has similar syntactic restrictions — it must appear inside a function body. Both yield and await are compile-time syntactic constructs, not runtime-checked operations.",
        ],
        "description": "Code domain: AST/syntax structure dominates (β emphasized)",
    },

    "cybersecurity": {
        "domain": "cybersecurity",
        "query": "How does a SQL injection attack chain from initial input to database compromise?",
        "corpus": [
            "SQL injection begins when user-supplied input is concatenated directly into SQL queries without sanitization. An attacker inserts SQL metacharacters such as single quotes or comment sequences (--) to alter the query logic, causing the database to execute attacker-controlled SQL.",
            "Once SQL injection succeeds, attackers can use UNION-based injection to extract data from other tables, or use stacked queries to execute arbitrary SQL commands including DDL statements. Blind injection techniques infer data via boolean or time-based responses when output is not directly visible.",
            "Privilege escalation occurs when the database user account running the application has elevated permissions. An attacker who achieves SQL injection can escalate to reading arbitrary files (LOAD DATA INFILE), executing OS commands (xp_cmdshell in MSSQL), or creating new database accounts.",
            "Parameterized queries and prepared statements prevent SQL injection by separating SQL code from data. The database driver treats user input as a data value, never as executable SQL. This is the primary prevention mechanism.",
            "Cross-site scripting (XSS) is a different injection attack targeting client-side JavaScript. While both are injection vulnerabilities, XSS and SQL injection have different attack surfaces, mechanisms, and prevention strategies.",
        ],
        "description": "Cybersecurity domain: attack vectors are causally specific (β+γ balanced)",
    },

    "educational": {
        "domain": "educational",
        "query": "What is the step-by-step process by which a student learns to solve quadratic equations?",
        "corpus": [
            "Learning quadratic equations begins with understanding the standard form ax² + bx + c = 0. Students must first be comfortable with polynomial notation and the meaning of coefficients before attempting solution methods. Prior knowledge of linear equations is essential.",
            "Factoring quadratic equations requires students to find two numbers that multiply to give c and add to give b. This builds on multiplication and factor pair knowledge. Students learn to decompose the quadratic into a product of two linear factors.",
            "The quadratic formula x = (-b ± √(b²-4ac)) / 2a is introduced as a general method that always works. Students need to understand the discriminant (b²-4ac) to determine the number and type of solutions before applying the formula.",
            "Completing the square is an algebraic technique that transforms the quadratic into (x + h)² = k form. It develops algebraic manipulation skills and is the foundation for understanding the quadratic formula derivation.",
            "Graphical understanding connects the algebraic solutions to the x-intercepts of the parabola y = ax² + bx + c. Students who understand both representations develop deeper conceptual mastery than those who only memorize formulas.",
        ],
        "description": "Educational domain: topic coverage is most important (α emphasized)",
    },

    "historical": {
        "domain": "historical",
        "query": "What caused the fall of the Roman Empire in the West?",
        "corpus": [
            "Economic pressures, including inflation caused by currency debasement, trade disruptions, and the cost of maintaining a large military on long borders, weakened the Western Roman Empire's capacity to defend itself. Tax burdens increased while economic productivity declined.",
            "Military pressures from Germanic tribes including the Visigoths, Vandals, and eventually the Huns pushed large groups westward into Roman territory. The Battle of Adrianople (378 CE) was a turning point in which Gothic forces decisively defeated a Roman army.",
            "Political instability, including frequent imperial succession crises and the division of the empire into Eastern and Western halves, weakened central authority. The Eastern Roman (Byzantine) Empire survived for another thousand years, suggesting the fall was not inevitable but structurally linked to specific Western conditions.",
            "The sack of Rome by the Visigoths in 410 CE was a symbolic event but not the direct cause of collapse. The final deposition of Romulus Augustulus in 476 CE by Odoacer is conventionally dated as the end, but the empire had been functionally disintegrating for decades.",
            "Climate change and pandemic disease (the Antonine Plague, Plague of Cyprian) reduced population and agricultural productivity in earlier centuries, weakening the empire's demographic and economic base over the long term.",
        ],
        "description": "Historical domain: causation-through-time matters (γ moderate)",
    },

    "customer": {
        "domain": "customer",
        "query": "My Python script crashes with AttributeError when I try to call .read() on a file I just opened. What is wrong?",
        "corpus": [
            "An AttributeError on .read() typically occurs when the variable storing the file is not an actual file object. The most common cause is opening the file in write mode ('w') rather than read mode ('r'). In write mode, calling .read() will raise AttributeError because write-mode file objects don't expose a read interface in some frameworks.",
            "Another common cause is accidentally closing the file before reading it. If you call file.close() or use a context manager (with open(...) as f:) and then try to read outside the with block, the file object is closed and .read() will raise ValueError: I/O operation on closed file.",
            "If the variable name shadows the built-in open function or another variable, you may be calling .read() on a string or None instead of a file object. Check for name conflicts like open = something_else earlier in your code.",
            "For binary files opened with 'rb' mode, .read() returns bytes, not a string. If you try to use string methods on bytes, you'll get AttributeError. Use .decode('utf-8') to convert bytes to string.",
            "Feature request: customers have asked for a smart open() wrapper that automatically detects mode and provides better error messages. This is tracked in issue #4521.",
        ],
        "description": "Customer support domain: intent matching is flexible (α emphasized)",
    },

    "creative": {
        "domain": "creative",
        "query": "What themes of transformation and decay connect Keats' Ode to a Nightingale with Shelley's Ozymandias?",
        "corpus": [
            "Keats' Ode to a Nightingale explores the tension between the immortal song of the bird and the speaker's awareness of human mortality and decay. The nightingale's song has been heard by emperors and clowns throughout history, transcending individual human lifespans. This immortality through art is contrasted with the speaker's physical deterioration.",
            "Shelley's Ozymandias presents a collapsed statue in a desert, its boastful inscription ('Look on my works, ye mighty, and despair!') now surrounded by empty sand. The poem captures the decay of political power and material legacy, showing that even the greatest human achievements are subject to time and ruin.",
            "Both poems engage the Romantic preoccupation with transience: what endures beyond human life? For Keats, natural beauty and its expression in art outlast individual humans. For Shelley, even monumental human creation is subject to decay, suggesting that political power is even more transient than natural forms.",
            "The figure of the poet in both works is positioned as an observer of transcendence and decay — witnessing beauty that outlasts the self (Keats) or absence that testifies to pride's failure (Shelley). Both poems use the speaker's subjective response to mediate between human limitation and the indifferent sublime.",
            "Mary Shelley's Frankenstein also explores transformation — from human to creator to destroyer — showing the Romantic engagement with transformation as a broader theme across the genre, not unique to these two poems.",
        ],
        "description": "Creative domain: semantic exploration dominates (α=0.50)",
    },

    "general": {
        "domain": "general",
        "query": "How does photosynthesis in plants produce glucose from carbon dioxide and water?",
        "corpus": [
            "Photosynthesis occurs in two stages in plant chloroplasts. The light-dependent reactions in the thylakoid membranes capture light energy and convert it to ATP and NADPH, splitting water molecules and releasing oxygen as a byproduct.",
            "The light-independent Calvin cycle takes place in the stroma of the chloroplast. It uses the ATP and NADPH from the light reactions to fix atmospheric CO₂ into three-carbon compounds, which are then reduced to glyceraldehyde-3-phosphate (G3P) and eventually converted to glucose.",
            "The overall reaction is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. Carbon dioxide enters through stomata, water is absorbed from roots, and light is captured by chlorophyll in the thylakoid membranes.",
            "Chlorophyll absorbs primarily red and blue wavelengths of light while reflecting green, which is why plants appear green. Different photosystems (I and II) operate in series to generate sufficient energy for both cyclic and non-cyclic electron flow.",
            "Plants also perform cellular respiration to break down glucose when light is unavailable, essentially reversing photosynthesis to release energy for cellular processes. Respiration occurs in mitochondria, not chloroplasts.",
        ],
        "description": "General domain: balanced α=0.40, β=0.30, γ=0.30",
    },
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_domain_example(domain_key: str, verbose: bool = False) -> dict:
    """Run a single domain example and return results."""
    example = DOMAIN_EXAMPLES[domain_key]

    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain_key.upper()}")
    print(f"Description: {example['description']}")
    print(f"Query: {example['query'][:80]}...")
    print(f"Corpus: {len(example['corpus'])} chunks")
    print("-" * 70)

    config = VortexRAGConfig(domain=example["domain"], verbose=verbose)
    config.vrc.candidate_pool = len(example["corpus"]) * 2
    config.vrc.top_k = min(len(example["corpus"]), 10)
    config.rfg.top_m = min(5, len(example["corpus"]))

    rag = VortexRAG(config=config, llm_fn=lambda ctx, q: ctx.split("\n\n")[0][:300] if ctx else "No context.")
    rag.index(additional_texts=example["corpus"])
    result = rag.query(example["query"])

    print(f"Results:")
    print(f"  Spiral pool size:  {result.spiral_pool_size}")
    print(f"  CPG purge count:   {result.purge_count}")
    print(f"  ESR:               {result.esr:.4f}")
    print(f"  Delta_R:           {result.delta_r:.4f}")
    print(f"  Accepted:          {result.accepted}")
    print(f"  Latency:           {result.latency_ms:.1f}ms")
    print(f"  Context chunks:    {len(result.context_window)}")
    if result.phi_scores:
        print(f"  Top Phi score:     {max(result.phi_scores):.4f}")

    if result.context_window:
        print(f"\n  Top context chunk (first 120 chars):")
        print(f"  | {result.context_window[0][:120].replace(chr(10), ' ')}...")

    return {
        "domain": domain_key,
        "query": example["query"],
        "esr": result.esr,
        "delta_r": result.delta_r,
        "accepted": result.accepted,
        "purge_count": result.purge_count,
        "context_chunks": len(result.context_window),
        "latency_ms": result.latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="VORTEXRAG Domain-Specific Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available domains: {', '.join(DOMAIN_EXAMPLES.keys())}",
    )
    parser.add_argument(
        "--domain",
        choices=list(DOMAIN_EXAMPLES.keys()) + ["all"],
        default="all",
        help="Domain to demonstrate (or 'all' for all 11)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable pipeline verbose output")
    args = parser.parse_args()

    domains_to_run = list(DOMAIN_EXAMPLES.keys()) if args.domain == "all" else [args.domain]

    print("VORTEXRAG Domain-Specific Examples")
    print(f"Running {len(domains_to_run)} domain(s)...")

    all_results = []
    for domain in domains_to_run:
        try:
            result = run_domain_example(domain, verbose=args.verbose)
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] Domain '{domain}' failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY — All 11 Domain Results")
        print(f"{'Domain':<18} {'ESR':>8} {'Delta_R':>9} {'Accepted':>10} {'Latency':>10}")
        print("-" * 70)
        for r in all_results:
            accepted_str = "YES" if r["accepted"] else "NO"
            print(f"  {r['domain']:<16} {r['esr']:>8.4f} {r['delta_r']:>9.4f} "
                  f"{accepted_str:>10} {r['latency_ms']:>8.1f}ms")

        total_lat = sum(r["latency_ms"] for r in all_results)
        print(f"\n  Total latency: {total_lat:.1f}ms across {len(all_results)} domains")


if __name__ == "__main__":
    main()
