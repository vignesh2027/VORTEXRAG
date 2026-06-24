# Contributing to VORTEXRAG

Thanks for wanting to contribute! Here's everything you need to know.

## Branch structure

```
main          ← stable, protected — only reviewed PRs merge here
dev           ← integration branch — your PR targets this
feat/*        ← feature branches (e.g. feat/beir-benchmark)
fix/*         ← bug fix branches (e.g. fix/sdc-threshold)
```

**Always open your PR against `dev`, not `main`.** Once reviewed and merged into `dev`, it gets merged into `main` from there.

If you're working on an open issue:
1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Make your changes + add tests
4. Open a PR targeting the `dev` branch
5. Comment on the issue so it gets assigned to you

## Getting started

```bash
git clone https://github.com/vignesh2027/VORTEXRAG
cd VORTEXRAG
pip install -r requirements.txt
python3 -m pytest tests/ -q   # 247 tests should pass
```

## Before submitting a PR

1. Tests must pass: `python3 -m pytest tests/ -q`
2. Lint must be clean: `ruff check core/ vortexrag.py --select=E,F --ignore=E501,F401`
3. Add tests for any new behaviour
4. If you change a layer formula, it must match the paper (DOI: [10.5281/zenodo.20579702](https://doi.org/10.5281/zenodo.20579702))

## Project structure

```
core/
  tve.py        # Layer 1 — Tri-Vector Encoding
  vrc.py        # Layer 2 — Vortex Retrieval Cone
  sdc.py        # Layer 3 — Semantic Drift Corrector
  cpg.py        # Layer 4 — Context Poison Guard
  rfg.py        # Layer 5 — Rank Fusion Gate
  ccb.py        # Layer 6 — Causal Context Builder
  fv.py         # Layer 7 — Faithfulness Verifier
vortexrag.py    # Top-level pipeline
integrations/   # LangChain, LlamaIndex wrappers
benchmarks/     # BEIR and other eval scripts
tests/          # 247 unit + integration tests
paper/          # LaTeX source + PDF
```

## Open issues for new contributors

| Issue | Branch to target | Difficulty |
|-------|-----------------|-----------|
| [BEIR benchmark script](https://github.com/vignesh2027/VORTEXRAG/issues/3) | `feat/beir-benchmark` | Good first issue |
| [LangChain integration](https://github.com/vignesh2027/VORTEXRAG/issues/4) | `feat/langchain-integration` | Good first issue |
| [Biomedical domain preset](https://github.com/vignesh2027/VORTEXRAG/issues/5) | `dev` | Easy |

## Recognition

Every merged contributor shows up in the [GitHub contributors graph](https://github.com/vignesh2027/VORTEXRAG/graphs/contributors).

## Questions?

Open a [Question issue](https://github.com/vignesh2027/VORTEXRAG/issues/new?template=question.md).
