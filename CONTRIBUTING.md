# Contributing to VORTEXRAG

Thank you for your interest in contributing! Every improvement helps make RAG systems more faithful.

## Ways to contribute

- **Bug reports** — open an issue with the bug report template
- **Feature requests** — open an issue with the feature request template
- **Code contributions** — fork the repo, make changes, open a PR
- **Documentation** — improve README, examples, or docstrings
- **Benchmarks** — run VORTEXRAG on new datasets and share results

## Getting started

```bash
git clone https://github.com/vignesh2027/VORTEXRAG
cd VORTEXRAG
pip install -r requirements.txt
python3 -m pytest tests/ -q   # all 229 should pass
```

## Before submitting a PR

1. **Tests must pass**: `python3 -m pytest tests/ -q`
2. **Lint must be clean**: `ruff check core/ vortexrag.py --select=E,F --ignore=E501,F401`
3. **Add tests** for any new behaviour
4. **Keep the math correct** — any change to a layer formula must match the paper (DOI: 10.5281/zenodo.20579702)

## Project structure

```
core/
  tve.py   # Layer 1 — Tri-Vector Encoding
  vrc.py   # Layer 2 — Vortex Retrieval Cone
  sdc.py   # Layer 3 — Semantic Drift Corrector
  cpg.py   # Layer 4 — Context Poison Guard
  rfg.py   # Layer 5 — Rank Fusion Gate
  ccb.py   # Layer 6 — Causal Context Builder
  fv.py    # Layer 7 — Faithfulness Verifier
vortexrag.py   # Top-level pipeline
tests/         # 229 unit + integration tests
paper/         # LaTeX source + compiled PDF
```

## Recognition

All contributors are listed in the GitHub contributors graph. Significant contributions will be acknowledged in the project README.

## Questions?

Open a [Question issue](https://github.com/vignesh2027/VORTEXRAG/issues/new?template=question.md) — happy to help.
