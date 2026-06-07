#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# compile.sh  —  Build vortexrag.pdf from vortexrag.tex
#
# Usage:
#   cd /Users/vignesh/VORTEXRAG
#   bash paper/compile.sh
#
# Requirements:
#   pdflatex  (TeX Live 2022+ or MiKTeX)
#   bibtex    (bundled with any TeX Live / MiKTeX installation)
#
# Optional — faster incremental recompiles:
#   latexmk -pdf paper/vortexrag.tex
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_FILE="$SCRIPT_DIR/vortexrag.tex"
BUILD_DIR="$SCRIPT_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$TEX_FILE" ]; then
  echo "ERROR: $TEX_FILE not found."
  exit 1
fi

command -v pdflatex >/dev/null 2>&1 || {
  echo "ERROR: pdflatex not found. Install TeX Live:"
  echo "  macOS   : brew install --cask mactex"
  echo "  Ubuntu  : sudo apt install texlive-full"
  exit 1
}

# ── Compilation sequence ──────────────────────────────────────────────────────
echo "==> Pass 1/3  (pdflatex — initial compile)"
pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$TEX_FILE"

echo "==> Pass 2/3  (bibtex — process bibliography)"
cd "$BUILD_DIR"
bibtex vortexrag || true   # bibtex may warn on inline bib; non-fatal

echo "==> Pass 3/4  (pdflatex — resolve citations)"
pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$TEX_FILE"

echo "==> Pass 4/4  (pdflatex — finalise cross-references)"
pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$TEX_FILE"

# ── Result ────────────────────────────────────────────────────────────────────
if [ -f "$BUILD_DIR/vortexrag.pdf" ]; then
  PDF_SIZE=$(du -sh "$BUILD_DIR/vortexrag.pdf" | cut -f1)
  echo ""
  echo "======================================================"
  echo "  SUCCESS — vortexrag.pdf  ($PDF_SIZE)"
  echo "  Location: $BUILD_DIR/vortexrag.pdf"
  echo "======================================================"
  # Open PDF on macOS if running interactively
  if [[ "$(uname)" == "Darwin" ]] && [[ -t 1 ]]; then
    open "$BUILD_DIR/vortexrag.pdf"
  fi
else
  echo "ERROR: PDF not generated — check vortexrag.log for details."
  exit 1
fi

# ── Optional cleanup ──────────────────────────────────────────────────────────
# Uncomment to remove auxiliary files after a successful build:
# rm -f "$BUILD_DIR"/{vortexrag.aux,vortexrag.bbl,vortexrag.blg,\
#                     vortexrag.log,vortexrag.out,vortexrag.toc}
