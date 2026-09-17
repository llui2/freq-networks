#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "${1:-}" != "--after-pull" ]]; then
    git pull --ff-only
    exec bash "$0" --after-pull
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Python was not found." >&2
    exit 1
fi

"$PYTHON" scripts/fig1_edge_spectra.py
"$PYTHON" scripts/fig2_hierarchy.py
"$PYTHON" scripts/fig3_shift.py

cd paper

if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi

printf '\nBuilt %s\n' "$ROOT/paper/main.pdf"
