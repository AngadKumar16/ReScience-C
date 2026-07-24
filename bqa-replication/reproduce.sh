#!/usr/bin/env bash
# Reproduce all figures/analyses end to end (see src/figures.py).
# The identifiability sweep (76 grid-inference calls) and the SBC runs
# (including the joint-grid vs product comparison, Section 4.1) dominate
# runtime; expect a few minutes total.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Running test suite..."
pytest -q

echo
echo "Running figure/analysis pipeline..."
python -m src.figures
