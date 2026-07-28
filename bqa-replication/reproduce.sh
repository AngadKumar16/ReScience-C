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

# Conditions-scaling harness (paper Table 1). Runs all four combination rules
# on identical draws with an independent per-iteration seed.
echo
echo "Running conditions-scaling harness (Table 1)..."
python -m src.conditions_scaling batch "product,loglik,density,joint" 1e9
python -m src.conditions_scaling finalize

# Calibration self-consistency checks (paper Section 4.1.1). These are what
# establish that the interval defect is upstream of the combination step, so
# they belong in the end-to-end run rather than in a standalone script.
echo
echo "Running SBC self-consistency checks (Section 4.1.1)..."
for prior in induced geometric flat; do
  python -m src.sbc_self_consistency batch "$prior" 1e9
done
python -m src.sbc_self_consistency finalize
python -m src.sbc_fixed_n_check

# Grid-resolution sweep on the wide n-candidate set over a seed ensemble
# (paper Section 4.5).
echo
echo "Running grid-resolution sweep (Section 4.5)..."
python -m src.resolution_check
