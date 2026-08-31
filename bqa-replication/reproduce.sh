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

# PIT diagnostics (PIT_DEFECT.md): the three independent confirmations that
# the earlier quantal-size failure was a defect in our own rank statistic.
echo
echo "Running PIT diagnostics..."
python -m src.pit_diagnostics

# Localisation of the residual: how the quantal-size rank statistic responds
# to the width of the candidate-n set (Section 4.1.1).
echo
echo "Running n-marginalisation sweep..."
python -m src.n_marginalisation_check

# Is conditioning on a data-derived mu responsible? True mean vs sample mean,
# same draws, same amplitudes (Section 4.1.1).
echo
echo "Running mu-conditioning check..."
python -m src.mu_conditioning_check

# Alternative shared-q priors and the resolution-256 run behind Table 3.
echo
echo "Running joint-SBC prior and resolution variants..."
for prior in geometric induced flat; do
  python -m src.run_joint_sbc joint "$prior" 1e9
  python -m src.run_joint_sbc joint "$prior" finalize
done
BQA_SBC_RES=256 python -m src.run_joint_sbc joint flat 1e9
BQA_SBC_RES=256 python -m src.run_joint_sbc joint flat finalize

# Low signal-to-noise twelve-seed ensemble (Section 4.5).
echo
echo "Running low signal-to-noise seed ensemble..."
python -m src.low_snr_seeds
