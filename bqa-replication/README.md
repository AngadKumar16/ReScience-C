# bqa-replication

Independent clean-room reimplementation of Bhumbra & Beato (2013),
"Reliable evaluation of the quantal determinants of synaptic efficacy using
Bayesian analysis," *J Neurophysiol* 109(2):603-620,
DOI [10.1152/jn.00528.2012](https://doi.org/10.1152/jn.00528.2012).
Target venue: ReScience C.

## Contribution

Reproduction of the original Bayesian Quantal Analysis (BQA) is the
*foundation* here, not the headline. The actual contributions of this
replication are:

1. **Independent verification** of the BQA grid-inference procedure against
   the paper's own figures, written from the paper text alone.
2. **Simulation-based calibration (SBC)** of the inference procedure --
   something the original paper never performed. SBC checks that BQA
   recovers correctly calibrated posteriors when data are generated from
   the model's own likelihood.
3. **An identifiability map** over release-probability contrasts (DeltaP
   sweep) -- also absent from the original paper -- characterizing where
   BQA can and cannot resolve its parameters.

## Clean-room boundary

This implementation is written from the published paper (all 18 pages,
including the Appendix) only. It does **not** fetch, read, or vendor the
original authors' code ([Pyclamp](https://github.com/Bhumbra/Pyclamp),
[probayes](https://github.com/Bhumbra/probayes)). All originally-scoped
pieces of the method -- including the Eq. 8 gamma quantal likelihood
(`src/q_model.py::q_function`) and the Appendix A1-A13 grid
change-of-variables and marginalisation (`src/bqa.py`) -- are now
implemented, transcribed directly from the paper text; see `NOTES.md` for
the transcribed equations and `reproduction_map.md` for the full
element-to-code mapping, including known limitations and one place (SBC's
prior over q) where a necessary modeling convention was added that is not
in the paper, clearly labeled as such.

No numeric results from the paper are hardcoded anywhere in this repo.

## The model (summary)

BQA infers (p1, p2, v, n) by **brute-force grid search**, not MCMC:

- Grid axes: `arcsin(sqrt(p))` at resolution 128, `log_e(v)` at resolution
  128, and integer `n` enumerated (see `src/grid.py`).
- Priors: arcsine/Jeffreys on p (0.04-0.96), uniform on log(v) (0.05-1),
  and **uniform** on integer n -- the 2013 paper's choice; the authors'
  later 2018/2019 work switches to a Jeffreys' prior on n, which is a
  different model and is deliberately not used here.
- Condition means and baseline-noise variance are fixed, known inputs
  computed from data -- not sampled parameters.
- q, lambda, and r are derived quantities, not free parameters. n is
  estimated as `median(r) / median(q)`.

See `NOTES.md` for the full list of confirmed constants and open items, and
`reproduction_map.md` for a paper-element-to-code-location table.

## The two-simulator rule

This repo uses **two distinct simulators that must never be conflated**:

| Simulator | File | Noise model | Used for |
|---|---|---|---|
| `simulate_responses` | `src/simulate.py` | Gaussian intrasite noise | Reproducing the paper's figures only |
| `simulate_from_q_model` | `src/simulate_q.py` | Model's own Eq. 8 gamma likelihood | SBC only |

SBC is only valid if synthetic data are generated from the model's *own*
likelihood. Using `simulate_responses` for SBC would measure
Gaussian-vs-gamma model misspecification, not inference error of the BQA
grid procedure -- so `src/simulate_q.py` and `src/sbc.py` use
`simulate_from_q_model` exclusively, and `src/figure1.py` /
figure-reproduction code uses `simulate_responses` exclusively.

## Feasibility gate

Before any inference pipeline is built or trusted, `src/figure1.py`
reproduces Fig 1A-D, which is fully analytic from stated parameters:

- **A**: Binomial(n=6, p=0.35) release-count distribution.
- **B**: Normal(0, SD=25 pA) baseline noise.
- **C**: Gamma(shape=11.1, rate/scale=9) quantal amplitude (q ~ 100 pA,
  CV ~ 0.30).
- **D**: the combined quantal likelihood Q, via `q_model.q_function`.

**Go/no-go rule:** Panel D is the checkpoint. If the Q-function
implementation cannot be made to reproduce Fig 1D, **stop** before building
the grid-search inference pipeline in `src/bqa.py` -- a mismatch there means
the Eq. 8 implementation or its parameterisation is wrong, and any
inference machinery built on top of it would only compound the error.
Run `python -m src.figure1` to check gate status.

The gate **passes a numeric check**: `figure1.gate_peak_check` locates the
modes of Q(x) and requires each interior mode within 15% of an integer
multiple of q. The modes land at 91.8 and 193.8 pA (q and 2q), max relative
error 8.1%, asserted in `tests/test_consistency.py`. It is still not checked
pixel-for-pixel against the published image (no digitized reference curve
was available).

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

or via conda: `conda env create -f environment.yml`.

## Running

```bash
bash reproduce.sh
```

Runs the test suite (`pytest -q`, 23 tests) followed by the full
figure/analysis pipeline (`python -m src.figures`): the numeric feasibility
gate, a 200-iteration SBC calibration run, the 76-point identifiability
sweep, the Gaussian-vs-gamma robustness illustration, the BQA-vs-MPFA
comparison, the grid-resolution bias check, and the low-SNR stress test.
Each stage uses an independently seeded generator and writes a CSV to
`results/`. Expect a few minutes (the SBC run and identifiability sweep
dominate).

The powered SBC run reports that the q and r posteriors are NOT calibrated
(biased low); this is a real finding, not a bug, and is discussed in
`paper/content.md`.

## Repository layout

```
src/simulate.py          Gaussian intrasite simulator (figures only)
src/q_model.py            Eq. 8 Q-function
src/simulate_q.py         Model's-own-likelihood simulator (SBC only)
src/grid.py                Grid axes + transforms + Jacobians
src/bqa.py                  Per-condition likelihood, combination, A1-A13, heterogeneous-release pmf
src/figure1.py               Fig 1A-D, feasibility gate
src/sbc.py                    SBC loop + rank/PIT helpers + uniformity test
src/identifiability.py         DeltaP identifiability sweep (novel analysis)
src/mg_illustration.py          Gaussian-vs-gamma mismatch robustness illustration
src/mpfa.py                      MPFA baseline + BQA-vs-MPFA comparison (novel)
src/resolution_check.py          Grid-resolution bias check (novel)
src/pathological_mg.py            Low signal-to-noise stress test (novel)
src/figures.py                     Top-level pipeline entry point
tests/                              test_simulate/grid/q_model/bqa/consistency/mpfa.py
results/{calibration,identifiability}/  Output directories
paper/                                    ReScience C submission materials
```

See `reproduction_map.md` for the full paper-element-to-code map (with
per-item status/caveats) and `NOTES.md` for confirmed constants, settled
prior choices, transcribed equations, and known limitations.

## License

MIT, see `LICENSE`.
