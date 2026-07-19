"""Pathological Myasthenia Gravis (MG) Boundary Stress-Test.

This module models true biological Myasthenia Gravis: post-synaptic receptor 
destruction leading to a severely dropped quantal size (q=30 pA) combined with 
compensatory structural remodeling (high n_sites=12), buried under a high 
noise floor. It evaluates how well the BQA grid inference handles extreme 
biological signal-to-noise degradation.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.simulate import simulate_responses


def run_pathological_mg_analysis():
    """Runs the biological MG simulation and evaluates it via the BQA engine."""
    print("\n--- Starting Pathological Myasthenia Gravis Stress Test ---")
    
    # Configuration matching fixed seed protocols for ReScience validation
    rng = np.random.default_rng(42)
    
    # True Pathological Parameters
    n_sites = 12          # Structural remodeling (high site count)
    q_true = 30.0         # Severe receptor loss (low quantal size, down from 100)
    cv_intra = 0.3        
    baseline_sd = 25.0    # High noise floor relative to the weak 30 pA signal
    n_obs = 100           # Observations per condition
    
    # Test across a low and a high release probability condition
    p1, p2 = 0.15, 0.45 
    eps2 = baseline_sd**2
    
    conditions = []
    for p_true in (p1, p2):
        # Dynamically draw from your partner's Gaussian simulator engine
        sim = simulate_responses(
            n_sites=n_sites,
            p=p_true,
            q=q_true,
            cv_intra=cv_intra,
            sigma_baseline=baseline_sd,
            n_obs=n_obs,
            rng=rng
        )
        mu_true = n_sites * p_true * q_true
        conditions.append(ConditionData(mu=mu_true, eps2=eps2, amplitudes=sim.amplitudes))

    # Define candidate search grid boundaries for BQA around the true n=12
    n_candidates = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    grid = build_grid(n_values=np.array(n_candidates))
    
    print("Running BQA grid search over pathological data matrix...")
    posterior = change_of_variables_and_marginalise(conditions, grid, n_candidates)
    
    # Package the results
    results = {
        "Metric": ["n_true", "n_hat", "q_true", "q_hat", "v_hat"],
        "Value": [n_sites, posterior["n_hat"], q_true, posterior["q_hat"], posterior["v_hat"]]
    }
    df = pd.DataFrame(results)
    
    # Ensure output directory exists and save the results
    os.makedirs("results/identifiability", exist_ok=True)
    output_path = "results/identifiability/pathological_mg_results.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Pathological analysis complete! Summary saved to: {output_path}")
    print(df.to_string(index=False))
    print("-----------------------------------------------------------\n")


if __name__ == "__main__":
    run_pathological_mg_analysis()
