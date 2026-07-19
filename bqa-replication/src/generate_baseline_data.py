import numpy as np
import pandas as pd
import os

# Set seed for exact computational determinism (ReScience Requirement)
np.random.seed(42)

# ==========================================
# REPLICATION MODE SELECTION
# ==========================================
# Options: 'fig2_sweep', 'fig4_3cond', 'fig4_converse', 'fig4_size', 'fig5_noise', 'fig5_n_sweep'
RUN_MODE = 'fig2_sweep'  

# --- Global Fixed Parameters ---
NUM_SIMULATIONS = 100
N_control = 6
q_control = 100.0
cv_control = 0.30
noise_control = 25.0  # 25% of baseline quantal size

all_rows = []

# ==========================================
# MODE GENERATOR LOGIC
# ==========================================
if RUN_MODE == 'fig2_sweep':
    print("Baseline: Executing Fig 2 Standard Sweep...")
    p2_steps = np.round(np.arange(0.05, 0.81, 0.01), 2)
    conditions = [{'p1': 0.10, 'p2': p2, 'N': N_control, 'trials': 60, 'noise': noise_control} for p2 in p2_steps]
    meta_cols = ['Sim_ID', 'P2_Val', 'Trial_ID', 'Cond1_Amp', 'Cond2_Amp']

elif RUN_MODE == 'fig4_3cond':
    print("Baseline: Executing Fig 4 Three-Condition Sweep...")
    p2_steps = np.round(np.arange(0.05, 0.81, 0.01), 2)
    conditions = [{'p1': 0.10, 'p2': p2, 'p3': np.round((0.10 + p2)/2, 3), 'N': N_control, 'trials': 60, 'noise': noise_control} for p2 in p2_steps]
    meta_cols = ['Sim_ID', 'P2_Val', 'Trial_ID', 'Cond1_Amp', 'Cond2_Amp', 'Cond3_Amp']

elif RUN_MODE == 'fig4_converse':
    print("Baseline: Executing Fig 4 Converse Sweep...")
    p1_steps = np.round(np.arange(0.10, 0.41, 0.01), 2)
    conditions = [{'p1': p1, 'p2': np.round(p1 + 0.5, 2), 'N': N_control, 'trials': 60, 'noise': noise_control} for p1 in p1_steps]
    meta_cols = ['Sim_ID', 'P1_Val', 'Trial_ID', 'Cond1_Amp', 'Cond2_Amp']

elif RUN_MODE == 'fig4_size':
    print("Baseline: Executing Fig 4 Sample Size Sweep...")
    size_steps = list(range(30, 122, 2))
    conditions = [{'p1': 0.10, 'p2': 0.60, 'N': N_control, 'trials': size, 'noise': noise_control} for size in size_steps]
    meta_cols = ['Sim_ID', 'Trial_Size', 'Trial_ID', 'Cond1_Amp', 'Cond2_Amp']

elif RUN_MODE == 'fig5_noise':
    print("Baseline: Executing Fig 5 Baseline Noise Sweep...")
    noise_pcts = np.round(np.arange(0.10, 0.65, 0.05), 2)
    conditions = [{'p1': 0.10, 'p2': 0.60, 'N': N_control, 'trials': 60, 'noise': (pct * q_control)} for pct in noise_pcts]
    meta_cols = ['Sim_ID', 'Noise_Level', 'Trial_ID', 'Cond1_Amp', 'Cond2_Amp']

elif RUN_MODE == 'fig5_n_sweep':
    print("Baseline: Executing Fig 5 N Structural Sweep...")
    n_steps = list(range(3, 13))
    conditions = [{'p1': 0.10, 'p2': 0.60, 'N': n, 'trials': 60, 'noise': noise_control} for n in n_steps]
    meta_cols = ['Sim_ID', 'N_Sites', 'Trial_ID', 'Cond1_Amp', 'Cond2_Amp']

# ==========================================
# CORE SIMULATION ENGINE
# ==========================================
for c in conditions:
    # Identify unique identifier for indexing summary plots
    if RUN_MODE == 'fig2_sweep': tracker = c['p2']
    elif RUN_MODE == 'fig4_3cond': tracker = c['p2']
    elif RUN_MODE == 'fig4_converse': tracker = c['p1']
    elif RUN_MODE == 'fig4_size': tracker = c['trials']
    elif RUN_MODE == 'fig5_noise': tracker = c['noise']
    elif RUN_MODE == 'fig5_n_sweep': tracker = c['N']

    for sim_id in range(1, NUM_SIMULATIONS + 1):
        num_t = c['trials']
        
        # Binomial vesicle release count calculations
        rel_c1 = np.random.binomial(c['N'], c['p1'], num_t)
        rel_c2 = np.random.binomial(c['N'], c['p2'], num_t)
        if RUN_MODE == 'fig4_3cond': rel_c3 = np.random.binomial(c['N'], c['p3'], num_t)
        
        # Instrument Background Static Noise
        ns_c1 = np.random.normal(0, c['noise'], num_t)
        ns_c2 = np.random.normal(0, c['noise'], num_t)
        if RUN_MODE == 'fig4_3cond': ns_c3 = np.random.normal(0, c['noise'], num_t)
        
        # Quantal Amplitudes (Gamma vesicular variation + Gaussian noise floor)
        shape = 1.0 / (cv_control**2)
        scale = q_control / shape
        
        for t in range(num_t):
            amp_c1 = np.sum(np.random.gamma(shape, scale, rel_c1[t])) + ns_c1[t] if rel_c1[t] > 0 else ns_c1[t]
            amp_c2 = np.sum(np.random.gamma(shape, scale, rel_c2[t])) + ns_c2[t] if rel_c2[t] > 0 else ns_c2[t]
            
            if RUN_MODE == 'fig4_3cond':
                amp_c3 = np.sum(np.random.gamma(shape, scale, rel_c3[t])) + ns_c3[t] if rel_c3[t] > 0 else ns_c3[t]
                all_rows.append([sim_id, tracker, t + 1, round(amp_c1, 4), round(amp_c2, 4), round(amp_c3, 4)])
            else:
                all_rows.append([sim_id, tracker, t + 1, round(amp_c1, 4), round(amp_c2, 4)])

# Export directly to data/ directory
df = pd.DataFrame(all_rows, columns=meta_cols)
df.to_csv(f"master_synthetic_data_baseline_{RUN_MODE}.csv", index=False)
print(f"COMPLETE: Baseline file exported successfully ({len(df)} rows).")
