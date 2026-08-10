"""
Policy Interpretability and Behavioral Analysis Script.

Analyzes how the Actor-Critic policy behaves across different water classes,
inspecting the learned correlation between predictive uncertainty and dynamic neighborhood size (k_i).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets_loader.water_benchmarks import load_water_benchmark, WATER_BENCHMARK_CONFIGS
from solvers.rl_transductive_solver import RLHydroTransductiveSolver
from experiments.run_water_experiments import train_fast_policy
from rl_core.hydro_env import HydroTransductiveEnv


def analyze_policy_behavior(
    dataset_key: str = 'eurosat_water',
    shots: int = 2,
    seed: int = 42,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    print(f"================================================================")
    print(f"  POLICY BEHAVIORAL & INTERPRETABILITY ANALYSIS: {dataset_key.upper()}")
    print(f"================================================================\n")

    data = load_water_benchmark(dataset_key, shots=shots, seed=seed, device=device)
    s_opt = data['support_opt']
    s_sar = data['support_sar']
    s_lbl = data['support_labels']
    v_opt = data['val_opt']
    v_sar = data['val_sar']
    v_lbl = data['val_labels']
    q_opt = data['query_opt']
    q_sar = data['query_sar']
    q_lbl = data['query_labels']
    clip_w = data['clip_weights']
    class_names = data['class_names']

    policy = train_fast_policy(s_opt, s_lbl, v_opt, v_lbl, clip_w, extra_features=v_sar, device=device)
    solver = RLHydroTransductiveSolver(policy=policy)

    probs, metrics = solver.solve(s_opt, s_lbl, q_opt, clip_w, extra_features=q_sar)

    k_chosen = solver.k_per_sample.cpu().numpy()
    lambda_chosen = solver.lambda_lc_per_sample.squeeze().cpu().numpy()
    weights_chosen = solver.modality_weights.cpu().numpy()
    q_lbl_np = q_lbl.cpu().numpy()

    # Per-class analysis
    per_class_k = {}
    for idx, cname in enumerate(class_names):
        c_mask = (q_lbl_np == idx)
        per_class_k[cname] = {
            'mean_k': float(np.mean(k_chosen[c_mask])),
            'mean_lambda_lc': float(np.mean(lambda_chosen[c_mask])),
            'mean_opt_weight': float(np.mean(weights_chosen[c_mask, 0])),
            'mean_sar_weight': float(np.mean(weights_chosen[c_mask, 1])),
            'sample_count': int(np.sum(c_mask)),
        }
        print(f"  Class: {cname:<25} -> Mean k: {per_class_k[cname]['mean_k']:4.2f} | Mean lambda: {per_class_k[cname]['mean_lambda_lc']:4.2f} | Opt/SAR: {per_class_k[cname]['mean_opt_weight']:.2f}/{per_class_k[cname]['mean_sar_weight']:.2f}")

    analysis_data = {
        'dataset': dataset_key,
        'shots': shots,
        'overall_mean_k': float(np.mean(k_chosen)),
        'k_distribution': {str(int(k)): int(np.sum(k_chosen == k)) for k in [1, 3, 5, 8, 12, 16]},
        'per_class_stats': per_class_k,
    }

    out_file = './caches/policy_analysis_summary.json'
    with open(out_file, 'w') as fh:
        json.dump(analysis_data, fh, indent=2)
    print(f"\nSaved policy analysis to {out_file}\n")
    return analysis_data


if __name__ == '__main__':
    analyze_policy_behavior()
