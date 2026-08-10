"""
Master GPU Benchmark Runner for Water Resources & Flood Monitoring Foundation Models.

Evaluates 10 methods across 4 water benchmarks and multiple shot regimes:
1. Zero-Shot
2. ProtoNet (Prototypical Networks)
3. LP++ (Inductive Linear Probe with Zero-Shot Prior)
4. Tip-Adapter (Training-Free Cache Model)
5. LaplacianShot (Transductive Laplacian Smoothing)
6. TransCLIP (Gaussian-Mixture Transductive Baseline)
7. TIM++ (Transductive Information Maximization)
8. LC-TIM (Static Locally Consistent TIM, k=5, lambda=0.3)
9. LC-TIM+SAR (Static Optical-SAR Fusion Baseline)
10. RL-HydroFM (Ours: Dynamic Graph Transduction)
11. RL-HydroFM+SAR (Ours: Multi-Source Optical + SAR Dynamic Router)
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import build_text_classifier_weights, WATER_PROMPT_TEMPLATES
from datasets_loader.water_benchmarks import load_water_benchmark, WATER_BENCHMARK_CONFIGS
from solvers.baseline_solvers import (
    evaluate_zero_shot,
    run_protonet,
    run_linear_probe_pp,
    run_tip_adapter,
    run_laplacianshot,
    run_transclip_solver,
    run_tim_pp_solver,
    run_lctim_solver,
    compute_accuracy_metrics,
)
from solvers.rl_transductive_solver import RLHydroTransductiveSolver
from rl_core.graph_policy import HydroGraphPolicy
from rl_core.hydro_env import HydroTransductiveEnv
from rl_core.ppo_agent import FastPolicyGradient

METHODS = [
    'zero_shot',
    'protonet',
    'lp_pp',
    'tip_adapter',
    'laplacianshot',
    'transclip',
    'tim_pp',
    'lctim',
    'lctim_sar',
    'rl_hydrofm',
    'rl_hydrofm_sar',
]

METHOD_NAMES = {
    'zero_shot': 'Zero-Shot Baseline',
    'protonet': 'ProtoNet',
    'lp_pp': 'LP++ (Inductive)',
    'tip_adapter': 'Tip-Adapter',
    'laplacianshot': 'LaplacianShot',
    'transclip': 'TransCLIP',
    'tim_pp': 'TIM++',
    'lctim': 'LC-TIM (Static k=5)',
    'lctim_sar': 'LC-TIM+SAR (Static Fusion)',
    'rl_hydrofm': 'RL-HydroFM (Ours: Dynamic Graph)',
    'rl_hydrofm_sar': 'RL-HydroFM+SAR (Ours: Multi-Source Router)',
}


def train_fast_policy(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    clip_weights: torch.Tensor,
    extra_features: torch.Tensor = None,
    epochs: int = 15,
    device: torch.device = torch.device('cuda'),
) -> HydroGraphPolicy:
    """Trains a fast lightweight graph policy using validation subset."""
    policy = HydroGraphPolicy(state_dim=14, candidate_k_values=(1, 3, 5, 8, 12, 16)).to(device)
    trainer = FastPolicyGradient(policy, lr=0.01)
    env = HydroTransductiveEnv(device=device)

    policy.train()
    for ep in range(epochs):
        state = HydroTransductiveEnv.extract_state_features(
            query_features=val_features,
            clip_weights=clip_weights,
            support_prototypes=None,
            extra_features=extra_features,
        )
        out = policy(state, deterministic=False)
        k_vals = out['k_values']
        max_k = max(1, min(int(k_vals.max().item()), val_features.shape[0] - 1))

        knn_idx = RLHydroTransductiveSolver._build_dynamic_knn_graph(
            feature_sets=[val_features] + ([extra_features] if extra_features is not None else []),
            modality_weights=out['modality_weights'],
            max_k=max_k,
        )

        zs_probs = F.softmax(100.0 * (val_features @ clip_weights), dim=-1)
        reward, _ = env.compute_reward(
            predicted_probs=zs_probs,
            target_labels=val_labels,
            neighbor_indices=knn_idx,
        )
        trainer.step(state, out, float(reward.item()))

    policy.eval()
    return policy


def evaluate_benchmark(
    datasets: List[str] = None,
    shots_list: List[int] = None,
    seeds: List[int] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    if datasets is None:
        datasets = ['eurosat_water', 'sentinel2_water', 'sen12_flood', 'resisc45_water']
    if shots_list is None:
        shots_list = [1, 2, 4, 8, 16]
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    print(f"================================================================")
    print(f"  RUNNING COMPREHENSIVE WATER RESOURCES BENCHMARK (11 METHODS)")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Datasets: {datasets} | Shots: {shots_list} | Seeds: {len(seeds)}")
    print(f"================================================================\n")

    results = {ds: {m: {s: [] for s in shots_list} for m in METHODS} for ds in datasets}
    policy_diagnostics = {ds: {} for ds in datasets}

    total_start = time.time()

    for ds in datasets:
        cfg = WATER_BENCHMARK_CONFIGS[ds]
        print(f"\n>>> BENCHMARK DATASET: {ds.upper()} ({cfg['name']})")
        print(f"    Classes: {cfg['classes']}")

        for shots in shots_list:
            print(f"\n  --- Shot Regime: {shots}-Shot ---")
            seed_metrics = {m: [] for m in METHODS}

            for seed in seeds:
                data = load_water_benchmark(ds, shots=shots, seed=seed, device=device)
                s_opt, s_sar, s_lbl = data['support_opt'], data['support_sar'], data['support_labels']
                v_opt, v_sar, v_lbl = data['val_opt'], data['val_sar'], data['val_labels']
                q_opt, q_sar, q_lbl = data['query_opt'], data['query_sar'], data['query_labels']
                clip_w = data['clip_weights']

                # 1. Zero-Shot
                zs_acc, _ = evaluate_zero_shot(q_opt, clip_w, q_lbl)
                seed_metrics['zero_shot'].append({'top1_accuracy': zs_acc})

                # 2. ProtoNet
                proto_acc = run_protonet(s_opt, s_lbl, q_opt, q_lbl)
                seed_metrics['protonet'].append({'top1_accuracy': proto_acc})

                # 3. LP++
                lppp_acc = run_linear_probe_pp(s_opt, s_lbl, v_opt, v_lbl, q_opt, q_lbl, clip_w)
                seed_metrics['lp_pp'].append({'top1_accuracy': lppp_acc})

                # 4. Tip-Adapter
                tip_acc = run_tip_adapter(s_opt, s_lbl, q_opt, q_lbl, clip_w)
                seed_metrics['tip_adapter'].append({'top1_accuracy': tip_acc})

                # 5. LaplacianShot
                lap_acc = run_laplacianshot(s_opt, s_lbl, q_opt, q_lbl, clip_w)
                seed_metrics['laplacianshot'].append({'top1_accuracy': lap_acc})

                # 6. TransCLIP
                transclip_acc = run_transclip_solver(s_opt, s_lbl, q_opt, q_lbl, clip_w)
                seed_metrics['transclip'].append({'top1_accuracy': transclip_acc})

                # 7. TIM++
                tim_acc = run_tim_pp_solver(s_opt, s_lbl, q_opt, q_lbl, clip_w)
                seed_metrics['tim_pp'].append({'top1_accuracy': tim_acc})

                # 8. LC-TIM (Optical)
                lctim_acc = run_lctim_solver(s_opt, s_lbl, q_opt, q_lbl, clip_w)
                seed_metrics['lctim'].append({'top1_accuracy': lctim_acc})

                # 9. LC-TIM+SAR (Static Fusion)
                lctim_sar_acc = run_lctim_solver(s_opt, s_lbl, q_opt, q_lbl, clip_w, extra_features=q_sar)
                seed_metrics['lctim_sar'].append({'top1_accuracy': lctim_sar_acc})

                # 10. RL-HydroFM (Optical)
                policy_opt = train_fast_policy(s_opt, s_lbl, v_opt, v_lbl, clip_w, extra_features=None, device=device)
                solver_rl_opt = RLHydroTransductiveSolver(policy=policy_opt, fine_tuning_steps=150)
                probs_rl_opt, _ = solver_rl_opt.solve(s_opt, s_lbl, q_opt, clip_w)
                m_rl_opt = compute_accuracy_metrics(probs_rl_opt, q_lbl)
                seed_metrics['rl_hydrofm'].append(m_rl_opt)

                # 11. RL-HydroFM+SAR (Multi-Source Router)
                policy_sar = train_fast_policy(s_opt, s_lbl, v_opt, v_lbl, clip_w, extra_features=v_sar, device=device)
                solver_rl_sar = RLHydroTransductiveSolver(policy=policy_sar, fine_tuning_steps=150)
                probs_rl_sar, diag = solver_rl_sar.solve(s_opt, s_lbl, q_opt, clip_w, extra_features=q_sar)
                m_rl_sar = compute_accuracy_metrics(probs_rl_sar, q_lbl)
                seed_metrics['rl_hydrofm_sar'].append(m_rl_sar)

            print(f"    [Results at {shots}-shot] (Averaged over {len(seeds)} seeds):")
            for m in METHODS:
                accs = [x['top1_accuracy'] for x in seed_metrics[m]]
                results[ds][m][shots] = seed_metrics[m]
                print(f"      {METHOD_NAMES[m]:43s}: {np.mean(accs):6.2f}% \u00b1 {np.std(accs):4.2f}%")

    total_time = time.time() - total_start
    print(f"\n================================================================")
    print(f"  BENCHMARK COMPLETED IN {total_time:.2f} SECONDS")
    print(f"================================================================\n")

    summary = {
        'methods': METHODS,
        'method_names': METHOD_NAMES,
        'datasets': datasets,
        'shots': shots_list,
        'seeds': seeds,
        'results': results,
        'total_runtime_seconds': total_time,
    }

    os.makedirs('./caches', exist_ok=True)
    summary_file = './caches/experiments_summary.json'
    with open(summary_file, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved complete benchmark results to {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Comprehensive Water Resources Benchmarks.")
    parser.add_argument('--datasets', nargs='+', default=['eurosat_water', 'sentinel2_water', 'sen12_flood', 'resisc45_water'])
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_benchmark(args.datasets, args.shots, args.seeds, device=device)


if __name__ == '__main__':
    main()
