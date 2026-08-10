"""
Cloud-Degradation & Multi-Modal Optical-SAR Ablation Experiment.

Evaluates how Optical-only, SAR-only, Static Fusion (LC-TIM+DINO), and RL-HydroFM+SAR
perform as optical imagery is progressively corrupted by simulated atmospheric cloud cover
(from 0% clear sky to 80% heavy overcast).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets_loader.water_benchmarks import load_water_benchmark
from solvers.rl_transductive_solver import RLHydroTransductiveSolver
from solvers.baseline_solvers import run_lctim_solver, evaluate_zero_shot, compute_accuracy_metrics
from experiments.run_water_experiments import train_fast_policy


def evaluate_cloud_resilience(
    dataset_key: str = 'sen12_flood',
    shots: int = 4,
    cloud_levels: list = [0.0, 0.2, 0.4, 0.6, 0.8],
    seeds: list = [1, 2, 3, 4, 5],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    print(f"================================================================")
    print(f"  CLOUD DEGRADATION & MULTI-MODAL ABLATION ON {dataset_key.upper()}")
    print(f"  Cloud Attenuation Levels: {cloud_levels}")
    print(f"================================================================\n")

    results = {
        'cloud_levels': cloud_levels,
        'optical_only': [],
        'sar_only': [],
        'static_fusion_lctim': [],
        'rl_hydrofm_multimodal': [],
    }

    for cloud in cloud_levels:
        opt_accs = []
        sar_accs = []
        static_accs = []
        rl_accs = []

        for seed in seeds:
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

            # Simulate cloud degradation on optical query features
            # Cloud attenuation mixes optical features with diffuse cloud scattering noise
            if cloud > 0.0:
                cloud_noise = torch.randn_like(q_opt)
                q_opt_corrupted = F.normalize((1.0 - cloud) * q_opt + cloud * cloud_noise, dim=-1)
            else:
                q_opt_corrupted = q_opt

            # 1. Optical-only RL-HydroFM
            policy_opt = train_fast_policy(s_opt, s_lbl, v_opt, v_lbl, clip_w, extra_features=None, device=device)
            solver_opt = RLHydroTransductiveSolver(policy=policy_opt)
            probs_opt, _ = solver_opt.solve(s_opt, s_lbl, q_opt_corrupted, clip_w)
            opt_accs.append(float((probs_opt.argmax(dim=-1) == q_lbl).float().mean().item() * 100.0))

            # 2. Static Multi-Modal Fusion (LC-TIM)
            static_acc = run_lctim_solver(s_opt, s_lbl, q_opt_corrupted, q_lbl, clip_w, extra_features=q_sar)
            static_accs.append(static_acc)

            # 3. Dynamic RL-HydroFM+SAR Router
            policy_multi = train_fast_policy(s_opt, s_lbl, v_opt, v_lbl, clip_w, extra_features=v_sar, device=device)
            solver_multi = RLHydroTransductiveSolver(policy=policy_multi)
            probs_multi, _ = solver_multi.solve(s_opt, s_lbl, q_opt_corrupted, clip_w, extra_features=q_sar)
            rl_accs.append(float((probs_multi.argmax(dim=-1) == q_lbl).float().mean().item() * 100.0))

        results['optical_only'].append(float(np.mean(opt_accs)))
        results['static_fusion_lctim'].append(float(np.mean(static_accs)))
        results['rl_hydrofm_multimodal'].append(float(np.mean(rl_accs)))

        print(f"  Cloud {cloud*100:2.0f}% -> Optical: {np.mean(opt_accs):.2f}% | Static Fusion: {np.mean(static_accs):.2f}% | RL-HydroFM+SAR: {np.mean(rl_accs):.2f}%")

    out_file = './caches/cloud_ablation_summary.json'
    with open(out_file, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved cloud ablation results to {out_file}\n")
    return results


if __name__ == '__main__':
    evaluate_cloud_resilience()
