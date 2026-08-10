"""
Comprehensive Baseline Few-Shot Solvers and Evaluation Metrics.

Includes:
1. Zero-Shot Baseline (VLM text-visual cosine alignment)
2. Standard Linear Probe (Logistic Regression / Ridge)
3. Linear Probe++ (LP++ with learnable zero-shot scalar)
4. Prototypical Networks (ProtoNet)
5. Tip-Adapter (Training-Free Cache Model for CLIP)
6. LaplacianShot (Transductive Laplacian Graph Smoothing)
7. TransCLIP (Gaussian-Mixture Transductive Inference)
8. TIM++ (Transductive Information Maximization)
9. LC-TIM (Locally Consistent TIM with static k=5, lambda_LC=0.3)
10. LC-TIM+SAR / LC-TIM+DINO (Static Multi-Source Multiplicative Fusion)
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_accuracy_metrics(
    predictions: Tensor,
    targets: Tensor,
) -> Dict[str, float]:
    """Computes Top-1 Accuracy, Macro-F1, Precision, and Recall."""
    preds_np = predictions.argmax(dim=-1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    acc = float((preds_np == targets_np).mean() * 100.0)
    macro_f1 = float(f1_score(targets_np, preds_np, average='macro', zero_division=0) * 100.0)
    macro_prec = float(precision_score(targets_np, preds_np, average='macro', zero_division=0) * 100.0)
    macro_rec = float(recall_score(targets_np, preds_np, average='macro', zero_division=0) * 100.0)

    return {
        'top1_accuracy': acc,
        'macro_f1': macro_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
    }


def evaluate_zero_shot(
    query_features: Tensor,
    clip_weights: Tensor,
    targets: Tensor,
) -> Tuple[float, Tensor]:
    """Zero-shot foundation model evaluation."""
    logits = 100.0 * (query_features @ clip_weights)
    probs = F.softmax(logits, dim=-1)
    acc = float((probs.argmax(dim=-1) == targets).float().mean().item() * 100.0)
    return acc, probs


def run_protonet(
    support_features: Tensor,
    support_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
) -> float:
    """Prototypical Networks (ProtoNet) Nearest-Centroid Classifier."""
    device = support_features.device
    K = len(torch.unique(support_labels))
    centroids = torch.stack([
        support_features[support_labels == k].mean(0) for k in range(K)
    ])
    centroids = F.normalize(centroids, dim=-1)
    q_norm = F.normalize(query_features, dim=-1)
    sim = q_norm @ centroids.T
    acc = float((sim.argmax(dim=-1) == query_labels).float().mean().item() * 100.0)
    return acc


def run_linear_probe_pp(
    support_features: Tensor,
    support_labels: Tensor,
    val_features: Optional[Tensor],
    val_labels: Optional[Tensor],
    query_features: Tensor,
    query_labels: Tensor,
    clip_weights: Tensor,
    epochs: int = 50,
) -> float:
    """LP++ Inductive Linear Probe with Learnable Zero-Shot Weighting."""
    device = support_features.device
    n_s, d = support_features.shape
    K = clip_weights.shape[1]

    centroids = torch.stack([
        support_features[support_labels == k].float().mean(0) for k in range(K)
    ])
    classifier = nn.Linear(d, K, bias=True, device=device)
    classifier.weight.data = centroids
    nn.init.zeros_(classifier.bias)

    alpha = nn.Parameter(torch.ones(1, K, device=device))
    optimizer = torch.optim.SGD([
        {'params': classifier.parameters(), 'lr': 0.01, 'momentum': 0.9},
        {'params': [alpha], 'lr': 0.001},
    ])

    val_f = val_features.to(device) if val_features is not None else support_features
    val_l = val_labels.to(device) if val_labels is not None else support_labels

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(epochs):
        classifier.train()
        ones_s = torch.ones(n_s, 1, device=device)
        logits = classifier(support_features) + (ones_s @ alpha) * (support_features @ clip_weights)
        loss = F.cross_entropy(logits, support_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            ones_v = torch.ones(val_f.shape[0], 1, device=device)
            val_logits = classifier(val_f) + (ones_v @ alpha) * (val_f @ clip_weights)
            val_acc = float((val_logits.argmax(dim=-1) == val_l).float().mean().item() * 100.0)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                ones_q = torch.ones(query_features.shape[0], 1, device=device)
                q_logits = classifier(query_features) + (ones_q @ alpha) * (query_features @ clip_weights)
                best_test_acc = float((q_logits.argmax(dim=-1) == query_labels).float().mean().item() * 100.0)

    return best_test_acc


def run_tip_adapter(
    support_features: Tensor,
    support_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
    clip_weights: Tensor,
    alpha: float = 1.0,
    beta: float = 5.5,
) -> float:
    """Tip-Adapter: Training-Free CLIP-Adapter via Key-Value Memory Cache."""
    device = support_features.device
    K = clip_weights.shape[1]

    # Key-value cache: keys are support features, values are one-hot labels
    F_keys = F.normalize(support_features, dim=-1) # [n_s, d]
    L_values = F.one_hot(support_labels, K).float().to(device) # [n_s, K]
    F_query = F.normalize(query_features, dim=-1) # [n_q, d]

    # Affinity cache: A = exp(-beta * (1 - Q @ K^T))
    affinity = torch.exp(-beta * (1.0 - F_query @ F_keys.T)) # [n_q, n_s]
    cache_logits = affinity @ L_values # [n_q, K]

    clip_logits = 100.0 * (F_query @ clip_weights) # [n_q, K]
    tip_logits = clip_logits + alpha * cache_logits

    acc = float((tip_logits.argmax(dim=-1) == query_labels).float().mean().item() * 100.0)
    return acc


def run_laplacianshot(
    support_features: Tensor,
    support_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
    clip_weights: Tensor,
    lambda_lap: float = 0.7,
    k_nn: int = 5,
    iters: int = 20,
) -> float:
    """LaplacianShot: Transductive Graph Smoothing with Laplacian Regularization."""
    device = support_features.device
    n_q = query_features.shape[0]
    K = clip_weights.shape[1]

    # Construct kNN Laplacian graph over query set
    q_norm = F.normalize(query_features, dim=-1)
    sim = q_norm @ q_norm.T
    sim.fill_diagonal_(0.0)
    k_top = min(k_nn, n_q - 1) if n_q > 1 else 1
    top_sim, top_idx = sim.topk(k_top, dim=-1)

    A = torch.zeros(n_q, n_q, device=device)
    A.scatter_(1, top_idx, top_sim)
    W_graph = 0.5 * (A + A.T)

    # Initial probability estimates via Zero-Shot & Support Centroids
    centroids = torch.stack([
        support_features[support_labels == k].mean(0) for k in range(K)
    ])
    proto_logits = 20.0 * (q_norm @ F.normalize(centroids, dim=-1).T)
    P = F.softmax(proto_logits, dim=-1)

    # Fixed point graph diffusion iterations
    D = W_graph.sum(dim=1, keepdim=True).clamp(min=1e-8)
    W_norm = W_graph / D

    for _ in range(iters):
        P_smooth = W_norm @ P
        P = (1.0 - lambda_lap) * P + lambda_lap * P_smooth
        P = F.normalize(P, p=1, dim=-1)

    acc = float((P.argmax(dim=-1) == query_labels).float().mean().item() * 100.0)
    return acc


def run_transclip_solver(
    support_features: Tensor,
    support_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
    clip_weights: Tensor,
    iters: int = 15,
) -> float:
    """TransCLIP Gaussian-Mixture Transductive Solver."""
    device = support_features.device
    K = clip_weights.shape[1]
    n_q = query_features.shape[0]

    prototypes = torch.stack([
        support_features[support_labels == k].mean(0) for k in range(K)
    ])

    sim = query_features @ query_features.T
    sim.fill_diagonal_(0.0)
    k_nn_tc = min(5, n_q - 1) if n_q > 1 else 1
    top_sim, top_idx = sim.topk(k_nn_tc, dim=-1)
    A = torch.zeros(n_q, n_q, device=device)
    A.scatter_(1, top_idx, top_sim)
    A = 0.5 * (A + A.T)

    P_q = F.softmax(100.0 * (query_features @ clip_weights), dim=-1)
    for _ in range(iters):
        P_smooth = A @ P_q
        P_q = 0.7 * P_q + 0.3 * F.normalize(P_smooth, p=1, dim=-1)
        prototypes = (P_q.T @ query_features) / (P_q.sum(dim=0, keepdim=True).T + 1e-8)
        prototypes = F.normalize(prototypes, dim=-1)
        P_q = F.softmax(20.0 * (query_features @ prototypes.T), dim=-1)

    acc = float((P_q.argmax(dim=-1) == query_labels).float().mean().item() * 100.0)
    return acc


def run_tim_pp_solver(
    support_features: Tensor,
    support_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
    clip_weights: Tensor,
    steps: int = 150,
) -> float:
    """TIM++ Mutual Information Maximization Baseline."""
    from solvers.rl_transductive_solver import RLHydroTransductiveSolver
    solver = RLHydroTransductiveSolver(
        policy=None,
        fine_tuning_steps=steps,
        default_lambda_lc=0.0,
    )
    probs, _ = solver.solve(
        support_features=support_features,
        support_labels=support_labels,
        query_features=query_features,
        clip_weights=clip_weights,
    )
    acc = float((probs.argmax(dim=-1) == query_labels).float().mean().item() * 100.0)
    return acc


def run_lctim_solver(
    support_features: Tensor,
    support_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
    clip_weights: Tensor,
    extra_features: Optional[Tensor] = None,
    steps: int = 150,
) -> float:
    """LC-TIM Locally Consistent Transductive Information Maximization Baseline (Static k=5, lambda=0.3)."""
    from solvers.rl_transductive_solver import RLHydroTransductiveSolver
    solver = RLHydroTransductiveSolver(
        policy=None,
        fine_tuning_steps=steps,
        default_lambda_lc=0.3,
    )
    probs, _ = solver.solve(
        support_features=support_features,
        support_labels=support_labels,
        query_features=query_features,
        clip_weights=clip_weights,
        extra_features=extra_features,
    )
    acc = float((probs.argmax(dim=-1) == query_labels).float().mean().item() * 100.0)
    return acc
