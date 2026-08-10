"""
Hydrological & Remote Sensing Transductive Reinforcement Learning Environment.

Models transductive few-shot inference over multi-source Earth observation foundation models
as a Contextual Markov Decision Process.
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HydroTransductiveEnv:
    """Environment modeling transductive few-shot inference for water resources & flood monitoring."""

    def __init__(
        self,
        candidate_k_values: Tuple[int, ...] = (1, 3, 5, 8, 12, 16),
        alpha_mi: float = 1.0,
        consensus_weight: float = 0.5,
        entropy_weight: float = 0.2,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.candidate_k_values = candidate_k_values
        self.num_k_choices = len(candidate_k_values)
        self.alpha_mi = alpha_mi
        self.consensus_weight = consensus_weight
        self.entropy_weight = entropy_weight
        self.device = device

    @staticmethod
    def extract_state_features(
        query_features: Tensor,
        clip_weights: Tensor,
        support_prototypes: Optional[Tensor] = None,
        extra_features: Optional[Tensor] = None,
        top_candidates: int = 20,
    ) -> Tensor:
        """Constructs diagnostic state representations for each query sample.
        
        Returns:
            state: [n_q, 14] state tensor capturing semantic confidence, local topology,
                   and cross-modal optical-SAR concordance.
        """
        device = query_features.device
        n_q, d = query_features.shape
        K = clip_weights.shape[1]
        eps = 1e-12

        # 1. Zero-shot Predictive Distribution & Uncertainty Metrics
        zs_logits = 100.0 * (query_features @ clip_weights)
        zs_probs = F.softmax(zs_logits, dim=-1) # [n_q, K]

        # Normalized Entropy in [0, 1]
        entropy = -(zs_probs * torch.log(zs_probs + eps)).sum(dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(K), device=device)) + eps
        norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0) # [n_q, 1]

        # Margin: Gap between Top-1 and Top-2 predicted classes
        top2_vals, _ = zs_probs.topk(min(2, K), dim=-1)
        if top2_vals.shape[1] > 1:
            margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(-1) # [n_q, 1]
        else:
            margin = top2_vals[:, 0].unsqueeze(-1)

        # Top-1 confidence
        top1_prob = top2_vals[:, 0:1] # [n_q, 1]

        # 2. Local Manifold Geometry & Candidate Density
        sim_matrix = query_features @ query_features.T
        sim_matrix.fill_diagonal_(-1.0)
        k_eval = min(top_candidates, n_q - 1) if n_q > 1 else 1
        top_sims, _ = sim_matrix.topk(k_eval, dim=-1) # [n_q, k_eval]

        sim_mean = top_sims.mean(dim=-1, keepdim=True)
        sim_std = top_sims.std(dim=-1, keepdim=True) if k_eval > 1 else torch.zeros_like(sim_mean)
        sim_max = top_sims[:, 0:1]
        sim_min = top_sims[:, -1:]
        sim_spread = sim_max - sim_min
        geo_stats = torch.cat([sim_mean, sim_std, sim_max, sim_min, sim_spread], dim=-1) # [n_q, 5]

        # 3. Support Prototype Alignment
        if support_prototypes is not None and support_prototypes.numel() > 0:
            proto_norm = F.normalize(support_prototypes, dim=-1)
            proto_sims = query_features @ proto_norm.T # [n_q, K]
            proto_max, _ = proto_sims.max(dim=-1, keepdim=True)
            proto_mean = proto_sims.mean(dim=-1, keepdim=True)
            proto_gap = proto_max - proto_mean
            proto_stats = torch.cat([proto_max, proto_gap], dim=-1) # [n_q, 2]
        else:
            proto_stats = torch.zeros(n_q, 2, device=device)

        # 4. Multi-Modal Concordance (Optical vs. SAR / DINOv3)
        if extra_features is not None:
            extra_norm = F.normalize(extra_features, dim=-1)
            extra_sim = extra_norm @ extra_norm.T
            extra_sim.fill_diagonal_(-1.0)
            top_extra, _ = extra_sim.topk(k_eval, dim=-1)
            extra_mean = top_extra.mean(dim=-1, keepdim=True)
            extra_max = top_extra[:, 0:1]

            # Cross-modal correlation (cosine similarity between similarity profiles)
            u = F.normalize(sim_matrix, dim=-1)
            v = F.normalize(extra_sim, dim=-1)
            cross_agreement = (u * v).sum(dim=-1, keepdim=True) # [n_q, 1]
            extra_stats = torch.cat([extra_mean, extra_max, cross_agreement, (sim_mean - extra_mean).abs()], dim=-1) # [n_q, 4]
        else:
            extra_stats = torch.zeros(n_q, 4, device=device)

        # Concatenate: 1(norm_entropy) + 1(margin) + 1(top1_prob) + 5(geo_stats) + 2(proto_stats) + 4(extra_stats) = 14
        state = torch.cat([norm_entropy, margin, top1_prob, geo_stats, proto_stats, extra_stats], dim=-1)
        return state

    def compute_reward(
        self,
        predicted_probs: Tensor,
        target_labels: Optional[Tensor] = None,
        neighbor_indices: Optional[Tensor] = None,
        neighbor_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Computes comprehensive transductive reward combining supervised validation,
        information gain, and graph consensus.
        """
        eps = 1e-12
        n_q, K = predicted_probs.shape
        metrics = {}

        # 1. Marginal Entropy (discouraging class collapse)
        class_marginals = predicted_probs.mean(dim=0).clamp(min=eps) # [K]
        marginal_entropy = -(class_marginals * torch.log(class_marginals)).sum()

        # 2. Conditional Entropy (encouraging decisive per-sample predictions)
        sample_entropy = -(predicted_probs * torch.log(predicted_probs + eps)).sum(dim=-1)
        cond_entropy = sample_entropy.mean()

        mi_gain = self.alpha_mi * marginal_entropy - cond_entropy
        metrics['marginal_entropy'] = float(marginal_entropy.item())
        metrics['conditional_entropy'] = float(cond_entropy.item())
        metrics['mutual_information'] = float(mi_gain.item())

        # 3. Neighborhood Consensus Agreement
        consensus_score = torch.tensor(0.0, device=predicted_probs.device)
        if neighbor_indices is not None:
            neighbor_p = predicted_probs[neighbor_indices] # [n_q, k, K]
            p_expanded = predicted_probs.unsqueeze(1)      # [n_q, 1, K]
            dot_products = (p_expanded * neighbor_p).sum(dim=-1) # [n_q, k]
            if neighbor_weights is not None:
                if neighbor_weights.dim() == 2:
                    dot_products = dot_products * neighbor_weights
                elif neighbor_weights.dim() == 1:
                    dot_products = dot_products * neighbor_weights.unsqueeze(-1)
            consensus_score = dot_products.mean()
            metrics['consensus_score'] = float(consensus_score.item())

        # 4. Supervised Support Accuracy (when validation labels are available)
        acc_reward = torch.tensor(0.0, device=predicted_probs.device)
        if target_labels is not None and target_labels.numel() > 0:
            preds = predicted_probs.argmax(dim=-1)
            correct = (preds == target_labels).float()
            acc_reward = correct.mean()
            metrics['accuracy'] = float(acc_reward.item() * 100.0)

        total_reward = (
            acc_reward
            + 0.15 * mi_gain
            + self.consensus_weight * consensus_score
            - self.entropy_weight * cond_entropy
        )
        metrics['total_reward'] = float(total_reward.item())
        return total_reward, metrics
