"""
Reinforcement-Learned Transductive Information Maximization Solver (RL-HydroFM).

Features:
- Sample-adaptive neighborhood cardinality k_i
- Policy-driven multi-source optical-SAR sensor gating beta_i
- Uncertainty-calibrated local consistency exponentiation lambda_LC,i
- Closed-form Alternating Direction Method (ADM) updates
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rl_core.graph_policy import HydroGraphPolicy
from rl_core.hydro_env import HydroTransductiveEnv


class RLHydroTransductiveSolver(nn.Module):
    """Vectorized ADM Transductive Solver with RL-Guided Dynamic Graphs."""

    def __init__(
        self,
        policy: Optional[HydroGraphPolicy] = None,
        fine_tuning_steps: int = 150,
        fine_tuning_lr: float = 1e-4,
        cross_entropy_weight: float = 0.4,
        marginal_entropy_weight: float = 1.0,
        conditional_entropy_weight: float = 0.1,
        temperature: float = 120.0,
        alpha: float = 1.0,
        gamma: float = 0.05,
        default_lambda_lc: float = 0.3,
        deterministic_policy: bool = True,
    ):
        super().__init__()
        self.policy = policy
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.cross_entropy_weight = cross_entropy_weight
        self.marginal_entropy_weight = marginal_entropy_weight
        self.conditional_entropy_weight = conditional_entropy_weight
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.default_lambda_lc = default_lambda_lc
        self.deterministic_policy = deterministic_policy

        self.loss_weights = [
            self.cross_entropy_weight,
            self.marginal_entropy_weight,
            self.conditional_entropy_weight,
        ]

        self.prototypes: Optional[Tensor] = None
        self.weights: Optional[Tensor] = None
        self.Q: Optional[Tensor] = None
        self.k_per_sample: Optional[Tensor] = None
        self.lambda_lc_per_sample: Optional[Tensor] = None
        self.modality_weights: Optional[Tensor] = None

    def get_logits(self, samples: Tensor) -> Tensor:
        n_tasks = samples.size(0)
        logits = self.temperature * (
            samples.matmul(self.weights.transpose(1, 2))
            - 0.5 * (self.weights ** 2).sum(2).view(n_tasks, 1, -1)
            - 0.5 * (samples ** 2).sum(2).view(n_tasks, -1, 1)
        )
        return logits

    def q_update(self, P: Tensor, clip_logits: Tensor, knn_idx: Tensor):
        """Vectorized closed-form q-update with sample-adaptive local consistency."""
        l1, l2 = self.loss_weights[1], self.loss_weights[2]
        l3 = 1.0
        alpha = l2 / l3
        beta = l1 / (l1 + l3)

        unnorm = (P ** (1 + alpha)) * (clip_logits ** self.gamma)

        if knn_idx is not None and self.k_per_sample is not None:
            p_flat = P.squeeze(0) # [n_q, K]
            n_q, K = p_flat.shape
            max_k = knn_idx.shape[1]

            neighbor_p = p_flat[knn_idx] # [n_q, max_k, K]

            # Vectorized masking for variable k_i
            k_range = torch.arange(max_k, device=p_flat.device).unsqueeze(0).expand(n_q, -1)
            k_counts = self.k_per_sample.unsqueeze(1)
            mask = (k_range < k_counts).unsqueeze(-1).float() # [n_q, max_k, 1]

            # Neighborhood consensus prediction
            p_bar = (neighbor_p * mask).sum(dim=1) / self.k_per_sample.unsqueeze(-1).float().clamp(min=1.0)
            p_bar = p_bar.unsqueeze(0).clamp(min=1e-12) # [1, n_q, K]

            if self.lambda_lc_per_sample is not None:
                lam = self.lambda_lc_per_sample.unsqueeze(0) # [1, n_q, 1]
                unnorm = unnorm * (p_bar ** lam)
            else:
                unnorm = unnorm * (p_bar ** self.default_lambda_lc)

        Q = unnorm / (unnorm.sum(dim=1, keepdim=True) ** beta)
        self.Q = (Q / Q.sum(dim=2, keepdim=True)).float()

    def weights_update(self, support: Tensor, query: Tensor, y_s_one_hot: Tensor):
        """ADM closed-form update for prototype matrix W."""
        n_tasks = support.size(0)

        P_s = self.get_logits(support).softmax(2)
        P_q = self.get_logits(query).softmax(2)

        src_scale = self.loss_weights[0] / (1 + self.loss_weights[2])
        src_part = src_scale * y_s_one_hot.transpose(1, 2).matmul(support)
        src_part += src_scale * (
            self.weights * P_s.sum(1, keepdim=True).transpose(1, 2)
            - P_s.transpose(1, 2).matmul(support)
        )
        src_norm = src_scale * y_s_one_hot.sum(1).view(n_tasks, -1, 1)

        qry_scale = self.N_s / self.N_q
        qry_part = qry_scale * self.Q.transpose(1, 2).matmul(query)
        qry_part += qry_scale * (
            self.weights * P_q.sum(1, keepdim=True).transpose(1, 2)
            - P_q.transpose(1, 2).matmul(query)
        )
        qry_norm = qry_scale * self.Q.sum(1).view(n_tasks, -1, 1)

        new_weights = (src_part + qry_part) / (src_norm + qry_norm)
        self.weights = self.weights + self.alpha * (new_weights - self.weights)

    def init_weights(self, support: Tensor, y_s: Tensor, query: Tensor, clip_logits: Tensor):
        n_tasks = support.size(0)
        max_indices = clip_logits.argmax(dim=-1)
        one_hot_text = torch.zeros_like(clip_logits)
        tasks, n_query = max_indices.shape
        one_hot_text[torch.arange(tasks)[:, None], torch.arange(n_query)[None, :], max_indices] = 1

        counts = y_s.sum(1).view(n_tasks, -1, 1).float().to(support.device)
        counts_text = one_hot_text.sum(1).view(tasks, -1, 1).float().to(support.device)
        weights = y_s.transpose(1, 2).matmul(support)
        weights_text = one_hot_text.transpose(1, 2).matmul(query)
        self.weights = (weights + weights_text) / (counts + counts_text)

    def solve(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
        clip_weights: Tensor,
        extra_features: Optional[Tensor] = None,
        precomputed_clip_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Solves transductive few-shot inference problem.
        
        Args:
            support_features: [n_s, d] L2-normalized support embeddings.
            support_labels: [n_s] support class labels (0 to K-1).
            query_features: [n_q, d] L2-normalized query embeddings.
            clip_weights: [d, K] zero-shot classifier weights.
            extra_features: Optional [n_q, d_extra] secondary structural/SAR features.
            
        Returns:
            predicted_probs: [n_q, K] posterior class assignment probabilities.
            policy_metrics: Dictionary of diagnostic statistics.
        """
        device = clip_weights.device
        n_q, d = query_features.shape
        K = clip_weights.shape[1]

        # 1. State Extraction & Policy Inference
        proto_init = torch.stack([
            support_features[support_labels == k].mean(0) for k in range(K)
        ]).to(device) # [K, d]

        state = HydroTransductiveEnv.extract_state_features(
            query_features=query_features,
            clip_weights=clip_weights,
            support_prototypes=proto_init,
            extra_features=extra_features,
        )

        if self.policy is not None:
            self.policy.to(device).eval()
            with torch.no_grad():
                out = self.policy(state, deterministic=self.deterministic_policy)
            self.k_per_sample = out['k_values'].to(device)
            self.lambda_lc_per_sample = out['lambda_lc'].to(device)
            self.modality_weights = out['modality_weights'].to(device)
        else:
            self.k_per_sample = torch.full((n_q,), 5, dtype=torch.long, device=device)
            self.lambda_lc_per_sample = torch.full((n_q, 1), self.default_lambda_lc, device=device)
            self.modality_weights = torch.full((n_q, 2), 0.5, device=device)

        # 2. Dynamic kNN Graph Construction
        max_k = int(self.k_per_sample.max().item())
        feature_sets = [query_features]
        if extra_features is not None:
            feature_sets.append(extra_features)

        knn_idx = self._build_dynamic_knn_graph(
            feature_sets=feature_sets,
            modality_weights=self.modality_weights,
            max_k=max_k,
        )

        # 3. Setup Prior Logits & Supervision
        if precomputed_clip_logits is not None:
            clip_logits = precomputed_clip_logits.unsqueeze(0).float().to(device)
        else:
            clip_logits = 100.0 * (query_features @ clip_weights).unsqueeze(0)
            clip_logits = F.softmax(clip_logits, dim=-1)

        y_s_one_hot = F.one_hot(support_labels, K).unsqueeze(0).float().to(device)
        s_feat = support_features.unsqueeze(0).float().to(device)
        q_feat = query_features.unsqueeze(0).float().to(device)

        self.N_s = support_labels.shape[0]
        self.N_q = n_q

        self.init_weights(s_feat, y_s_one_hot, q_feat, clip_logits)

        # 4. ADM Iterative Loop
        for _ in range(self.fine_tuning_steps):
            P_q = self.get_logits(q_feat).softmax(2)
            self.q_update(P=P_q, clip_logits=clip_logits, knn_idx=knn_idx)
            self.weights_update(s_feat, q_feat, y_s_one_hot)

        self.prototypes = self.weights[0]

        # Final predictions via prototype distance
        cos_sim = F.normalize(query_features, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        final_probs = F.softmax(self.temperature * cos_sim, dim=-1)

        policy_metrics = {
            'mean_k': float(self.k_per_sample.float().mean().item()),
            'min_k': int(self.k_per_sample.min().item()),
            'max_k': int(self.k_per_sample.max().item()),
            'mean_lambda_lc': float(self.lambda_lc_per_sample.mean().item()),
            'mean_opt_weight': float(self.modality_weights[:, 0].mean().item()),
            'mean_sar_weight': float(self.modality_weights[:, 1].mean().item()) if self.modality_weights.shape[1] > 1 else 0.0,
        }

        return final_probs, policy_metrics

    @staticmethod
    def _build_dynamic_knn_graph(
        feature_sets: List[Tensor],
        modality_weights: Tensor,
        max_k: int,
    ) -> Tensor:
        """Batched dynamic kNN graph construction with per-sample modality fusion."""
        with torch.no_grad():
            n_q = feature_sets[0].shape[0]
            max_k = max(1, min(max_k, n_q - 1)) if n_q > 1 else 1
            chunk = min(1024, n_q)
            chunks = []

            for start in range(0, n_q, chunk):
                end = min(start + chunk, n_q)
                chunk_len = end - start

                if len(feature_sets) == 1:
                    sim_row = feature_sets[0][start:end] @ feature_sets[0].T
                else:
                    sim_row = torch.zeros(chunk_len, n_q, device=feature_sets[0].device)
                    for m_idx, f in enumerate(feature_sets):
                        sim_m = (f[start:end] @ f.T + 1.0) / 2.0
                        w_m = modality_weights[start:end, m_idx:m_idx+1]
                        sim_row = sim_row + w_m * sim_m

                for li in range(chunk_len):
                    sim_row[li, start + li] = -1e9 # Mask self

                chunks.append(sim_row.topk(max_k, dim=1)[1])

            return torch.cat(chunks, dim=0) # [n_q, max_k]
