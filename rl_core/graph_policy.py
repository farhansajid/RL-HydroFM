"""
Actor-Critic Policy Networks for Adaptive Graph Transduction & Multi-Modal Foundation Model Routing.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


class HydroGraphPolicy(nn.Module):
    """Actor-Critic Policy Network for Transductive Graph Policy Optimization."""

    def __init__(
        self,
        state_dim: int = 14,
        hidden_dim: int = 64,
        candidate_k_values: Tuple[int, ...] = (1, 3, 5, 8, 12, 16),
        num_modalities: int = 2,
    ):
        super().__init__()
        self.candidate_k_values = candidate_k_values
        self.num_k_choices = len(candidate_k_values)
        self.num_modalities = num_modalities

        # Shared feature trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Actor Heads:
        # 1. Discrete Neighborhood Cardinality (k_i)
        self.actor_kappa = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_k_choices),
        )

        # 2. Multi-Modal Sensor Gating (beta_i)
        self.actor_modality = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_modalities),
        )

        # 3. Sample-Adaptive Consistency Exponent (lambda_LC,i)
        self.actor_lambda = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Critic Head: Value baseline V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_orthogonal_weights()

    def _init_orthogonal_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_kappa[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.actor_modality[-1].weight, gain=0.01)

    def forward(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Tensor]:
        """Generates dynamic graph parameters given state tensor.
        
        Args:
            state: [n_q, state_dim]
            deterministic: If True, uses argmax actions for test-time inference.
        """
        features = self.shared_trunk(state) # [n_q, hidden_dim]

        # 1. Neighborhood size policy
        kappa_logits = self.actor_kappa(features) # [n_q, num_k_choices]
        kappa_dist = Categorical(logits=kappa_logits)

        if deterministic:
            k_indices = kappa_logits.argmax(dim=-1)
        else:
            k_indices = kappa_dist.sample()

        log_prob_kappa = kappa_dist.log_prob(k_indices)
        entropy_kappa = kappa_dist.entropy()

        k_candidates_t = torch.tensor(self.candidate_k_values, device=state.device, dtype=torch.long)
        k_values = k_candidates_t[k_indices] # [n_q]

        # 2. Modality fusion weights
        modality_logits = self.actor_modality(features) # [n_q, num_modalities]
        modality_weights = F.softmax(modality_logits, dim=-1) # [n_q, num_modalities]

        # 3. Adaptive local consistency weight
        lambda_lc = self.actor_lambda(features) # [n_q, 1]

        # 4. Critic baseline
        value = self.critic(features) # [n_q, 1]

        return {
            'k_indices': k_indices,
            'k_values': k_values,
            'modality_weights': modality_weights,
            'lambda_lc': lambda_lc,
            'log_prob': log_prob_kappa,
            'entropy': entropy_kappa,
            'value': value,
            'kappa_logits': kappa_logits,
        }

    def evaluate_actions(
        self,
        state: Tensor,
        k_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluates log probabilities, entropy, and critic values for PPO updates."""
        features = self.shared_trunk(state)
        kappa_logits = self.actor_kappa(features)
        kappa_dist = Categorical(logits=kappa_logits)

        log_prob = kappa_dist.log_prob(k_indices)
        entropy = kappa_dist.entropy()
        value = self.critic(features)
        return log_prob, entropy, value


class AdaptiveActorCritic(nn.Module):
    """Wrapper providing high-level deterministic inference utilities."""

    def __init__(self, policy: HydroGraphPolicy):
        super().__init__()
        self.policy = policy

    def get_dynamic_parameters(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            out = self.policy(state, deterministic=True)
        return out['k_values'], out['modality_weights'], out['lambda_lc']
