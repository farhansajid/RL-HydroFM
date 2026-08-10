"""
PPO and Policy Gradient Optimization Agents for Graph Policy Adaptation.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from rl_core.graph_policy import HydroGraphPolicy


class HydroPPOAgent:
    """Proximal Policy Optimization (PPO) Agent for Transductive Graph Tuning."""

    def __init__(
        self,
        policy: HydroGraphPolicy,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.policy = policy.to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        rewards: Tensor,
        values: Tensor,
    ) -> Dict[str, float]:
        """Executes clipped PPO update over collected trajectory buffers."""
        self.policy.train()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device).squeeze(-1)

        advantages = rewards - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = states.shape[0]
        indices = torch.randperm(n_samples)

        loss_actor_accum, loss_critic_accum, entropy_accum = 0.0, 0.0, 0.0
        num_batches = 0

        for _ in range(self.ppo_epochs):
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                b_idx = indices[start:end]

                b_states = states[b_idx]
                b_actions = actions[b_idx]
                b_old_log_probs = old_log_probs[b_idx]
                b_advantages = advantages[b_idx]
                b_targets = rewards[b_idx]

                new_log_probs, entropy, new_values = self.policy.evaluate_actions(b_states, b_actions)
                new_values = new_values.squeeze(-1)

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                loss_actor = -torch.min(surr1, surr2).mean()

                loss_critic = F.mse_loss(new_values, b_targets)
                loss_entropy = -entropy.mean()

                total_loss = loss_actor + self.value_coef * loss_critic + self.entropy_coef * loss_entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                loss_actor_accum += loss_actor.item()
                loss_critic_accum += loss_critic.item()
                entropy_accum += entropy.mean().item()
                num_batches += 1

        return {
            'loss_actor': loss_actor_accum / max(1, num_batches),
            'loss_critic': loss_critic_accum / max(1, num_batches),
            'entropy': entropy_accum / max(1, num_batches),
        }


class FastPolicyGradient:
    """Lightweight episodic policy gradient for fast zero-overhead online adaptation."""

    def __init__(
        self,
        policy: HydroGraphPolicy,
        lr: float = 2e-3,
        entropy_coef: float = 0.01,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.policy = policy.to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.entropy_coef = entropy_coef
        self.device = device
        self.moving_baseline = 0.0

    def step(self, states: Tensor, out_dict: Dict[str, Tensor], reward: float) -> float:
        self.policy.train()
        log_prob = out_dict['log_prob'].mean()
        entropy = out_dict['entropy'].mean()

        advantage = reward - self.moving_baseline
        self.moving_baseline = 0.9 * self.moving_baseline + 0.1 * reward

        loss = -log_prob * advantage - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
