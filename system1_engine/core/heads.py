from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class CategoricalPolicyHead(nn.Module):
    """Categorical policy head for discrete action spaces."""

    def __init__(self, in_features: int = 256, n_actions: int = 2) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_actions = n_actions
        self.linear = nn.Linear(in_features, n_actions)

    def forward(self, h: torch.Tensor) -> Categorical:
        logits = self.linear(h)
        return Categorical(logits=logits)

    def get_action_and_log_prob(
        self, h: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(h)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class GaussianPolicyHead(nn.Module):
    """Diagonal Gaussian policy head for continuous action spaces."""

    def __init__(
        self,
        in_features: int = 256,
        act_dim: int = 1,
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.act_dim = act_dim
        self.mu_net = nn.Linear(in_features, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, h: torch.Tensor) -> Normal:
        mu = self.mu_net(h)
        # Clamp log_std defensively
        log_std = torch.clamp(self.log_std, min=-20.0, max=2.0)
        std = log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def get_action_and_log_prob(
        self, h: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(h)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        # Sum log probs across action dimension if multidimensional
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class ValueHead(nn.Module):
    """Critic head estimating expected return V(s)."""

    def __init__(self, in_features: int = 256) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)
