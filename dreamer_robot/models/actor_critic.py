import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, feat_dim=384, action_dim=7, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden,   hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, feat):
        return torch.tanh(self.net(feat))

    def distribution(self, feat):
        mean = torch.tanh(self.net(feat))
        # Clamp log_std to (-4, 0): std in [exp(-4)≈0.018, exp(0)=1.0].
        # Prevents too-peaky distributions (log_p → -inf) and too-wide
        # ones (log_p ≈ 0, no gradient signal) that both cause actor explosion.
        std  = self.log_std.clamp(-4, 0).exp().expand_as(mean)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, feat_dim=384, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden,   hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        return self.net(feat)                        # (B, T, 1)
