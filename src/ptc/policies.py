"""Commit policies for streaming translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


READ = 0
COMMIT = 1


class CommitPolicy(nn.Module):
    """Lightweight MLP policy for READ/COMMIT decisions.

    The policy receives a concatenated state vector consisting of decoder state
    and scalar streaming features such as entropy, source read position, committed
    length, source character mass, and visible target mass.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return logits for READ/COMMIT actions."""
        return self.net(state)

    def act(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample or greedily select an action.

        Returns action and log-probability. During evaluation, deterministic=True
        gives argmax actions so the same checkpoint yields the same trace.
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


def make_policy_state(
    decoder_state: torch.Tensor,
    entropy: float,
    read_pos: int,
    committed_len: int,
    source_mass: float,
    visible_mass: float,
) -> torch.Tensor:
    """Build a single policy state tensor from vector and scalar features."""
    device = decoder_state.device
    scalar = torch.tensor(
        [[float(entropy), float(read_pos), float(committed_len), float(source_mass), float(visible_mass)]],
        dtype=torch.float32,
        device=device,
    )
    return torch.cat([decoder_state, scalar], dim=-1)
