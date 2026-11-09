"""Policy networks for sequential seed selection.

SE4095 Social Network Theory and Practice - Assignment 03
Author: M. Waqas Chohan (23F-3041)

Both policies map the N x 5 observation to one logit per participant. Selected
participants are masked only at the final action-selection stage (``masked_logits``).
"""

from __future__ import annotations

import torch
from torch import nn

NUM_FEATURES = 5
HIDDEN = 64


class MLPPolicy(nn.Module):
    """Shared per-participant MLP: 5 -> 64 -> ReLU -> 64 -> ReLU -> 1.

    The same weights score every participant independently from its own five
    features, so the only interaction between users comes through the
    selection-dependent features.
    """

    architecture = "mlp"

    def __init__(self, in_features: int = NUM_FEATURES, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # (N, 5) -> (N,)
        return self.net(features).squeeze(-1)

    def config(self) -> dict:
        return {"architecture": self.architecture, "input_features": NUM_FEATURES,
                "layers": "Linear(5,64)-ReLU-Linear(64,64)-ReLU-Linear(64,1)"}


class TransformerPolicy(nn.Module):
    """One token per participant: Linear(5,64) -> 1 Transformer encoder layer -> Linear(64,1).

    Settings: d_model 64, 4 heads, 1 layer, feed-forward 128, ReLU, dropout 0,
    post layer normalization. No positional or node-ID embeddings, no causal
    mask and no graph-restricted attention: every token attends to all N tokens,
    including the already selected ones.
    """

    architecture = "transformer"

    def __init__(self, in_features: int = NUM_FEATURES, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 1, dim_feedforward: int = 128):
        super().__init__()
        self.settings = {"d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                         "dim_feedforward": dim_feedforward, "activation": "relu",
                         "dropout": 0.0, "norm_first": False}
        self.input_projection = nn.Linear(in_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,  # post layer normalization
        )
        # norm=None: no extra final normalization beyond the layer's own LayerNorms.
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=None,
                                             enable_nested_tensor=False)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # (N, 5) -> (N,)
        tokens = self.input_projection(features).unsqueeze(0)  # (1, N, 64)
        encoded = self.encoder(tokens)                          # full attention, no mask
        return self.output_head(encoded).squeeze(0).squeeze(-1)

    def config(self) -> dict:
        return {"architecture": self.architecture, "input_features": NUM_FEATURES, **self.settings}


def build_policy(architecture: str) -> nn.Module:
    if architecture == "mlp":
        return MLPPolicy()
    if architecture == "transformer":
        return TransformerPolicy()
    raise ValueError(f"unknown architecture: {architecture}")


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of unavailable (already selected) participants to -inf."""
    return logits.masked_fill(~mask, float("-inf"))


def action_distribution(policy: nn.Module, features: torch.Tensor,
                        mask: torch.Tensor) -> torch.distributions.Categorical:
    """Masked categorical policy pi_theta(. | s); masked actions get probability 0."""
    return torch.distributions.Categorical(logits=masked_logits(policy(features), mask))


def greedy_action(policy: nn.Module, features: torch.Tensor, mask: torch.Tensor) -> int:
    """Highest-probability valid action; exact ties are broken by the smaller node id."""
    probs = torch.softmax(masked_logits(policy(features), mask), dim=-1)
    best = probs.max()
    candidates = torch.nonzero((probs == best) & mask, as_tuple=False).flatten()
    return int(candidates.min().item())
