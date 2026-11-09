"""REINFORCE with a scalar running baseline, implemented from scratch in PyTorch.

SE4095 Social Network Theory and Practice - Assignment 03
Author: M. Waqas Chohan (23F-3041)

Loss for a batch of B = 20 complete episodes (discount factor 1, so every step
of episode i uses the same return R_i):

    L(theta) = -(1/B) * sum_i (R_i - b) * sum_t log pi_theta(a_{i,t} | s_{i,t})

The baseline b starts at 0, is held fixed while a batch is collected and its loss
computed, and is updated after the single optimizer step: b <- 0.9 b + 0.1 mean(R).
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch

from environment import SeedSelectionEnv
from models import action_distribution, build_policy

# Fixed training settings (assignment Section 6).
LEARNING_RATE = 1e-3
GAMMA = 1.0
EPISODES_PER_UPDATE = 20
MAX_GRAD_NORM = 1.0
TRAINING_EPISODES = 1000
BASELINE_DECAY = 0.9


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch at the start of a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_diffusion_seed(train_seed: int, episode: int) -> int:
    return 1_000_000 + 10_000 * train_seed + episode


def run_episode(env: SeedSelectionEnv, policy: torch.nn.Module, diffusion_seed: int):
    """Sample one complete episode; keep the differentiable log-probabilities."""
    features, mask = env.reset(diffusion_seed)
    log_probs = []
    done, reward, info = False, 0.0, {}
    while not done:
        dist = action_distribution(policy, torch.from_numpy(features), torch.from_numpy(mask))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        features, mask, reward, done, info = env.step(int(action.item()))
    # gamma = 1 and only the terminal reward is non-zero, so G_t = R for every t.
    return torch.stack(log_probs).sum(), float(reward), info


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_reinforce(env: SeedSelectionEnv, architecture: str, train_seed: int,
                    episodes: int = TRAINING_EPISODES, device: str = "cpu", log=print):
    """Train one policy for ``episodes`` episodes (50 updates of 20 episodes).

    Returns the trained policy, a list of per-episode records, training seconds
    and diagnostics (losses, gradient norms, baselines) for checks.json.
    """
    device = torch.device(device)
    set_global_seeds(train_seed)
    policy = build_policy(architecture).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    baseline = 0.0  # re-initialized for every training run
    records, losses, grad_norms, baselines = [], [], [], []
    batch_log_probs, batch_rewards = [], []

    _synchronize(device)
    start = time.perf_counter()
    policy.train()
    for episode in range(episodes):
        seed = training_diffusion_seed(train_seed, episode)
        log_prob_sum, reward, info = run_episode(env, policy, seed)
        batch_log_probs.append(log_prob_sum)
        batch_rewards.append(reward)
        records.append({"train_seed": train_seed, "episode": episode,
                        "final_spread": info["final_spread"], "total_reward": reward})

        if len(batch_rewards) == EPISODES_PER_UPDATE:
            # Returns and baseline are plain floats -> they carry no gradient.
            returns = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            advantages = returns - baseline
            loss = -(advantages * torch.stack(batch_log_probs)).sum() / EPISODES_PER_UPDATE

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()  # exactly one optimizer step per batch

            losses.append(float(loss.item()))
            grad_norms.append(float(grad_norm.item()))
            baselines.append(baseline)
            baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * float(np.mean(batch_rewards))
            batch_log_probs, batch_rewards = [], []  # discard trajectories after the update

            update = len(losses)
            if update % 10 == 0:
                recent = np.mean([r["final_spread"] for r in records[-200:]])
                log(f"    [{architecture:<11} seed {train_seed:>2}] update {update:>2}/50  "
                    f"loss {loss.item():+.4f}  baseline {baseline:.4f}  "
                    f"mean spread (last 200) {recent:.2f}")
    _synchronize(device)
    training_seconds = time.perf_counter() - start

    diagnostics = {
        "losses": losses,
        "grad_norms": grad_norms,
        "baselines_used": baselines,
        "final_baseline": baseline,
        "num_updates": len(losses),
        "all_finite": bool(all(math.isfinite(x) for x in losses + grad_norms)),
    }
    return policy, records, training_seconds, diagnostics
