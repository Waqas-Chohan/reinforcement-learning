# SE4095 Assignment 03: REINFORCE Seed Selection on a Workplace Contact Network

**Student:** M. Waqas Chohan  **Roll number:** 23F-3041
**Course:** SE4095 Social Network Theory and Practice

An organization can promote an awareness message to only **five of 92 participants** in the
SocioPatterns *Contacts in a Workplace* network. This project implements the following from scratch:

- an **Independent Cascade** simulator (live-arc formulation, p = 0.10);
- a sequential **seed-selection RL environment** (`reset` / `step`, N x 5 observation, action mask,
  terminal reward = spread / N);
- **REINFORCE** with a scalar running baseline in PyTorch;
- two policies: a shared per-node **MLP** and a **Transformer** encoder (one token per participant).

Both policies are compared with **Random** and **Degree** selection on the same 100 diffusion realizations.

## Results at a glance

| Method | Mean spread (100 eval realizations) | Population std |
|---|---:|---:|
| Random | 60.24 | 9.72 |
| Degree | 59.67 | 7.26 |
| REINFORCE MLP (mean of 3 seeds) | 57.53 ± 1.39 | per seed: 13.72 / 16.11 / 18.37 |
| REINFORCE Transformer (mean of 3 seeds) | 58.86 ± 0.65 | per seed: 16.11 / 7.26 / 14.74 |

All 17 verification checks pass (`results/checks.json`). See `report.pdf` for the full analysis.

## Project structure

```
main.py              # runs every experiment: python main.py --output results
environment.py       # graph loader, Independent Cascade simulator, SeedSelectionEnv
models.py            # MLPPolicy, TransformerPolicy, masking and greedy action helpers
reinforce.py         # REINFORCE training loop with running baseline
analysis.py          # network statistics and figures (helper)
requirements.txt     # pinned dependency versions
report.pdf           # 5-page report (source: report/report.html)
SNTP_Assignment03_Walkthrough.ipynb   # executed notebook walking through the project
data/                # supplied dataset package, unchanged
results/
  graph_edges.csv    training.csv (6,000 rows)   evaluation.csv (800 rows)
  summary.csv (8 rows)   checks.json   checkpoints/ (6 final .pt files)   figures/
  # extra descriptive outputs:
  network_stats.json   node_metrics.csv   top10_degree_centrality.csv
  architecture_summary.csv   selected_users.csv   seed_neighbourhood_overlap.csv
  policy_diagnostics.csv   run_info.json
```

## Installation

Python 3.13 was used (Python 3.10 or newer should work).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CPU-only PyTorch build is sufficient. To install it explicitly, run
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Execution

```bash
python main.py --output results
```

This command runs the following, in order:

1. Network analysis: statistics, top-10 table and network figure.
2. Simulator and environment checks.
3. Six training runs (MLP and Transformer x seeds 7, 21, 42; 1,000 episodes each).
4. The controlled evaluation (800 episodes).
5. All CSV files, checkpoints and figures, written under `results/`.

All paths are relative to the project folder.

To load a checkpoint:

```python
import torch
from models import build_policy
ckpt = torch.load("results/checkpoints/reinforce_transformer_seed7.pt")
policy = build_policy(ckpt["architecture"]["architecture"])
policy.load_state_dict(ckpt["state_dict"])
```

## Hardware and expected runtime

- Results were produced on a cloud Linux VM with 2 vCPUs (Intel Xeon @ 2.30 GHz) and 8 GB RAM. No GPU was used.
- PyTorch runs on the CPU with 1 thread (`torch.set_num_threads(1)`), which keeps runs deterministic.
- Expected runtime: about 35 s in total.
  - About 2 s per MLP run and about 8–10 s per Transformer run.
  - Under 1 s for the evaluation.
  - The rest is figures and the network analysis.

## Reproducibility notes

- Seeds follow the protocol exactly:
  - Python, NumPy and PyTorch are seeded with `s` at the start of each run.
  - Training diffusion seeds are `1,000,000 + 10,000*s + e`.
  - Evaluation diffusion seeds are 900000–900099.
  - Random uses `diffusion_seed + 10,000,000`.
- Spreads, selections, rewards and all CSV values reproduce exactly on re-running.
- Wall-clock time columns (`selection_seconds`, `training_seconds`) vary slightly between runs and machines.
- The supplied CSVs are included unchanged. Their SHA-256 hashes match `data/dataset_README.md`.

## Incomplete requirements

None known. Every item of the assignment protocol is implemented. The learned policies do not
outperform Degree; the assignment states this is not required. `report.pdf` explains why: the
policies stay near-uniform after 50 updates, and the graph is supercritical at p = 0.10.

## Attribution

- **Dataset:** SocioPatterns, *Contacts in a Workplace* (CC Public Domain Dedication). Génois et al. (2015),
  *Network Science* 3(3), 326–347, https://doi.org/10.1017/nws.2015.10
- **REINFORCE:** Williams (1992), *Machine Learning* 8, 229–256.
- `torch.nn.TransformerEncoderLayer` is used as permitted. No reinforcement-learning library, pre-trained
  model or external code was adapted.
