"""Run every required experiment for SE4095 Assignment 03.

Usage:
    python main.py --output results

Author: M. Waqas Chohan (23F-3041)

Pipeline
    1. Load the fixed 92-participant workplace graph and run the network analysis.
    2. Verify the Independent Cascade simulator and the environment (checks.json).
    3. Train REINFORCE MLP and REINFORCE Transformer with seeds 7, 21, 42 (six runs).
    4. Evaluate Random, Degree and the six final checkpoints on diffusion seeds
       900000-900099 (800 evaluation episodes).
    5. Write the required CSV files, checkpoints, figures and descriptive tables.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

import analysis
from environment import (ACTIVATION_PROBABILITY, BUDGET, IndependentCascadeSimulator,
                         SeedSelectionEnv, load_workplace_graph, sorted_undirected_edges)
from models import action_distribution, build_policy, greedy_action
from reinforce import TRAINING_EPISODES, train_reinforce

TRAIN_SEEDS = (7, 21, 42)
ARCHITECTURES = ("mlp", "transformer")
EVAL_SEEDS = tuple(range(900_000, 900_100))
RANDOM_SEED_OFFSET = 10_000_000


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def run_simulator_checks(graph: nx.Graph) -> dict:
    sim = IndependentCascadeSimulator(graph, ACTIVATION_PROBABILITY)
    sim_p0 = IndependentCascadeSimulator(graph, 0.0)
    sim_p1 = IndependentCascadeSimulator(graph, 1.0)
    test_sets = [[0], [5, 17, 40], [3, 3, 64, 64, 91], [10, 20, 30, 40, 50]]

    empty = sim.spread([], diffusion_seed=123) == 0
    p0 = all(sim_p0.spread(s, 99) == len(set(s)) for s in test_sets)
    p1 = all(sim_p1.spread(s, 99) == len(set().union(*(nx.node_connected_component(graph, u)
                                                       for u in s)))
             for s in test_sets)
    repeat = all(sim.spread(s, seed) == sim.spread(s, seed)
                 for s in test_sets for seed in (1, 900_000, 1_070_999))
    arcs_ok = sim.num_arcs == 2 * graph.number_of_edges()
    return {
        "diffusion_empty_seed_set_zero_spread": bool(empty),
        "diffusion_probability_0_only_distinct_seeds": bool(p0),
        "diffusion_probability_1_all_reachable_active": bool(p1),
        "diffusion_same_inputs_same_spread": bool(repeat),
        "diffusion_two_arcs_per_undirected_edge": bool(arcs_ok),
    }


def run_environment_checks(graph: nx.Graph) -> dict:
    env = SeedSelectionEnv(graph)
    n = env.num_nodes

    # Duplicate (and out-of-range) actions must raise instead of being replaced.
    env.reset(1)
    env.step(4)
    try:
        env.step(4)
        duplicate_rejected = False
    except ValueError:
        duplicate_rejected = True
    try:
        env.step(n)
        out_of_range_rejected = False
    except ValueError:
        out_of_range_rejected = True

    # Five distinct users, terminal reward = spread / N, zero intermediate rewards.
    obs, mask = env.reset(2024)
    shape_ok = obs.shape == (n, 5) and mask.dtype == bool and mask.all()
    rewards = []
    for a in (0, 11, 22, 33, 44):
        obs, mask, r, done, info = env.step(a)
        rewards.append(r)
    five_distinct = len(set(env.selected)) == BUDGET and done and int((~mask).sum()) == BUDGET
    expected = IndependentCascadeSimulator(graph).spread([0, 11, 22, 33, 44], 2024)
    reward_ok = (rewards[:4] == [0.0] * 4 and math.isclose(rewards[4], expected / n)
                 and info["final_spread"] == expected)
    try:
        env.step(55)
        step_after_done_rejected = False
    except RuntimeError:
        step_after_done_rejected = True

    # Masked actions receive exactly zero probability under both policies.
    masked_zero = True
    for arch in ARCHITECTURES:
        torch.manual_seed(0)
        policy = build_policy(arch)
        obs, mask = env.reset(5)
        for a in (2, 9, 60):
            obs, mask, *_ = env.step(a)
        with torch.no_grad():
            dist = action_distribution(policy, torch.from_numpy(obs), torch.from_numpy(mask))
        probs = dist.probs
        masked_zero &= bool((probs[~torch.from_numpy(mask)] == 0).all())
        masked_zero &= bool(math.isclose(float(probs.sum()), 1.0, rel_tol=1e-5))
    return {
        "env_observation_shape_and_mask": bool(shape_ok),
        "env_duplicate_action_rejected": bool(duplicate_rejected),
        "env_out_of_range_action_rejected": bool(out_of_range_rejected),
        "env_step_after_termination_rejected": bool(step_after_done_rejected),
        "env_five_distinct_selected_users": bool(five_distinct),
        "env_reward_equals_spread_over_N": bool(reward_ok),
        "policy_zero_probability_for_masked_actions": bool(masked_zero),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def degree_selection(env: SeedSelectionEnv) -> list[int]:
    """Five highest-degree users; ties broken by the smaller node id."""
    order = sorted(range(env.num_nodes), key=lambda v: (-env.degree[v], v))
    return order[:BUDGET]


def evaluate_fixed(env: SeedSelectionEnv, method: str) -> list[dict]:
    rows = []
    for seed in EVAL_SEEDS:
        env.reset(seed)
        start = time.perf_counter()
        if method == "random":
            rng = np.random.default_rng(seed + RANDOM_SEED_OFFSET)
            actions = [int(a) for a in rng.choice(env.num_nodes, size=BUDGET, replace=False)]
        else:
            actions = degree_selection(env)
        for a in actions:
            _, _, reward, done, info = env.step(a)
        elapsed = time.perf_counter() - start - info["diffusion_seconds"]
        rows.append({"method": method, "train_seed": -1, "diffusion_seed": seed,
                     "selected_nodes": json.dumps(info["selected_nodes"]),
                     "final_spread": info["final_spread"], "total_reward": reward,
                     "selection_seconds": elapsed})
    return rows


def evaluate_policy(env: SeedSelectionEnv, policy: torch.nn.Module, method: str,
                    train_seed: int) -> list[dict]:
    rows = []
    policy.eval()
    with torch.no_grad():
        for seed in EVAL_SEEDS:
            obs, mask = env.reset(seed)
            start = time.perf_counter()
            done = False
            while not done:
                a = greedy_action(policy, torch.from_numpy(obs), torch.from_numpy(mask))
                obs, mask, reward, done, info = env.step(a)
            elapsed = time.perf_counter() - start - info["diffusion_seconds"]
            rows.append({"method": method, "train_seed": train_seed, "diffusion_seed": seed,
                         "selected_nodes": json.dumps(info["selected_nodes"]),
                         "final_spread": info["final_spread"], "total_reward": reward,
                         "selection_seconds": elapsed})
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="SE4095 Assignment 03 experiments")
    parser.add_argument("--output", default="results", help="output directory")
    parser.add_argument("--data", default="data", help="directory with the supplied CSVs")
    args = parser.parse_args()

    torch.set_num_threads(1)  # deterministic, and fastest for these tiny models on CPU
    device = "cpu"
    out = Path(args.output)
    figures = out / "figures"
    checkpoints = out / "checkpoints"
    for d in (out, figures, checkpoints):
        d.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()

    # 1. Network analysis ------------------------------------------------------
    print("[1/5] Network analysis")
    wg = load_workplace_graph(args.data)
    graph = wg.graph
    dept_map = dict(zip(wg.departments["node_id"].astype(int), wg.departments["department"]))
    pd.DataFrame(sorted_undirected_edges(graph), columns=["source", "target"]).to_csv(
        out / "graph_edges.csv", index=False)
    stats = analysis.network_statistics(graph)
    (out / "network_stats.json").write_text(json.dumps(stats, indent=2))
    table = analysis.node_table(graph, wg.nodes, wg.departments)
    table.to_csv(out / "node_metrics.csv", index=False)
    top10 = analysis.top_degree_table(table)
    top10.to_csv(out / "top10_degree_centrality.csv", index=False)
    analysis.plot_network(graph, dept_map, figures / "network_departments.png",
                          top_nodes=top10["node_id"].tolist())
    analysis.plot_degree_distribution(graph, figures / "degree_clustering.png")
    print(f"      {stats['participants']} participants, {stats['edges']} edges, "
          f"density {stats['density']:.4f}, components {stats['connected_components']}")

    # 2. Verification -----------------------------------------------------------
    print("[2/5] Simulator and environment checks")
    checks = {**run_simulator_checks(graph), **run_environment_checks(graph)}
    for name, ok in checks.items():
        print(f"      {'PASS' if ok else 'FAIL'}  {name}")

    # 3. Training ----------------------------------------------------------------
    print("[3/5] REINFORCE training (2 architectures x 3 seeds x 1,000 episodes)")
    env = SeedSelectionEnv(graph)
    training_rows, train_times, policies, diagnostics = [], {}, {}, {}
    for arch in ARCHITECTURES:
        method = f"reinforce_{arch}"
        for seed in TRAIN_SEEDS:
            policy, records, seconds, diag = train_reinforce(env, arch, seed, device=device)
            for r in records:
                training_rows.append({"method": method, **r})
            train_times[(method, seed)] = seconds
            policies[(method, seed)] = policy
            diagnostics[f"{method}_seed{seed}"] = diag
            torch.save({"method": method, "train_seed": seed,
                        "training_episodes": TRAINING_EPISODES,
                        "architecture": policy.config(),
                        "state_dict": policy.state_dict(),
                        "final_baseline": diag["final_baseline"],
                        "training_seconds": seconds,
                        "torch_version": str(torch.__version__)},
                       checkpoints / f"{method}_seed{seed}.pt")
            print(f"      {method} seed {seed}: {seconds:.1f} s")
    training = pd.DataFrame(training_rows,
                            columns=["method", "train_seed", "episode", "final_spread",
                                     "total_reward"])
    training.to_csv(out / "training.csv", index=False)
    checks["training_losses_and_gradients_finite"] = all(d["all_finite"]
                                                         for d in diagnostics.values())
    checks["training_50_updates_per_run"] = all(d["num_updates"] == 50
                                                for d in diagnostics.values())

    # Descriptive diagnostic: how far did each policy move from its initialization?
    # (first-step distribution on the empty selection; not used for any tuning)
    diag_rows = []
    obs0, mask0 = env.reset(EVAL_SEEDS[0])
    obs0_t, mask0_t = torch.from_numpy(obs0), torch.from_numpy(mask0)
    for (method, seed), policy in policies.items():
        torch.manual_seed(seed)  # rebuilds the exact initial weights of run `seed`
        initial = build_policy(method.replace("reinforce_", ""))
        for stage, net in (("initial", initial), ("final", policy)):
            net.eval()
            with torch.no_grad():
                probs = action_distribution(net, obs0_t, mask0_t).probs.numpy()
            diag_rows.append({"method": method, "train_seed": seed, "stage": stage,
                              "max_probability": probs.max(), "min_probability": probs.min(),
                              "entropy": float(-(probs * np.log(probs)).sum()),
                              "uniform_entropy": float(np.log(len(probs))),
                              "corr_probability_degree": float(np.corrcoef(probs, env.degree)[0, 1]),
                              "top_probability_node": int(probs.argmax())})
    pd.DataFrame(diag_rows).to_csv(out / "policy_diagnostics.csv", index=False)

    # 4. Evaluation ---------------------------------------------------------------
    print("[4/5] Controlled evaluation on diffusion seeds 900000-900099")
    eval_rows = evaluate_fixed(env, "random") + evaluate_fixed(env, "degree")
    for (method, seed), policy in policies.items():
        eval_rows += evaluate_policy(env, policy, method, seed)
    evaluation = pd.DataFrame(eval_rows)
    evaluation.to_csv(out / "evaluation.csv", index=False)

    n = graph.number_of_nodes()
    sel_sets = evaluation["selected_nodes"].map(json.loads)
    checks["eval_800_episodes"] = len(evaluation) == 800
    checks["eval_five_distinct_users_every_episode"] = bool(
        sel_sets.map(lambda s: len(s) == BUDGET and len(set(s)) == BUDGET).all())
    checks["eval_reward_equals_spread_over_N"] = bool(
        np.allclose(evaluation["total_reward"], evaluation["final_spread"] / n))

    summary_rows = []
    for (method, seed), grp in evaluation.groupby(["method", "train_seed"], sort=False):
        summary_rows.append({
            "method": method, "train_seed": seed,
            "spread_mean": grp["final_spread"].mean(),
            "spread_std": grp["final_spread"].std(ddof=0),  # population std
            "reward_mean": grp["total_reward"].mean(),
            "selection_seconds_mean": grp["selection_seconds"].mean(),
            "training_seconds": train_times.get((method, seed), 0.0),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)

    arch_rows = []
    for method in ("reinforce_mlp", "reinforce_transformer"):
        means = summary[summary["method"] == method]["spread_mean"].to_numpy()
        times = summary[summary["method"] == method]["training_seconds"].to_numpy()
        arch_rows.append({"method": method, "runs": len(means),
                          "mean_of_run_spread_means": means.mean(),
                          "population_std_of_run_spread_means": means.std(ddof=0),
                          "mean_training_seconds": times.mean()})
    pd.DataFrame(arch_rows).to_csv(out / "architecture_summary.csv", index=False)

    # Selected users and department coverage (descriptive only).
    selection_rows = []
    deg_sel = degree_selection(env)
    selection_rows.append({"method": "degree", "train_seed": -1,
                           "selected_nodes": json.dumps(deg_sel),
                           "departments": json.dumps([dept_map[v] for v in deg_sel]),
                           "department_coverage": analysis.department_coverage(deg_sel, dept_map),
                           "distinct_selections_over_100": 1})
    for (method, seed), grp in evaluation[evaluation["method"].str.startswith("reinforce")] \
            .groupby(["method", "train_seed"], sort=False):
        distinct = grp["selected_nodes"].unique()
        nodes = json.loads(distinct[0])
        selection_rows.append({"method": method, "train_seed": seed,
                               "selected_nodes": json.dumps(nodes),
                               "departments": json.dumps([dept_map[v] for v in nodes]),
                               "department_coverage": analysis.department_coverage(nodes, dept_map),
                               "distinct_selections_over_100": len(distinct)})
    rand = evaluation[evaluation["method"] == "random"]["selected_nodes"].map(json.loads)
    rand_cov = rand.map(lambda s: analysis.department_coverage(s, dept_map))
    selection_rows.append({"method": "random", "train_seed": -1,
                           "selected_nodes": "varies per episode", "departments": "varies",
                           "department_coverage": rand_cov.mean(),
                           "distinct_selections_over_100": rand.map(tuple).nunique()})
    pd.DataFrame(selection_rows).to_csv(out / "selected_users.csv", index=False)

    # Union coverage of the seeds' 1-hop neighbourhoods (overlapping audiences).
    overlap_rows = []
    for row in selection_rows:
        if row["method"] == "random":
            continue
        nodes = json.loads(row["selected_nodes"])
        neigh = [set(graph.neighbors(v)) for v in nodes]
        union = set(nodes).union(*neigh)
        overlap_rows.append({"method": row["method"], "train_seed": row["train_seed"],
                             "sum_of_degrees": int(sum(len(s) for s in neigh)),
                             "closed_neighbourhood_union": len(union),
                             "edges_among_seeds": graph.subgraph(nodes).number_of_edges()})
    pd.DataFrame(overlap_rows).to_csv(out / "seed_neighbourhood_overlap.csv", index=False)

    # 5. Figures --------------------------------------------------------------------
    print("[5/5] Figures")
    reference = {m: summary[summary["method"] == m]["spread_mean"].iloc[0]
                 for m in ("random", "degree")}
    analysis.plot_learning_curves(training, figures / "learning_curves.png", reference=reference)
    analysis.plot_final_spread(evaluation, figures / "final_spread_comparison.png")
    tr7 = json.loads(evaluation[(evaluation["method"] == "reinforce_transformer")
                                & (evaluation["train_seed"] == 7)]["selected_nodes"].iloc[0])
    analysis.plot_selection(graph, dept_map, tr7, deg_sel,
                            figures / "transformer_seed7_selection.png",
                            "REINFORCE Transformer selection (training seed 7)\n"
                            "stars = seeds in selection order, highlighted = direct neighbours")

    checks_out = {
        "all_passed": bool(all(checks.values())),
        "checks": {k: ("pass" if v else "fail") for k, v in checks.items()},
        "training_diagnostics": {k: {"num_updates": v["num_updates"],
                                     "all_finite": v["all_finite"],
                                     "max_abs_loss": max(abs(x) for x in v["losses"]),
                                     "max_grad_norm_before_clip": max(v["grad_norms"]),
                                     "final_baseline": v["final_baseline"]}
                                 for k, v in diagnostics.items()},
    }
    (out / "checks.json").write_text(json.dumps(checks_out, indent=2))
    (out / "run_info.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "networkx": nx.__version__, "device": device, "torch_threads": 1,
        "total_wall_seconds": time.perf_counter() - wall_start,
    }, indent=2))

    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nAll checks passed: {checks_out['all_passed']}")
    print(f"Total runtime: {time.perf_counter() - wall_start:.1f} s")


if __name__ == "__main__":
    main()
