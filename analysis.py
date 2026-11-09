"""Network analysis and figure generation (helper module for main.py).

SE4095 Social Network Theory and Practice - Assignment 03
Author: M. Waqas Chohan (23F-3041)

Department labels are used here only for visualization and the final
descriptive analysis; they never reach the policies, rewards or tuning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LAYOUT_SEED = 3041
DEPT_COLORS = {"DISQ": "#4C72B0", "DMCT": "#DD8452", "DSE": "#55A868",
               "SFLE": "#C44E52", "SRH": "#8172B3"}
METHOD_LABELS = {"random": "Random", "degree": "Degree",
                 "reinforce_mlp": "REINFORCE MLP",
                 "reinforce_transformer": "REINFORCE Transformer"}
METHOD_COLORS = {"random": "#9AA5B1", "degree": "#E1A33A",
                 "reinforce_mlp": "#3B7DD8", "reinforce_transformer": "#C0392B"}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def network_statistics(graph: nx.Graph) -> dict:
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    components = list(nx.connected_components(graph))
    degrees = np.array([d for _, d in graph.degree()])
    clustering = nx.clustering(graph)
    largest = graph.subgraph(max(components, key=len))
    return {
        "participants": n,
        "edges": m,
        "density": nx.density(graph),
        "connected_components": len(components),
        "largest_component_size": len(largest),
        "isolated_participants": int((degrees == 0).sum()),
        "degree_min": int(degrees.min()),
        "degree_max": int(degrees.max()),
        "degree_mean": float(degrees.mean()),
        "degree_median": float(np.median(degrees)),
        "average_clustering": float(np.mean(list(clustering.values()))),
        "transitivity": nx.transitivity(graph),
        "diameter_largest_component": nx.diameter(largest),
        "average_shortest_path_largest_component": nx.average_shortest_path_length(largest),
    }


def node_table(graph: nx.Graph, nodes: pd.DataFrame, departments: pd.DataFrame) -> pd.DataFrame:
    n = graph.number_of_nodes()
    centrality = nx.degree_centrality(graph)
    clustering = nx.clustering(graph)
    table = pd.DataFrame({
        "node_id": range(n),
        "degree": [graph.degree(i) for i in range(n)],
        "degree_centrality": [centrality[i] for i in range(n)],
        "clustering_coefficient": [clustering[i] for i in range(n)],
    })
    table = table.merge(nodes, on="node_id").merge(departments, on="node_id")
    return table


def top_degree_table(table: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    ranked = table.sort_values(["degree", "node_id"], ascending=[False, True]).head(k).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[["rank", "node_id", "original_id", "department", "degree",
                   "degree_centrality", "clustering_coefficient"]]


def department_coverage(selected, dept_map: dict) -> int:
    return len({dept_map[int(u)] for u in selected})


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _layout(graph: nx.Graph):
    return nx.spring_layout(graph, seed=LAYOUT_SEED, k=0.35, iterations=200)


def _dept_legend(ax, dept_map):
    present = sorted(set(dept_map.values()))
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DEPT_COLORS[d],
                          markersize=9, label=d) for d in present]
    return handles


def plot_network(graph: nx.Graph, dept_map: dict, path: Path, top_nodes=()):
    pos = _layout(graph)
    deg = dict(graph.degree())
    fig, ax = plt.subplots(figsize=(8.5, 7))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.12, width=0.6, edge_color="#34495E")
    nx.draw_networkx_nodes(graph, pos, ax=ax,
                           node_color=[DEPT_COLORS[dept_map[v]] for v in graph.nodes()],
                           node_size=[25 + 9 * deg[v] for v in graph.nodes()],
                           edgecolors="white", linewidths=0.8)
    nx.draw_networkx_labels(graph, pos, labels={v: str(v) for v in top_nodes}, ax=ax,
                            font_size=8, font_weight="bold")
    ax.legend(handles=_dept_legend(ax, dept_map), title="Department", loc="lower left",
              frameon=False)
    ax.set_title("Workplace contact network (92 participants, 755 edges)\n"
                 "node size = degree, colour = department, labels = top-10 degree",
                 fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_degree_distribution(graph: nx.Graph, path: Path):
    degrees = [d for _, d in graph.degree()]
    clustering = list(nx.clustering(graph).values())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(degrees, bins=range(0, max(degrees) + 3, 2), color="#3B7DD8", edgecolor="white")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Participants")
    axes[0].set_title("Degree distribution")
    axes[1].scatter(degrees, clustering, s=18, color="#C0392B", alpha=0.75)
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Local clustering coefficient")
    axes[1].set_title("Clustering vs degree")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_selection(graph: nx.Graph, dept_map: dict, selected, degree_selected, path: Path,
                   title: str):
    pos = _layout(graph)
    selected = list(selected)
    covered = set(selected) | {v for s in selected for v in graph.neighbors(s)}
    fig, ax = plt.subplots(figsize=(8.5, 7))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.08, width=0.5, edge_color="#7F8C8D")
    seed_edges = [(s, v) for s in selected for v in graph.neighbors(s)]
    nx.draw_networkx_edges(graph, pos, edgelist=seed_edges, ax=ax, alpha=0.45, width=1.0,
                           edge_color="#C0392B")
    others = [v for v in graph.nodes() if v not in selected]
    nx.draw_networkx_nodes(graph, pos, nodelist=others, ax=ax,
                           node_color=[DEPT_COLORS[dept_map[v]] for v in others],
                           node_size=[70 if v in covered else 35 for v in others],
                           alpha=[1.0 if v in covered else 0.35 for v in others],
                           edgecolors="white", linewidths=0.6)
    nx.draw_networkx_nodes(graph, pos, nodelist=selected, ax=ax,
                           node_color=[DEPT_COLORS[dept_map[v]] for v in selected],
                           node_size=420, node_shape="*", edgecolors="black", linewidths=1.4)
    degree_only = [v for v in degree_selected if v not in selected]
    if degree_only:
        nx.draw_networkx_nodes(graph, pos, nodelist=degree_only, ax=ax, node_color="none",
                               node_size=260, edgecolors="#E1A33A", linewidths=2.2)
    for i, v in enumerate(selected):
        x, y = pos[v]
        ax.annotate(f"#{i + 1}  node {v}", (x, y), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#555555",
                          "lw": 0.6, "alpha": 0.9})
    handles = _dept_legend(ax, dept_map) + [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#BBBBBB",
                   markeredgecolor="black", markersize=14, label="Transformer seed"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#E1A33A", markeredgewidth=2, markersize=11,
                   label="Degree seed (not chosen)"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(training: pd.DataFrame, path: Path, block: int = 20,
                         reference: dict | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, method in zip(axes, ["reinforce_mlp", "reinforce_transformer"]):
        sub = training[training["method"] == method]
        curves = []
        for seed, colour in zip(sorted(sub["train_seed"].unique()), ["#1f77b4", "#2ca02c", "#9467bd"]):
            spreads = sub[sub["train_seed"] == seed].sort_values("episode")["final_spread"].to_numpy()
            blocks = spreads.reshape(-1, block).mean(axis=1)
            curves.append(blocks)
            x = np.arange(1, len(blocks) + 1) * block
            ax.plot(x, blocks, color=colour, alpha=0.55, lw=1.2, label=f"seed {seed}")
        mean_curve = np.mean(curves, axis=0)
        ax.plot(x, mean_curve, color="black", lw=2.2, label="mean of 3 seeds")
        if reference:
            for name, value in reference.items():
                ax.axhline(value, ls="--", lw=1.2, color=METHOD_COLORS[name],
                           label=f"{METHOD_LABELS[name]} (eval mean)")
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Training episode (end of 20-episode block)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean final spread per block")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center",
               ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Learning curves (non-overlapping blocks of 20 training episodes)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_final_spread(evaluation: pd.DataFrame, path: Path):
    groups, labels, colours = [], [], []
    for method in ["random", "degree", "reinforce_mlp", "reinforce_transformer"]:
        sub = evaluation[evaluation["method"] == method]
        for seed in sorted(sub["train_seed"].unique()):
            groups.append(sub[sub["train_seed"] == seed]["final_spread"].to_numpy())
            short = METHOD_LABELS[method].replace("REINFORCE ", "")
            labels.append(short if seed < 0 else f"{short}\nseed {seed}")
            colours.append(METHOD_COLORS[method])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    box = ax.boxplot(groups, patch_artist=True, widths=0.55, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "white",
                                "markeredgecolor": "black", "markersize": 5},
                     medianprops={"color": "black"})
    for patch, colour in zip(box["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.8)
    for i, g in enumerate(groups, start=1):
        ax.text(i, g.max() + 0.6, f"{g.mean():.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Final spread (participants)")
    ax.set_title("Final spread on the same 100 evaluation realizations "
                 "(diamond and number = mean)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
