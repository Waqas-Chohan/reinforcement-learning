"""Graph loading, Independent Cascade simulator and the seed-selection RL environment.

SE4095 Social Network Theory and Practice - Assignment 03
Author: M. Waqas Chohan (23F-3041)

Everything in this module follows the fixed protocol of the assignment:

* the supplied 92-node workplace graph is used unchanged (undirected, unweighted);
* the Independent Cascade Model (ICM) is implemented as a live-arc simulation
  with one uniform draw per directed arc, in (source, target) order;
* the environment exposes ``reset(diffusion_seed)`` and ``step(action)`` with an
  N x 5 observation, a Boolean action mask and a single terminal reward
  R = final_spread / N after the fifth selection.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ACTIVATION_PROBABILITY = 0.10
BUDGET = 5
NUM_FEATURES = 5
FEATURE_NAMES = (
    "degree_centrality",
    "clustering_coefficient",
    "selected_indicator",
    "selected_neighbor_fraction",
    "remaining_budget_fraction",
)


# --------------------------------------------------------------------------- #
# Graph loading
# --------------------------------------------------------------------------- #
@dataclass
class WorkplaceGraph:
    """Container for the fixed workplace contact graph."""

    graph: nx.Graph
    nodes: pd.DataFrame
    departments: pd.DataFrame

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()


def load_workplace_graph(data_dir: str | Path = "data") -> WorkplaceGraph:
    """Load nodes first, then edges, exactly as the dataset README instructs.

    Every participant in ``nodes.csv`` is added even if it has no contact,
    self-loops are discarded and each undirected pair is stored once.
    """
    data_dir = Path(data_dir)
    nodes = pd.read_csv(data_dir / "nodes.csv")
    edges = pd.read_csv(data_dir / "edges.csv")
    departments = pd.read_csv(data_dir / "departments.csv")

    node_ids = sorted(int(n) for n in nodes["node_id"])
    if node_ids != list(range(len(node_ids))):
        raise ValueError("node_id values must be consecutive integers starting at 0")

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for u, v in edges[["source", "target"]].itertuples(index=False):
        u, v = int(u), int(v)
        if u == v:  # defensive: the package contains no self-loops
            continue
        graph.add_edge(u, v)  # nx.Graph keeps one copy of duplicate pairs

    return WorkplaceGraph(graph=graph, nodes=nodes, departments=departments)


def sorted_undirected_edges(graph: nx.Graph) -> list[tuple[int, int]]:
    """Each undirected pair once with source < target, sorted lexicographically."""
    return sorted((min(u, v), max(u, v)) for u, v in graph.edges())


# --------------------------------------------------------------------------- #
# Independent Cascade simulator
# --------------------------------------------------------------------------- #
class IndependentCascadeSimulator:
    """Reproducible live-arc implementation of the Independent Cascade Model.

    In ICM each newly activated user gets exactly one chance to activate each
    inactive neighbour with probability p. Because every attempt is an
    independent coin flip that is used at most once, flipping all coins in
    advance gives the same distribution: an arc is *live* with probability p
    and the final active set is everything reachable from the seeds through
    live arcs. Live arcs and random numbers never leave this class.
    """

    def __init__(self, graph: nx.Graph, probability: float = ACTIVATION_PROBABILITY):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("activation probability must lie in [0, 1]")
        self.num_nodes = graph.number_of_nodes()
        self.probability = float(probability)

        # 1. Every undirected edge becomes two directed arcs, sorted by (source, target).
        arcs = sorted({(u, v) for a, b in graph.edges() for u, v in ((a, b), (b, a))})
        self._arcs = np.asarray(arcs, dtype=np.int64).reshape(-1, 2)
        self.num_arcs = len(self._arcs)

        # CSR-style index so that the out-arcs of node u are a contiguous slice.
        sources = self._arcs[:, 0]
        self._targets = self._arcs[:, 1]
        self._offsets = np.searchsorted(sources, np.arange(self.num_nodes + 1))

    def _live_arcs(self, diffusion_seed: int) -> np.ndarray:
        # 2. One uniform draw per arc, in the stored arc order.
        rng = np.random.default_rng(diffusion_seed)
        draws = rng.random(self.num_arcs)
        # 3. An arc is live when its draw is strictly less than p.
        return draws < self.probability

    def spread(self, seeds, diffusion_seed: int) -> int:
        """Number of users reachable from ``seeds`` through live arcs (seeds included)."""
        seeds = {int(s) for s in seeds}
        if not seeds:
            return 0
        for s in seeds:
            if not 0 <= s < self.num_nodes:
                raise ValueError(f"seed {s} is not a valid node id")

        live = self._live_arcs(diffusion_seed)
        active = np.zeros(self.num_nodes, dtype=bool)
        queue = deque(sorted(seeds))
        active[list(seeds)] = True
        while queue:  # breadth-first cascade: each active node tries its arcs once
            u = queue.popleft()
            start, end = self._offsets[u], self._offsets[u + 1]
            for idx in range(start, end):
                if live[idx]:
                    v = self._targets[idx]
                    if not active[v]:
                        active[v] = True
                        queue.append(v)
        return int(active.sum())


# --------------------------------------------------------------------------- #
# Reinforcement-learning environment
# --------------------------------------------------------------------------- #
class SeedSelectionEnv:
    """Budgeted sequential seed selection on the fixed workplace graph.

    Observation: ``N x 5`` float32 matrix in ascending node-ID order.
    Mask:        Boolean vector, True for participants that can still be chosen.
    Action:      integer node id in [0, N-1] that has not been selected.
    Reward:      0 for actions 1-4, final_spread / N after action 5.
    """

    def __init__(self, graph: nx.Graph, budget: int = BUDGET,
                 probability: float = ACTIVATION_PROBABILITY):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.budget = budget
        self.simulator = IndependentCascadeSimulator(graph, probability)

        # Static features are computed once on the fixed graph.
        n = self.num_nodes
        self.degree = np.array([graph.degree(i) for i in range(n)], dtype=np.float64)
        self.degree_centrality = self.degree / (n - 1)
        clustering = nx.clustering(graph)
        self.clustering = np.array([clustering[i] for i in range(n)], dtype=np.float64)
        self.adjacency = nx.to_numpy_array(graph, nodelist=range(n), dtype=np.float64)

        self._diffusion_seed: int | None = None
        self.selected: list[int] = []
        self._selected_flag = np.zeros(n, dtype=bool)
        self.done = True

    # -- helpers ------------------------------------------------------------ #
    @property
    def remaining_budget(self) -> int:
        return self.budget - len(self.selected)

    def action_mask(self) -> np.ndarray:
        return ~self._selected_flag

    def observation(self) -> np.ndarray:
        n = self.num_nodes
        selected = self._selected_flag.astype(np.float64)
        selected_neighbors = self.adjacency @ selected
        with np.errstate(divide="ignore", invalid="ignore"):
            neighbor_fraction = np.where(self.degree > 0, selected_neighbors / self.degree, 0.0)
        features = np.empty((n, NUM_FEATURES), dtype=np.float32)
        features[:, 0] = self.degree_centrality
        features[:, 1] = self.clustering
        features[:, 2] = selected
        features[:, 3] = neighbor_fraction
        features[:, 4] = self.remaining_budget / self.budget
        return features

    # -- API ---------------------------------------------------------------- #
    def reset(self, diffusion_seed: int):
        """Start a new episode whose diffusion realization is fixed by ``diffusion_seed``."""
        self._diffusion_seed = int(diffusion_seed)
        self.selected = []
        self._selected_flag[:] = False
        self.done = False
        return self.observation(), self.action_mask()

    def step(self, action):
        """Apply one selection. Returns (observation, mask, reward, done, info)."""
        if self.done:
            raise RuntimeError("episode has terminated; call reset() first")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError(f"action must be an integer node id, got {action!r}")
        action = int(action)
        if not 0 <= action < self.num_nodes:
            raise ValueError(f"action {action} is outside [0, {self.num_nodes - 1}]")
        if self._selected_flag[action]:
            raise ValueError(f"action {action} was already selected (duplicate action)")

        self.selected.append(action)
        self._selected_flag[action] = True

        reward, info = 0.0, {}
        if len(self.selected) == self.budget:
            # Diffusion runs only once, after the fifth selection.
            t0 = time.perf_counter()
            spread = self.simulator.spread(self.selected, self._diffusion_seed)
            diffusion_seconds = time.perf_counter() - t0
            reward = spread / self.num_nodes
            self.done = True
            info = {"final_spread": spread, "selected_nodes": list(self.selected),
                    "diffusion_seconds": diffusion_seconds}
        return self.observation(), self.action_mask(), reward, self.done, info
