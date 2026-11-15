import json
from pathlib import Path

import networkx as nx
import numpy as np

from src.environment.events import sample_coupled
from src.environment.schedule import Schedule
from src.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


class RoutingData:

    def __init__(self, output):
        self.output = Path(output)
        (self.output / "schedule_cache").mkdir(parents=True, exist_ok=True)
        self.model = np.load(ROOT / "data/routing_model.npz")
        source = np.load(ROOT / "data/edge_lengths.npz")
        meta = json.loads((ROOT / "data/routes.json").read_text())
        self._build_graph_tables(meta, source)
        self._build_route_pairs(meta)
        self.cache = {}

    def _build_graph_tables(self, metadata, source):
        self.nodes = [node["node_id"] for node in metadata["node_mapping"]]
        self.index = {node: node_index for node_index, node in enumerate(self.nodes)}
        self.edges = [tuple(map(int, e)) for e in self.model["edges"]]
        self.edge_index = {e: node_index for node_index, e in enumerate(self.edges)}
        self.lengths = source["lengths_m"]
        self.mapping = self.model["mapping"]
        self.N = len(self.nodes)
        self.G = len(self.model["base_speeds_kmh"])
        self.graph = nx.DiGraph(self.edges)
        edge_speeds = self.model["base_speeds_kmh"][self.mapping]
        free_flow_times = 3.6 * self.lengths / edge_speeds
        weights = dict(zip(self.edges, free_flow_times))
        nx.set_edge_attributes(self.graph, weights, "time_s")
        self.max_actions = max(dict(self.graph.out_degree()).values())
        self.successors = np.full((self.N, self.max_actions), -1, dtype=int)
        self.groups = np.full_like(self.successors, -1)
        self.edge_times = np.zeros_like(self.successors, dtype=np.float32)

        for node, node_index in self.index.items():
            for action, target_node in enumerate(sorted(self.graph.successors(node))):
                self.successors[node_index, action] = self.index[target_node]
                edge_index = self.edge_index[node, target_node]
                self.groups[node_index, action] = self.mapping[edge_index]
                self.edge_times[node_index, action] = weights[node, target_node]

        self.distances = np.full((self.N, self.N), 1000000.0, dtype=np.float32)

        shortest_paths = nx.all_pairs_dijkstra_path_length(
            self.graph, weight="time_s"
        )

        for source_node, distances in shortest_paths:
            for target_node, time in distances.items():
                self.distances[self.index[source_node], self.index[target_node]] = time

    def _build_route_pairs(self, metadata):
        self.test_pairs = []

        for pair in metadata["pairs"]:
            start = self.index[pair["start_id"]]
            goal = self.index[pair["destination_id"]]
            self.test_pairs.append((start, goal))

        excluded = set(self.test_pairs)
        self.train_pairs = []

        for start in range(self.N):
            for goal in range(self.N):
                if start == goal:
                    continue

                reachable = self.distances[start, goal] < 1000000.0
                if not reachable:
                    continue

                pair = (start, goal)
                if pair not in excluded:
                    self.train_pairs.append(pair)

    def schedule(self, seed):
        seed = int(seed)

        if seed not in self.cache:
            path = self.output / "schedule_cache" / f"seed_{seed}.npz"

            if path.exists():
                with np.load(path) as s:
                    events, speeds, distance = (s["events"], s["speeds"], s["distance"])
            else:
                rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
                u = rng.random((6000, self.G))
                events, _ = sample_coupled(
                    u,
                    self.model["probabilities"],
                    self.model["spatial_weights"],
                    30.0,
                    0.7,
                )
                group_schedule = Schedule(
                    [(g, g) for g in range(self.G)],
                    self.model["base_speeds_kmh"],
                    self.model["representative_lengths_m"],
                    events,
                    self.model["recovery_tau_s"],
                    self.model["reductions"],
                )
                speeds, distance = (group_schedule.speeds, group_schedule.distance)
                np.savez_compressed(
                    path, events=events, speeds=speeds, distance=distance
                )

            self.cache[seed] = (events, speeds, distance)

        return self.cache[seed]

    def arrival(self, seed, u, v, t):
        e = self.edge_index[u, v]
        g = self.mapping[e]
        length = self.lengths[e]

        if length == 0:
            return float(t)

        _, speeds, cumulative = self.schedule(seed)
        second = int(np.floor(t))

        if second >= len(speeds):
            raise RuntimeError("Time exceeds saved process horizon")

        target = cumulative[second, g] + (t - second) * speeds[second, g] / 3.6 + length

        if target > cumulative[-1, g]:
            raise RuntimeError("Traversal exceeds saved process horizon")

        stop = int(np.searchsorted(cumulative[:, g], target, side="left"))
        k = max(stop - 1, 0)

        return float(k + (target - cumulative[k, g]) / (speeds[k, g] / 3.6))
