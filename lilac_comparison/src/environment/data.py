import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

import src.environment.dynamics as dynamics
from src.environment.constants import MAX_H
from src.paths import PROJECT_ROOT


class BlockData:

    def __init__(self, output, config=None, data_directory=None):
        if data_directory is None:
            data_directory = PROJECT_ROOT / "data"
        else:
            data_directory = Path(data_directory)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        (output / "schedule_cache").mkdir(exist_ok=True)
        self._load_benchmark(config, data_directory)
        self._build_graph_tables()
        self._build_route_pairs(data_directory)
        self.cache = {}

    def _load_benchmark(self, config, data_directory):
        if config is None:
            config_path = data_directory / "environment.json"
            self.config = json.loads(config_path.read_text())
        else:
            self.config = dict(config)

        with (data_directory / "spatial_graph.pkl").open("rb") as stream:
            spatial = pickle.load(stream)

        self.model = dynamics.build_model(spatial["operational_graph"])
        self.params = dynamics.parameters(self.model, self.config)
        exact = json.loads((data_directory / "road_parameters.json").read_text())

        if config is None:
            for road, i in self.model["index"].items():
                road_parameters = exact[str(road)]
                assert int(self.params["travel"][i]) == road_parameters["travel_time_s"]
                assert int(self.params["blocks"][i]) == road_parameters["block_duration_s"]
                assert self.params["base"][i] == road_parameters["base_probability"]

    def _build_graph_tables(self):
        self.nodes = sorted(self.model["graph"], key=str)
        self.index = {node: node_index for node_index, node in enumerate(self.nodes)}
        self.N = len(self.nodes)
        self.G = len(self.model["roads"])
        graph = self.model["graph"]
        self.max_actions = max(dict(graph.out_degree()).values())
        self.successors = np.full((self.N, self.max_actions), -1, dtype=int)
        self.groups = np.full_like(self.successors, -1)
        self.edge_times = np.zeros_like(self.successors, dtype=np.float32)
        self.edges = {}
        weighted = nx.DiGraph()
        weighted.add_nodes_from(self.nodes)

        for source in self.nodes:
            outgoing_edges = graph.out_edges(source, keys=True, data=True)
            ordered_edges = sorted(outgoing_edges, key=lambda e: e[2])

            for action, (_, target, key, edge) in enumerate(ordered_edges):
                node_index = self.index[source]
                road_index = self.model["index"][edge["road"]]
                self.successors[node_index, action] = self.index[target]
                self.groups[node_index, action] = road_index
                self.edge_times[node_index, action] = self.params["travel"][road_index]
                self.edges[node_index, action] = (source, target, key, edge)
                cost = int(self.params["travel"][road_index])

                if (
                    not weighted.has_edge(source, target)
                    or cost < weighted[source][target]["weight"]
                ):
                    weighted.add_edge(source, target, weight=cost)

        self.distances = np.full((self.N, self.N), 1000000.0, dtype=np.float32)

        for source, distances in nx.all_pairs_dijkstra_path_length(weighted):
            for target, cost in distances.items():
                self.distances[self.index[source], self.index[target]] = cost

    def _build_route_pairs(self, data_directory):
        self.pairs = json.loads((data_directory / "pairs.json").read_text())
        self.test_pairs = []

        for pair in self.pairs:
            start = self.index["N" + str(pair["start_node"])]
            goal = self.index["N" + str(pair["goal_node"])]
            self.test_pairs.append((start, goal))

        departure_rng = np.random.default_rng(43001)
        departure_times = np.linspace(0, 2400, 30, dtype=int)
        self.departures = departure_rng.permutation(departure_times)
        junctions = [self.index[node] for node in self.nodes if node.startswith("N")]
        excluded = set(self.test_pairs)
        self.train_pairs = []

        for start in junctions:
            for goal in junctions:
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
            path = self.output / "schedule_cache" / ("seed_%s.npz" % seed)

            if path.exists():
                with np.load(path) as archive:
                    data = {
                        k: archive[k].copy()
                        for k in ["events", "blocked", "next_open", "warmup_events"]
                    }
            else:
                full = dynamics.simulate(
                    self.model,
                    self.config,
                    self.params,
                    horizon=7200,
                    warmup=4096,
                    seed=seed,
                )
                data = {
                    k: full[k]
                    for k in ["events", "blocked", "next_open", "warmup_events"]
                }
                np.savez_compressed(path, **data)

            recent_warmup = data["warmup_events"][-MAX_H:]
            data["history"] = np.vstack([recent_warmup, data["events"]])
            data["horizon"] = 7200
            self.cache[seed] = data

        return self.cache[seed]

    def specs(self, seeds):
        specs = []

        for seed in seeds:
            trips = zip(self.test_pairs, self.departures, self.pairs)

            for (start, goal), departure, pair in trips:
                spec = (start, goal, seed, int(departure), pair["pair"])
                specs.append(spec)

        return specs
