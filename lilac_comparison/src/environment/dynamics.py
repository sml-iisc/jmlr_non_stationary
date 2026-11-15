import itertools
from collections import defaultdict

import networkx as nx
import numpy as np


def build_model(graph):
    expanded = nx.MultiDiGraph()
    road_graph = nx.Graph()
    junction_roads = defaultdict(set)

    for node, attributes in graph.nodes(data=True):
        expanded.add_node(
            f"N{node}", context_node=node, junction_id=attributes["junction_id"]
        )

    for source, destination, action in graph.edges(data=True):
        roads = [(s[1], s[2]) for s in action["word"] if s[0] == "R"]

        assert roads
        road_graph.add_nodes_from(road for road, _ in roads)
        for first_road, next_road in zip(roads, roads[1:]):
            if first_road[0] != next_road[0]:
                road_graph.add_edge(first_road[0], next_road[0])
        junction_roads[graph.nodes[source]["junction_id"]].add(roads[0][0])
        junction_roads[graph.nodes[destination]["junction_id"]].add(roads[-1][0])

        previous = f"N{source}"

        for segment_index, (road, direction) in enumerate(roads):
            if segment_index + 1 == len(roads):
                target = f"N{destination}"
            else:
                target = f"{action['id']}:{segment_index + 1}"

            expanded.add_node(
                target,
                context_node=destination if segment_index + 1 == len(roads) else None,
            )
            expanded.add_edge(
                previous,
                target,
                key=f"{action['id']}/r{segment_index}",
                id=f"{action['id']}/r{segment_index}",
                road=road,
                direction=direction,
                macro=action["id"],
                segment=segment_index,
                last=segment_index + 1 == len(roads),
            )
            previous = target

    for roads in junction_roads.values():
        road_graph.add_edges_from(itertools.combinations(sorted(roads), 2))

    roads = sorted(road_graph)
    index = {road: segment_index for segment_index, road in enumerate(roads)}
    hops = np.full((len(roads), len(roads)), np.inf)

    for road, distances in nx.all_pairs_shortest_path_length(road_graph):
        for neighbor, attributes in distances.items():
            hops[index[road], index[neighbor]] = attributes

    return dict(
        graph=expanded, roads=roads, index=index, hops=hops, road_graph=road_graph
    )


def simulate(model, profile, parameters, horizon=7200, warmup=4096, seed=20261030):
    road_count = len(model["roads"])
    rng = np.random.default_rng(seed)
    memory = np.zeros(road_count)
    blocked_until = np.full(road_count, -warmup - 1, dtype=int)
    memory_decay = np.exp(-1.0 / profile["temporal_decay_s"])

    events = np.zeros((horizon + 1, road_count), dtype=bool)
    blocked = np.zeros_like(events)
    warmup_events = np.zeros((warmup, road_count), dtype=bool)
    initial_memory = None
    initial_deadlines = None
    probabilities = np.zeros((horizon + 1, road_count), dtype=np.float32)

    for second in range(-warmup, horizon + 1):
        if second == 0:
            initial_memory = memory.copy()
            initial_deadlines = blocked_until.copy()

        local_probability = parameters["base"] + profile["self_boost"] * memory
        neighbor_memory = parameters["kernel"] @ memory
        neighbor_probability = profile["neighbor_boost"] * neighbor_memory
        uncapped_probability = local_probability + neighbor_probability
        event_probability = np.minimum(profile["probability_cap"], uncapped_probability)
        event = rng.random(road_count) < event_probability
        extended_deadlines = np.maximum(blocked_until, second + parameters["blocks"])
        blocked_until = np.where(event, extended_deadlines, blocked_until)

        if second >= 0:
            events[second] = event
            blocked[second] = second < blocked_until
            probabilities[second] = event_probability
        else:
            warmup_events[second + warmup] = event

        memory = memory_decay * memory + event.astype(float)

    next_open = np.full((horizon + 2, road_count), horizon + 1, dtype=np.int32)

    for second in range(horizon, -1, -1):
        next_open[second] = np.where(~blocked[second], second, next_open[second + 1])

    return dict(
        events=events,
        blocked=blocked,
        probabilities=probabilities,
        next_open=next_open,
        horizon=horizon,
        warmup_events=warmup_events,
        initial_memory=initial_memory,
        initial_deadlines=initial_deadlines,
    )


def parameters(model, config):
    rng = np.random.default_rng(config["road_parameter_seed"])
    road_count = len(model["roads"])

    def multipliers(width):
        values = rng.uniform(1 - width, 1 + width, road_count)
        return values / values.mean()

    travel_multipliers = multipliers(config["travel_heterogeneity"])
    rounded_travel = np.rint(config["travel_mean_s"] * travel_multipliers)
    travel = np.maximum(1, rounded_travel).astype(int)

    block_multipliers = multipliers(config["block_heterogeneity"])
    rounded_blocks = np.rint(config["block_mean_s"] * block_multipliers)
    blocks = np.maximum(2, rounded_blocks).astype(int)

    base = config["base_probability"] * multipliers(config["base_heterogeneity"])

    weights = np.exp(-model["hops"] / config["hop_decay_scale"])
    np.fill_diagonal(weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-20)

    return dict(travel=travel, blocks=blocks, base=base, kernel=weights)
