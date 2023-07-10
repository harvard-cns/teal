from .graph_utils import path_to_edge_list
from itertools import islice
import networkx as nx
from sys import maxsize


# Remove cycles from path
def remove_cycles(path):
    stack = []
    visited = set()
    for node in path:
        if node in visited:
            # remove elements from this cycle
            while stack[-1] != node:
                visited.remove(stack[-1])
                stack = stack[:-1]
        else:
            stack.append(node)
            visited.add(node)
    return stack


def graph_copy_with_edge_weights(_G, dist_metric):
    G = _G.copy()

    if dist_metric == "inv-cap":
        for u, v, cap in G.edges.data("capacity"):
            if cap < 0.0:
                cap = 0.0
            try:
                G[u][v]["weight"] = 1.0 / cap
            except ZeroDivisionError:
                G[u][v]["weight"] = maxsize
    elif dist_metric == "min-hop":
        for u, v, cap in G.edges.data("capacity"):
            if cap <= 0.0:
                G[u][v]["weight"] = maxsize
            else:
                G[u][v]["weight"] = 1.0
    else:
        raise Exception("invalid dist_metric: {}".format(dist_metric))

    return G


def find_paths(G, s_k, t_k, num_paths, disjoint=True, include_weight=False):
    def compute_weight(G, path):
        return sum(G[u][v]["weight"] for u, v in path_to_edge_list(path))

    def k_shortest_paths(G, source, target, k, weight="weight"):
        try:
            # Yen's shortest path algorithm
            return list(
                islice(nx.shortest_simple_paths(
                    G, source, target, weight=weight), k)
            )
        except nx.NetworkXNoPath:
            return []

    def k_shortest_edge_disjoint_paths(G, source, target, k, weight="weight"):
        def compute_distance(path):
            return sum(G[u][v][weight] for u, v in path_to_edge_list(path))

        return [
            remove_cycles(path)
            for path in sorted(
                nx.edge_disjoint_paths(G, s_k, t_k),
                key=lambda path: compute_distance(path),
            )[:k]
        ]

    if disjoint:
        if include_weight:
            return [
                (path, compute_weight(path))
                for path in k_shortest_edge_disjoint_paths(
                    G, s_k, t_k, num_paths, weight="weight"
                )
            ]
        else:
            return k_shortest_edge_disjoint_paths(
                G, s_k, t_k, num_paths, weight="weight"
            )
    else:
        if include_weight:
            return [
                (path, compute_weight(path))
                for path in k_shortest_paths(
                    G, s_k, t_k, num_paths, weight="weight")
            ]
        else:
            return k_shortest_paths(G, s_k, t_k, num_paths, weight="weight")
