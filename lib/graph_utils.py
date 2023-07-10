from .utils import uni_rand
from itertools import tee
from sys import maxsize
from collections import defaultdict

EPS = 1e-4


def add_bi_edge(G, src, dest, capacity=None):
    G.add_edge(src, dest)
    G.add_edge(dest, src)
    if capacity:
        G[src][dest]["capacity"] = capacity
        G[dest][src]["capacity"] = capacity


def assert_flow_conservation(flow_list, commod_key):
    if len(flow_list) == 0:
        return 0.0

    src = commod_key[-1][0]
    sink = commod_key[-1][1]

    in_flow, out_flow = defaultdict(float), defaultdict(float)

    for (u, v), l in flow_list:
        in_flow[v] += l
        out_flow[u] += l

    in_flow = dict(in_flow)
    out_flow = dict(out_flow)

    if src in in_flow:
        assert False

    if sink in out_flow:
        assert False

    assert abs(out_flow[src] - in_flow[sink]) < EPS
    assert out_flow[src] > -EPS

    for node in in_flow.keys():
        if node == sink or node == src:
            continue
        assert abs(in_flow[node] - out_flow[node]) < EPS

    return out_flow[src]


# subtract flows in sol_dict from edges in G
def compute_residual_problem(problem, sol_dict):
    tm = problem.traffic_matrix.tm
    for (k, (s_k, t_k, d_k)), flow_list in sol_dict.items():
        out_flow = compute_in_or_out_flow(flow_list, 0, {s_k})
        assert out_flow >= -EPS
        if out_flow < 0:
            out_flow = 0
        new_d_k = d_k - out_flow
        # clamp new demand to 0.0 to avoid floating point errors
        if new_d_k < EPS:
            new_d_k = 0.0
        tm[s_k, t_k] = new_d_k

        for (u, v), l in flow_list:
            problem.G[u][v]["capacity"] -= l
            # same here; clamp capacity to 0.0
            if problem.G[u][v]["capacity"] < 0.0:
                problem.G[u][v]["capacity"] = 0.0

    problem._invalidate_commodity_lists()
    return problem


# subtract flows in sol_dict from edges in G
def compute_residual_graph(G, sol_dict):
    for flow_list in sol_dict.values():
        for (u, v), l in flow_list:
            G[u][v]["capacity"] -= l


# If target_node is not present in the flow sequence, return [], []
def get_in_and_out_neighbors(flow_list, target_node):
    in_neighbors, out_neighbors = set(), set()
    for (u, v), l in flow_list:
        if u == target_node:
            out_neighbors.add(v)
        elif v == target_node:
            in_neighbors.add(u)
    return in_neighbors, out_neighbors


# Return a list of nodes that neighbor one of the nodes in the node set, along
# with the flow that neighbor sent
def neighbors_and_flows(flow_list, edge_idx, node_set={}):
    n_and_f = []
    for edge, l in flow_list:
        if edge[edge_idx] in node_set:
            n_and_f.append((edge[1 + edge_idx], l))

    return n_and_f


# Merges multiple flow entries on the same edge; return values has only unique
# edges and total flow
def merge_flows(flow_list):
    result = defaultdict(float)
    for (u, v), l in flow_list:
        result[(u, v)] += l
    return [((u, v), l) for (u, v), l in result.items()]


# Compute in flow (edge_idx => -1) or out flow (edge_idx => 0)
# to/from a given set of nodes (node_set) for a flow list
def compute_in_or_out_flow(flow_list, edge_idx, node_set={}):
    flow = 0.0
    for edge, l in flow_list:
        if edge[edge_idx] in node_set:
            flow += l
        elif edge[1 - edge_idx] in node_set:
            flow -= l

    return flow


# taken from `pairwise` in Itertools Recipes:
# https://docs.python.org/3/library/itertools.html
def path_to_edge_list(path):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(path)
    next(b, None)
    return zip(a, b)


# Assumes that the flow sequence has been sorted. If target_node is not present
# in the flow sequence, return 0.0 flow
# TODO: handle sublists
def flow_through_node(flow_seq, target_node):
    for i in range(len(flow_seq)):
        (u, v), ll = flow_seq[i]
        if v == target_node:
            assert i < len(flow_seq)
            (u_next, v_next), l_next = flow_seq[i + 1]
            assert ll == l_next
            assert v == u_next
            return ll
    return 0.0


# Compute and return total flow from solution. Solution is a
# dict of {commodity (k, (s_k, t_k, d_k)): flow_seq [((u, v), l),...]}
def total_flow(solution):

    flow_sum = 0.0
    for (_, (s_k, _, _)), flow_seq in solution.items():
        for flow_edge in flow_seq:
            if isinstance(flow_edge, tuple):
                if flow_edge[0][0] == s_k:
                    flow_sum += flow_edge[-1]
            elif isinstance(flow_edge, list):
                for e in flow_edge:
                    if e[0][0] == s_k:
                        flow_sum += e[-1]
            else:
                # TODO: throw error
                pass
    return flow_sum


# @param: _flow_seq: [((u, v), l)...] where (u, v) is an edge,
# and l is the flow along that edge
# @param: src: the source of the flow
#
# Returns sorted list [((u, v), l),...] starting from src. If flow diverges at
# any point, we encode it as a sublist (i.e., we would return a list within a
# list)
def sort_flow_seq(_flow_seq, src):

    # Return (flow, sink)
    def sort_flow_seq(to_return, flow_seq, curr_node, curr_flow):

        new_edge_flows = []
        inds_to_delete = []

        for i, edge_flow in enumerate(flow_seq):
            (u, v), ll = edge_flow
            if curr_node == u:
                # Flow should never increase along a path;
                # if it does, then it means we must be forking
                if ll > curr_flow:
                    # If the flow increased, that means that we're joining up
                    # with another edge. We're done traversing down this path
                    break
                new_edge_flows.append(edge_flow)
                inds_to_delete.append(i)

        # Delete in descending order; otherwise, we'll get index-out-of-range
        # errors
        inds_to_delete.reverse()

        if len(new_edge_flows) == 0:
            # Base Case: return current node, which must be the sink
            return curr_node

        if len(new_edge_flows) == 1:
            # The flow is continuing to one other node; append to to_return
            # and continue
            new_edge_flow = new_edge_flows[0]  # (u, v), l
            (_, v), ll = new_edge_flow
            assert ll == curr_flow or curr_flow == float("inf")
            to_return.append(new_edge_flow)
            del flow_seq[inds_to_delete[0]]
            return sort_flow_seq(to_return, flow_seq, v, ll)

        if len(new_edge_flows) > 1:
            # This flow is splitting into multiple sub-flows.
            # Append a list for each sub-flow; each list should
            # end with the same sink, which may or may not be our
            # original sink. The sum of all the sub-flows should
            # equal our current flow.
            for i in inds_to_delete:
                del flow_seq[i]

            flow_sum = 0.0
            new_src = None
            for new_edge_flow in new_edge_flows:
                (_, v), ll = new_edge_flow
                new_sub_flow = [new_edge_flow]
                sink = sort_flow_seq(new_sub_flow, flow_seq, v, ll)
                # All the sub-flows end up at the same sink
                assert new_src == sink or new_src is None
                new_src = sink
                flow_sum += ll
                to_return.append(new_sub_flow)
            # all the sub-flows add up to the total flow
            assert flow_sum == curr_flow or curr_flow == float("inf")
            curr_flow = flow_sum

            return sort_flow_seq(to_return, flow_seq, new_src, curr_flow)

    to_return = []
    sort_flow_seq(to_return, _flow_seq.copy(), src, curr_flow=float("inf"))
    return to_return


# Assumes mat is 2D
def commodity_gen(mat, with_val=True, skip_zero=True):
    for x in range(mat.shape[0]):
        for y in range(mat.shape[-1]):
            # always skip diagonal values
            if x == y:
                continue
            if skip_zero and mat[x, y] == 0:
                continue
            if with_val:
                yield x, y, mat[x, y]
            else:
                yield x, y


def transform_for_network_simplex(problem, vis=False):

    G = problem.G.copy()
    node_index = len(G.nodes)
    for _, (s_k, t_k, d_k) in problem.commodity_list:
        # First add new source and sink nodes to the graph
        new_src, new_sink = node_index, node_index + 1
        if vis:
            src_pos, sink_pos = G.nodes[s_k]["pos"], G.nodes[t_k]["pos"]
            new_src_pos = src_pos[0] + uni_rand(-2, 0), \
                src_pos[-1] + uni_rand(-2, 0)
            new_sink_pos = sink_pos[0] - uni_rand(-2, 0), \
                sink_pos[-1] - uni_rand(-2, 0)

        G.add_node(new_src, demand=-d_k, label="{}: {}".format(new_src, -d_k))
        G.add_node(new_sink, demand=d_k, label="{}: {}".format(new_sink, d_k))
        if vis:
            G[new_src]["pos"] = new_src_pos
            G[new_sink]["pos"] = new_sink_pos

        # Then add edge in both directions connecting new source
        #  to old with infinite capacity
        G.add_edge(new_src, s_k, weight=1, capacity=maxsize)
        G.add_edge(s_k, new_src, weight=1, capacity=maxsize)

        # Same for sink
        G.add_edge(new_sink, t_k, capacity=maxsize)
        G.add_edge(t_k, new_sink, capacity=maxsize)

        node_index += 2

    return G
