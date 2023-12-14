import pickle
import json
import os
import math
import time
import random
from itertools import product

from networkx.readwrite import json_graph

import torch
import torch_scatter
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from .config import TOPOLOGIES_DIR
from .ADMM import ADMM
from .path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles


class TealEnv(object):

    def __init__(
            self, obj, topo, problems,
            num_path, edge_disjoint, dist_metric, rho,
            train_size, val_size, test_size, num_failure, device,
            raw_action_min=-10.0, raw_action_max=10.0):
        """Initialize Teal environment.

        Args:
            obj: objective
            topo: topology name
            problems: problem list
            num_path: number of paths per demand
            edge_disjoint: whether edge-disjoint paths
            dist_metric: distance metric for shortest paths
            rho: hyperparameter for the augumented Lagranian
            train size: train start index, stop index
            val size: val start index, stop index
            test size: test start index, stop index
            device: device id
            raw_action_min: min value when clamp raw action
            raw_action_max: max value when clamp raw action
        """

        self.obj = obj
        self.topo = topo
        self.problems = problems
        self.num_path = num_path
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric

        self.train_start, self.train_stop = train_size
        self.val_start, self.val_stop = val_size
        self.test_start, self.test_stop = test_size
        self.num_failure = num_failure
        self.device = device

        # init matrices related to topology
        self.G = self._read_graph_json(topo)
        self.capacity = torch.FloatTensor(
            [float(c_e) for u, v, c_e in self.G.edges.data('capacity')])
        self.num_edge_node = len(self.G.edges)
        self.num_path_node = self.num_path * self.G.number_of_nodes()\
            * (self.G.number_of_nodes()-1)
        self.edge_index, self.edge_index_values, self.p2e = \
            self.get_topo_matrix(topo, num_path, edge_disjoint, dist_metric)

        # init ADMM
        self.ADMM = ADMM(
            self.p2e, self.num_path, self.num_path_node,
            self.num_edge_node, rho, self.device)

        # min/max value when clamp raw action
        self.raw_action_min = raw_action_min
        self.raw_action_max = raw_action_max

        self.reset('train')

    def reset(self, mode='test'):
        """Reset the initial conditions in the beginning."""

        if mode == 'train':
            self.idx_start, self.idx_stop = self.train_start, self.train_stop
        elif mode == 'test':
            self.idx_start, self.idx_stop = self.test_start, self.test_stop
        else:
            self.idx_start, self.idx_stop = self.val_start, self.val_stop
        self.idx = self.idx_start
        self.obs = self._read_obs()

    def get_obs(self):
        """Return observation (capacity + traffic matrix)."""

        return self.obs

    def _read_obs(self):
        """Return observation (capacity + traffic matrix) from files."""

        topo, topo_fname, tm_fname = self.problems[self.idx]
        with open(tm_fname, 'rb') as f:
            tm = pickle.load(f)
        # remove demands within nodes
        tm = torch.FloatTensor(
            [[ele]*self.num_path for i, ele in enumerate(tm.flatten())
                if i % len(tm) != i//len(tm)]).flatten()
        obs = torch.concat([self.capacity, tm]).to(self.device)
        # simulate link failures in testing
        if self.num_failure > 0 and self.idx_start == self.test_start:
            idx_failure = torch.tensor(
                random.sample(range(self.num_edge_node),
                self.num_failure)).to(self.device)
            obs[idx_failure] = 0
        return obs

    def _next_obs(self):
        """Return next observation (capacity + traffic matrix)."""

        self.idx += 1
        if self.idx == self.idx_stop:
            self.idx = self.idx_start
        self.obs = self._read_obs()
        return self.obs

    def render(self):
        """Return a dictionary for the details of the current problem"""

        topo, topo_fname, tm_fname = self.problems[self.idx]
        problem_dict = {
            'problem_name': topo,
            'obj': self.obj,
            'tm_fname': tm_fname.split('/')[-1],
            'num_node': self.G.number_of_nodes(),
            'num_edge': self.G.number_of_edges(),
            'num_path': self.num_path,
            'edge_disjoint': self.edge_disjoint,
            'dist_metric': self.dist_metric,
            'traffic_model': tm_fname.split('/')[-2],
            'traffic_seed': int(tm_fname.split('_')[-3]),
            'scale_factor': float(tm_fname.split('_')[-2]),
            'total_demand': self.obs[
                -self.num_path_node::self.num_path].sum().item(),
        }
        return problem_dict

    def step(self, raw_action, num_sample=0, num_admm_step=0):
        """Return the reward of current action.

        Args:
            raw_action: raw action from actor
            num_sample: number of samples for reward during training
            num_admm_step: number of ADMM steps during testing
        """

        info = {}
        if self.idx_start == self.train_start:
            reward = self.take_action(raw_action, num_sample)
        else:
            start_time = time.time()
            action = self.transform_raw_action(raw_action)
            if self.obj == 'total_flow':
                # total flow require no constraint violation
                action = self.ADMM.tune_action(self.obs, action, num_admm_step)
                action = self.round_action(action)
            info['runtime'] = time.time() - start_time
            info['sol_mat'] = self.extract_sol_mat(action)
            reward = self.get_obj(action)

        # next observation
        self._next_obs()
        return reward, info

    def get_obj(self, action):
        """Return objective."""

        if self.obj == 'total_flow':
            return action.sum(axis=-1)
        elif self.obj == 'min_max_link_util':
            return (torch_scatter.scatter(
                action[self.p2e[0]], self.p2e[1]
                )/self.obs[:-self.num_path_node]).max()

    def transform_raw_action(self, raw_action):
        """Return network flow allocation as action.

        Args:
            raw_action: raw action directly from ML output
        """
        # clamp raw action between raw_action_min and raw_action_max
        raw_action = torch.clamp(
            raw_action, min=self.raw_action_min, max=self.raw_action_max)

        # translate ML output to split ratio through softmax
        # 1 in softmax represent unallocated traffic
        raw_action = raw_action.exp()
        raw_action = raw_action/(1+raw_action.sum(axis=-1)[:, None])

        # translate split ratio to flow
        raw_action = raw_action.flatten() * self.obs[-self.num_path_node:]

        return raw_action

    def round_action(
            self, action, round_demand=True, round_capacity=True,
            num_round_iter=2):
        """Return rounded action.
        Action can still violate constraints even after ADMM fine-tuning.
        This function rounds the action through cutting flow.

        Args:
            action: input action
            round_demand: whether to round action for demand constraints
            round_capacity: whether to round action for capacity constraints
            num_round_iter: number of rounds when iteratively cutting flow
        """

        demand = self.obs[-self.num_path_node::self.num_path]
        capacity = self.obs[:-self.num_path_node]

        # reduce action proportionally if action exceed demand
        if round_demand:
            action = action.reshape(-1, self.num_path)
            ratio = action.sum(-1) / demand
            action[ratio > 1, :] /= ratio[ratio > 1, None]
            action = action.flatten()

        # iteratively reduce action proportionally if action exceed capacity
        if round_capacity:
            path_flow = action
            path_flow_allocated_total = torch.zeros(path_flow.shape)\
                .to(self.device)
            for round_iter in range(num_round_iter):
                # flow on each edge
                edge_flow = torch_scatter.scatter(
                    path_flow[self.p2e[0]], self.p2e[1])
                # util of each edge
                util = 1 + (edge_flow/capacity-1).relu()
                # propotionally cut path flow by max util
                util = torch_scatter.scatter(
                    util[self.p2e[1]], self.p2e[0], reduce="max")
                path_flow_allocated = path_flow/util
                # update total allocation, residual capacity, residual flow
                path_flow_allocated_total += path_flow_allocated
                if round_iter != num_round_iter - 1:
                    capacity = (capacity - torch_scatter.scatter(
                        path_flow_allocated[self.p2e[0]], self.p2e[1])).relu()
                    path_flow = path_flow - path_flow_allocated
            action = path_flow_allocated_total

        return action

    def take_action(self, raw_action, num_sample):
        '''Return an approximate reward for action for each node pair.
        To make function fast and scalable on GPU, we only calculate delta.
        We assume when changing action in one node pair:
        (1) The change in edge utilization is very small;
        (2) The bottleneck edge in a path does not change due to (1).
        For evary path after change:
            path_flow/max(util, 1) =>
            (path_flow+delta_path_flow)/max(util+delta_util, 1)
            if util < 1:
                reward = - delta_path_flow
            if util > 1:
                reward = - delta_path_flow/(util+delta_util)
                    + path_flow*delta_util/(util+delta_util)/util
                    approx delta_path_flow/util - path_flow/util^2*delta_util

        Args:
            raw_action: raw action from policy network
            num_sample: number of samples in estimating reward
        '''

        path_flow = self.transform_raw_action(raw_action)
        edge_flow = torch_scatter.scatter(path_flow[self.p2e[0]], self.p2e[1])
        util = edge_flow/self.obs[:-self.num_path_node]

        # sample from uniform distribution [mean_min, min_max]
        distribution = Uniform(
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_min,
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_max)
        reward = torch.zeros(self.num_path_node//self.num_path).to(self.device)

        if self.obj == 'total_flow':

            # find bottlenack edge for each path
            util, path_bottleneck = torch_scatter.scatter_max(
                util[self.p2e[1]], self.p2e[0])
            path_bottleneck = self.p2e[1][path_bottleneck]

            # prepare -path_flow/util^2 for reward
            coef = path_flow/util**2
            coef[util < 1] = 0
            coef = torch_scatter.scatter(
                coef, path_bottleneck).reshape(-1, 1)

            # prepare path_util to bottleneck edge_util
            bottleneck_p2e = torch.sparse_coo_tensor(
                self.p2e, (1/self.obs[:-self.num_path_node])[self.p2e[1]],
                [self.num_path_node, self.num_edge_node])

            # sample raw_actions and change each node pair at a time for reward
            for _ in range(num_sample):
                sample = distribution.rsample()

                # add -delta_path_flow if util < 1 else -delta_path_flow/util
                delta_path_flow = self.transform_raw_action(sample) - path_flow
                reward += -(delta_path_flow/(1+(util-1).relu()))\
                    .reshape(-1, self.num_path).sum(-1)

                # add path_flow/util^2*delta_util for each path
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                # get utilization changes on edge
                # do not use torch_sparse.spspmm()
                # "an illegal memory access was encountered" in large topology
                delta_util = torch.sparse.mm(delta_path_flow, bottleneck_p2e)
                reward += torch.sparse.mm(delta_util, coef).flatten()

        elif self.obj == 'min_max_link_util':

            # find link with max utilization
            max_util_edge = util.argmax()

            # prepare paths related to max_util_edge
            max_util_paths = torch.zeros(self.num_path_node).to(self.device)
            max_util_paths[self.p2e[0, self.p2e[1] == max_util_edge]] =\
                1/self.obs[max_util_edge]

            # sample raw_actions and change each node pair at a time for reward
            for _ in range(num_sample):
                sample = distribution.rsample()

                delta_path_flow = self.transform_raw_action(sample) - path_flow
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                reward += torch.sparse.mm(
                    delta_path_flow, max_util_paths.reshape(-1, 1)).flatten()

        return reward/num_sample

    def _read_graph_json(self, topo):
        """Return network topo from json file."""

        assert topo.endswith(".json")
        with open(os.path.join(TOPOLOGIES_DIR, topo)) as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)

    def path_full_fname(self, topo, num_path, edge_disjoint, dist_metric):
        """Return full name of the topology path."""

        return os.path.join(
            TOPOLOGIES_DIR, "paths", "path-form",
            "{}-{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl".format(
                topo, num_path, edge_disjoint, dist_metric))

    def get_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary."""

        self.path_fname = self.path_full_fname(
            topo, num_path, edge_disjoint, dist_metric)
        print("Loading paths from pickle file", self.path_fname)
        try:
            with open(self.path_fname, 'rb') as f:
                path_dict = pickle.load(f)
                print("path_dict size:", len(path_dict))
                return path_dict
        except FileNotFoundError:
            print("Creating paths {}".format(self.path_fname))
            path_dict = self.compute_path(
                topo, num_path, edge_disjoint, dist_metric)
            print("Saving paths to pickle file")
            with open(self.path_fname, "wb") as w:
                pickle.dump(path_dict, w)
        return path_dict

    def compute_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary through computation."""

        path_dict = {}
        G = graph_copy_with_edge_weights(self.G, dist_metric)
        for s_k in G.nodes:
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(G, s_k, t_k, num_path, edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                path_dict[(s_k, t_k)] = paths_no_cycles
        return path_dict

    def get_regular_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary with the same number of paths per demand.
        Fill with the first path when number of paths is not enough.
        """

        path_dict = self.get_path(topo, num_path, edge_disjoint, dist_metric)
        for (s_k, t_k) in path_dict:
            if len(path_dict[(s_k, t_k)]) < self.num_path:
                path_dict[(s_k, t_k)] = [
                    path_dict[(s_k, t_k)][0] for _
                    in range(self.num_path - len(path_dict[(s_k, t_k)]))]\
                    + path_dict[(s_k, t_k)]
            elif len(path_dict[(s_k, t_k)]) > self.num_path:
                path_dict[(s_k, t_k)] = path_dict[(s_k, t_k)][:self.num_path]
        return path_dict

    def get_topo_matrix(self, topo, num_path, edge_disjoint, dist_metric):
        """
        Return matrices related to topology.
        edge_index, edge_index_values: index and value for matrix
        D^(-0.5)*(adjacent)*D^(-0.5) without self-loop
        p2e: [path_node_idx, edge_nodes_inx]
        """

        # get regular path dict
        path_dict = self.get_regular_path(
            topo, num_path, edge_disjoint, dist_metric)

        # edge nodes' degree, index lookup
        edge2idx_dict = {edge: idx for idx, edge in enumerate(self.G.edges)}
        node2degree_dict = {}
        edge_num = len(self.G.edges)

        # build edge_index
        src, dst, path_i = [], [], 0
        for s in range(len(self.G)):
            for t in range(len(self.G)):
                if s == t:
                    continue
                for path in path_dict[(s, t)]:
                    for (u, v) in zip(path[:-1], path[1:]):
                        src.append(edge_num+path_i)
                        dst.append(edge2idx_dict[(u, v)])

                        if src[-1] not in node2degree_dict:
                            node2degree_dict[src[-1]] = 0
                        node2degree_dict[src[-1]] += 1
                        if dst[-1] not in node2degree_dict:
                            node2degree_dict[dst[-1]] = 0
                        node2degree_dict[dst[-1]] += 1
                    path_i += 1

        # edge_index is D^(-0.5)*(adj)*D^(-0.5) without self-loop
        edge_index_values = torch.tensor(
            [1/math.sqrt(node2degree_dict[u]*node2degree_dict[v])
                for u, v in zip(src+dst, dst+src)]).to(self.device)
        edge_index = torch.tensor(
            [src+dst, dst+src], dtype=torch.long).to(self.device)
        p2e = torch.tensor([src, dst], dtype=torch.long).to(self.device)
        p2e[0] -= len(self.G.edges)

        return edge_index, edge_index_values, p2e

    def extract_sol_mat(self, action):
        """return sparse solution matrix.
        Solution matrix is of dimension num_of_demand x num_of_edge.
        The i, j entry represents the traffic flow from demand i on edge j.
        """

        # 3D sparse matrix to represent which path, which demand, which edge
        sol_mat_index = torch.stack([
            self.p2e[0] % self.num_path,
            torch.div(self.p2e[0], self.num_path, rounding_mode='floor'),
            self.p2e[1]])

        # merge allocation from different paths of the same demand
        sol_mat = torch.sparse_coo_tensor(
            sol_mat_index,
            action[self.p2e[0]],
            (self.num_path,
                self.num_path_node//self.num_path,
                self.num_edge_node))
        sol_mat = torch.sparse.sum(sol_mat, [0])

        return sol_mat
