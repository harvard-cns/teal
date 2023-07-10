import math

import torch
import torch.nn as nn
import torch_scatter
import torch_sparse

from .utils import weight_initialization


class FlowGNN(nn.Module):
    """Transform the demands into compact feature vectors known as embeddings.

    FlowGNN alternates between
    - GNN layers aimed at capturing capacity constraints;
    - DNN layers aimed at capturing demand constraints.

    Replace torch_sparse package with torch_geometric pakage is possible
    but require larger memory space.
    """

    def __init__(self, teal_env, num_layer):
        """Initialize flowGNN with the network topology.

        Args:
            teal_env: teal environment
            num_layer: num of layers in flowGNN
        """

        super(FlowGNN, self).__init__()

        self.env = teal_env
        self.num_layer = num_layer

        self.edge_index = self.env.edge_index
        self.edge_index_values = self.env.edge_index_values
        self.num_path = self.env.num_path
        self.num_path_node = self.env.num_path_node
        self.num_edge_node = self.env.num_edge_node
        # self.adj_adj = torch.sparse_coo_tensor(self.edge_index,
        #    self.edge_index_values,
        #    [self.num_path_node + self.num_edge_node,
        #    self.num_path_node + self.num_edge_node])

        self.gnn_list = []
        self.dnn_list = []
        for i in range(self.num_layer):
            # to replace with GCNConv package:
            # self.gnn_list.append(GCNConv(i+1, i+1))
            self.gnn_list.append(nn.Linear(i+1, i+1))
            self.dnn_list.append(
                nn.Linear(self.num_path*(i+1), self.num_path*(i+1)))
        self.gnn_list = nn.ModuleList(self.gnn_list)
        self.dnn_list = nn.ModuleList(self.dnn_list)

        # weight initialization for dnn and gnn
        self.apply(weight_initialization)

    def forward(self, h_0):
        """Return embeddings after forward propagation

        Args:
            h_0: inital embeddings
        """

        h_i = h_0
        for i in range(self.num_layer):

            # gnn
            # to replace with GCNConv package:
            # h_i = self.gnn_list[i](h_i, self.edge_index)
            h_i = self.gnn_list[i](h_i)
            # h_i = torch.sparse.mm(self.adj_adj, h_i)
            h_i = torch_sparse.spmm(
                self.edge_index, self.edge_index_values,
                h_0.shape[0], h_0.shape[0], h_i)

            # dnn
            h_i_path_node = self.dnn_list[i](
                h_i[-self.num_path_node:, :].reshape(
                    self.num_path_node//self.num_path,
                    self.num_path*(i+1)))\
                .reshape(self.num_path_node, i+1)
            h_i = torch.concat(
                [h_i[:-self.num_path_node, :], h_i_path_node], axis=0)

            # skip connection
            h_i = torch.cat([h_i, h_0], axis=-1)

        # return path-node embeddings
        return h_i[-self.num_path_node:, :]
