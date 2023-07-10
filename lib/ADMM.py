import numpy as np

import torch
import torch_scatter


class ADMM():
    """Fine-tunes the allocations and mitigates constraint violations.
    F1_d = d - sum_{p in e} x_p - s1 for demand d;
    F3_e = c - sum_{pe in e} z_pe - s3 for edge e;
    F4_pe = x_p - z_pe for path p edge e;
    The augumented Lagranian for TE is
    L(x, z, s, lambda)
        = - sum_d sum_p x_p
        + lambda1 * F1 + lambda3 * F3 + lambda4 * F4
        + rho/2 * F1^2 + rho/2 * F3^2 + rho/2 * F4^2.
    """

    def __init__(
            self, p2e, num_path, num_path_node, num_edge_node,
            rho, device):
        """Initialize ADMM with the network topology.

        Args:
            teal_env: teal environment
            rho: hyperparameter for the augumented Lagranian
            device: device for new tensor to be allocated
        """

        self.rho = rho
        self.device = device

        self.p2e = p2e
        self.num_path = num_path
        self.num_path_node = num_path_node
        self.num_edge_node = num_edge_node

        self.A_d_inv = self.init_A()
        self.b_extra = torch.zeros(self.num_edge_node).double().to(self.device)
        self.p2e_1_extra = torch.arange(self.num_edge_node).to(self.device)

    def init_A(self):
        """Return the inverse of A_d.
        A_d = diag(num_edges_for_paths) + 1 for demand d.
        """

        num_edges_for_path = torch_scatter.scatter(
            torch.ones(self.p2e.shape[1]).to(self.device),
            self.p2e[0])
        A_d_inv = torch.stack(
            [torch.linalg.inv(torch.diag(ele) + 1) for ele
                in num_edges_for_path.reshape(-1, self.num_path)])

        return A_d_inv

    def update_obs_action(self, obs, action):
        """Update demands, capacity, allocation."""

        # update demands and capacity
        self.d = obs[-self.num_path_node::self.num_path]
        self.c = obs[:-self.num_path_node]

        # update allocation in path-wise and edge-wise
        self.x = action
        self.z = self.x[self.p2e[0]]

        # init slack variables and lambda
        self.s1, self.s3 = 0, 0
        self.l1, self.l3, self.l4 = 0, 0, 0

    def update_admm(self):
        """Update ADMM for one round."""

        self.update_s()
        self.update_lambda()
        self.update_z()
        self.update_x()

    def update_s(self):
        """Update slack variables s = argmin_s L(x, z, s, lambda).
        s1 = - lambda1 / rho + (d - sum x_p)
        s3 = - lambda3 / rho + (c - sum z_pe)
        """

        self.s1 = self.l1 / self.rho \
            + (self.d - self.x.reshape(-1, self.num_path).sum(1))
        self.s3 = self.l3 / self.rho \
            + (self.c - torch_scatter.scatter(self.z, self.p2e[1]))

        self.s1 = self.s1.relu()
        self.s3 = self.s3.relu()

    def update_x(self):
        """Update x = argmin_x L(x, z, s, lambda).
        x = - A_d_inv * b_d,
        where [b_d]_p = - 1 - lambda1_d + sum_{e in p} lambda4_pe
            + rho * (- d + s1_d) - rho * sum_{e in p} z_pe.
        """

        b = -1 - self.l1[:, None] \
            + self.rho*(-self.d + self.s1)[:, None]\
            + torch_scatter.scatter(
                self.l4-self.rho*self.z,
                self.p2e[0]).reshape(-1, self.num_path)

        self.x = -torch.einsum(
            "nab,nb->na",
            self.A_d_inv/self.rho,
            b).reshape(-1)

        # use x.relu() to approximate the non-negative solution
        self.x = self.x.relu()

    def update_z(self, num_approx=1):
        """Update z = argmin_z L(x, z, s, lambda).
        z = - A_e_inv * b_e,
        where [b_e]_p = - lambda3_e - lambda4_pe
            + rho * (- c_e + s_e - x_p).
        where A_e = I + 1.

        Args:
            num_approx: num of approx rounds for the non-negative solution
        """

        p2e_1 = self.p2e[1].clone()

        # 'double' precision is necessary:
        # torch_scatter is implemented via atomic operations on the GPU and is
        # therefore **non-deterministic** since the order of parallel
        # operations to the same value is undetermined.
        # For floating-point variables, this results in a source of variance in
        # the result.
        b = (
            self.rho*(
                -self.c[self.p2e[1]] + self.s3[self.p2e[1]]
                - self.x[self.p2e[0]])
            - self.l3[self.p2e[1]] - self.l4
        ).double()

        # z = - A_e_inv * b_e = sum b_e / (|b_e| + 1) - b
        # use b_extra and p2e_1_extra for |b_e| + 1
        b_mean = torch_scatter.scatter(
            torch.concat([b, self.b_extra]),
            torch.concat([p2e_1, self.p2e_1_extra]), reduce='mean')
        self.z = (b_mean[self.p2e[1]] - b)/self.rho

        # cannot use x.relu() to approximate the non-negative solution
        # iteratively decide which z is 0 and solve the rest of z
        for _ in range(num_approx):
            p2e_1[self.z < 0] = self.num_edge_node
            b_mean = torch_scatter.scatter(
                torch.concat([b, self.b_extra]),
                torch.concat([p2e_1, self.p2e_1_extra]), reduce='mean')
            self.z = (b_mean[self.p2e[1]] - b)/self.rho
        self.z = self.z.float().relu()

    def update_lambda(self):
        """Update lambda.
        lambda1 = lambda1 + rho * (d - sum_{p in e} x_p - s1);
        lambda3 = lambda3 + rho * (c - sum_{pe in e} z_pe - s3);
        lambda4 = lambda4 + rho * (x_p - z_pe).
        """

        self.l1 = self.l1 + self.rho * (
            self.d - self.x.reshape(-1, self.num_path).sum(1) - self.s1)
        self.l3 = self.l3 + self.rho * (
            self.c - torch_scatter.scatter(self.z, self.p2e[1]) - self.s3)
        self.l4 = self.l4 + self.rho * (
            self.x[self.p2e[0]] - self.z)

    def tune_action(self, obs, action, num_admm_step):
        """Return fine-tuned allocations after ADMM.

        Args:
            obs: observation (capacity + traffic matrix)
            action: action to correct
            num_admm_step: number of admm steps
        """
        # init x, z, s, lambda
        self.update_obs_action(obs, action)

        # admm steps
        for _ in range(num_admm_step):
            self.update_admm()

        return self.x
