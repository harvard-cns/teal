import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial

from .FlowGNN import FlowGNN
from .utils import weight_initialization, print_


class TealActor(nn.Module):

    def __init__(
            self, teal_env, num_layer, model_dir, model_save, device,
            std=1, log_std_min=-10.0, log_std_max=10.0):
        """Initialize teal actor.

        Args:
            teal_env: teal environment
            num_layer: num of layers in flowGNN
            model_dir: model save directory
            model_save: whether to save the model
            device: device id
            std: std value, -1 if apply neuro networks for std
            log_std_min: lower bound for log std
            log_std_max: upper bound for log std
        """

        super(TealActor, self).__init__()

        # teal environment
        self.env = teal_env
        self.num_path = self.env.num_path
        self.num_path_node = self.env.num_path_node

        # init FlowGNN
        self.device = device
        self.FlowGNN = FlowGNN(self.env, num_layer).to(self.device)

        # init COMA policy
        self.std = std
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.mean_linear = nn.Linear(
            self.num_path*(self.FlowGNN.num_layer+1),
            self.num_path).to(self.device)
        # apply neuro networks for std
        if std < 0:
            self.log_std_linear = nn.Linear(
                self.num_path*(self.FlowGNN.num_layer+1),
                self.num_path).to(self.device)

        # get model fname
        self.model_fname = self.model_full_fname(
            model_dir, self.env.topo, num_layer, std)
        self.model_save = model_save
        # load model
        self.load_model()

    def model_full_fname(self, model_dir, topo, num_layer, std):
        """Return full name of the ML model."""

        return os.path.join(
            model_dir, "{}_flowGNN-{}_std-{}.pt".format(
                topo, num_layer, std < 0))

    def load_model(self):
        """Load from model fname."""

        if os.path.exists(self.model_fname):
            print_(f'Loading Teal model from {self.model_fname}')
            if self.device.type == 'cpu':
                self.load_state_dict(
                    torch.load(
                        self.model_fname, map_location=torch.device('cpu')))
            else:
                self.load_state_dict(torch.load(self.model_fname))
        else:
            print_(f'Creating model {self.model_fname}')
            # weight initialization for neural networks
            self.apply(weight_initialization)

    def save_model(self):
        """Save from model fname."""

        if self.model_save:
            print_(f'Saving Teal model from {self.model_fname}')
            torch.save(self.state_dict(), self.model_fname)

    def forward(self, feature):
        """Return mean of normal distribution after forward propagation

        Args:
            features: input features including capacity and demands
        """
        x = self.FlowGNN(feature)
        x = x.reshape(
            self.num_path_node//self.num_path,
            self.num_path*(self.FlowGNN.num_layer+1))
        mean = self.mean_linear(x)

        # to apply neuro networks for std
        if self.std < 0:
            log_std = self.log_std_linear(x)
            log_std_clamped = torch.clamp(
                log_std,
                min=self.log_std_min,
                max=self.log_std_max)
            nn_std = torch.exp(log_std_clamped)
            return mean, nn_std
        # deterministic std
        else:
            return mean, self.std

    def evaluate(self, obs, deterministic=False):
        """Return raw action before softmax split ratio.

        Args:
            obs: input features including capacity and demands
            deterministic: whether to have deterministic action
        """

        feature = obs.reshape(-1, 1)
        mean, std = self.forward(feature)
        mean = mean

        # test mode
        if deterministic:
            distribution = None
            raw_action = mean.detach()
            log_probability = None
        # train mode
        else:
            # use normal distribution for action
            distribution = Normal(mean, std)
            sample = distribution.rsample()
            raw_action = sample.detach()
            log_probability = distribution.log_prob(raw_action).sum(axis=-1)

        return raw_action, log_probability

    def act(self, obs):
        """Return split ratio as action with disabled gradient calculation.

        Args:
            obs: input observation including capacity and demands
        """

        with torch.no_grad():
            # deterministic action
            raw_action, _ = self.evaluate(obs, deterministic=True)
            return raw_action
