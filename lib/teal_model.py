import pickle
import time
import json
import sys
import os
from tqdm import tqdm
from networkx.readwrite import json_graph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .teal_actor import TealActor
from .teal_env import TealEnv
from .utils import print_


class Teal():
    def __init__(self, teal_env, teal_actor, lr, early_stop):
        """Initialize Teal model.

        Args:
            teal_env: teal environment
            num_layer: number of flowGNN layers
            lr: learning rate
            early_stop: whether to early stop
        """

        self.env = teal_env
        self.actor = teal_actor

        # init optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # early stop when val result no longer changes
        self.early_stop = early_stop
        if self.early_stop:
            self.val_reward = []

    def train(self, num_epoch, batch_size, num_sample):
        """Train Teal model.

        Args:
            num_epoch: number of training epoch
            batch_size: batch size
            num_sample: number of samples in COMA reward
        """

        for epoch in range(num_epoch):

            self.env.reset('train')

            ids = range(self.env.idx_start, self.env.idx_stop)
            loop_obj = tqdm(
                [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)],
                desc=f"Training epoch {epoch}/{num_epoch}: ")

            for idx in loop_obj:
                loss = 0
                for _ in idx:
                    torch.cuda.empty_cache()

                    # get observation
                    obs = self.env.get_obs()
                    # get action
                    raw_action, log_probability = self.actor.evaluate(obs)
                    # get reward
                    reward, info = self.env.step(
                        raw_action, num_sample=num_sample)
                    loss += -(log_probability*reward).mean()

                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                # break

            # early stop
            if self.early_stop:
                self.val()
                if len(self.val_reward) > 20 and abs(
                        sum(self.val_reward[-20:-10])/10
                        - sum(self.val_reward[-10:])/10) < 0.0001:
                    break
        self.actor.save_model()

    def val(self):
        """Validating Teal model."""

        self.actor.eval()
        self.env.reset('val')

        rewards = 0
        for idx in range(self.env.idx_start, self.env.idx_stop):

            # get observation
            problem_dict = self.env.render()
            obs = self.env.get_obs()
            # get action
            raw_action = self.actor.act(obs)
            # get reward
            reward, info = self.env.step(raw_action)
            # show satisfied demand instead of total flow
            rewards += reward.item()/problem_dict['total_demand']\
                if self.env.obj == 'total_flow' else reward.item()
        self.val_reward.append(
            rewards/(self.env.idx_stop - self.env.idx_start))

    def test(self, num_admm_step, output_header, output_csv, output_dir):
        """Test Teal model.

        Args:
            num_admm_step: number of ADMM steps
            output_header: header of the output csv
            output_csv: name of the output csv
            output_dir: directory to save output solution
        """

        self.actor.eval()
        self.env.reset('test')

        with open(output_csv, "a") as results:
            print_(",".join(output_header), file=results)

            runtime_list, obj_list = [], []
            loop_obj = tqdm(
                range(self.env.idx_start, self.env.idx_stop),
                desc="Testing: ")

            for idx in loop_obj:

                # get observation
                problem_dict = self.env.render()
                obs = self.env.get_obs()
                # get action
                start_time = time.time()
                raw_action = self.actor.act(obs)
                runtime = time.time() - start_time
                # get reward
                reward, info = self.env.step(
                    raw_action, num_admm_step=num_admm_step)
                # add runtime in transforming, ADMM, rounding
                runtime += info['runtime']
                runtime_list.append(runtime)
                # show satisfied demand instead of total flow
                obj_list.append(
                    reward.item()/problem_dict['total_demand']
                    if self.env.obj == 'total_flow' else reward.item())

                # display avg runtime, obj
                loop_obj.set_postfix({
                    'runtime': '%.4f' % (sum(runtime_list)/len(runtime_list)),
                    'obj': '%.4f' % (sum(obj_list)/len(obj_list)),
                    })

                # save solution matrix
                sol_mat = info['sol_mat']
                torch.save(sol_mat, os.path.join(
                    output_dir,
                    "{}-{}-{}-teal_objective-{}_{}-paths_"
                    "edge-disjoint-{}_dist-metric-{}_sol-mat.pt".format(
                        problem_dict['problem_name'],
                        problem_dict['traffic_model'],
                        problem_dict['traffic_seed'],
                        problem_dict['obj'],
                        problem_dict['num_path'],
                        problem_dict['edge_disjoint'],
                        problem_dict['dist_metric'])))

                PLACEHOLDER = ",".join("{}" for _ in output_header)
                result_line = PLACEHOLDER.format(
                    problem_dict['problem_name'],
                    problem_dict['num_node'],
                    problem_dict['num_edge'],
                    problem_dict['traffic_seed'],
                    problem_dict['scale_factor'],
                    problem_dict['traffic_model'],
                    problem_dict['total_demand'],
                    "Teal",
                    problem_dict['num_path'],
                    problem_dict['edge_disjoint'],
                    problem_dict['dist_metric'],
                    problem_dict['obj'],
                    reward,
                    runtime)
                print_(result_line, file=results)
                # break
