#! /usr/bin/env python

from teal_helper import get_args_and_problems, print_, PATH_FORM_HYPERPARAMS

import os
import sys

import torch

sys.path.append('..')

from lib.teal_env import TealEnv
from lib.teal_actor import TealActor
from lib.teal_model import Teal


TOP_DIR = "teal-logs"
MODEL_DIR = "teal-models"
HEADERS = [
    "problem",
    "num_nodes",
    "num_edges",
    "traffic_seed",
    "scale_factor",
    "tm_model",
    "total_demand",
    "algo",
    "num_paths",
    "edge_disjoint",
    "dist_metric",
    "objective",
    "obj_val",
    "runtime",
]

OUTPUT_CSV_TEMPLATE = "teal-{}-{}.csv"


def benchmark(problems, output_csv, arg):

    num_path, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
    obj, topo = args.obj, args.topo
    model_save = args.model_save
    device = torch.device(
        f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")

    # ========== load hyperparameters
    # env hyper-parameters
    train_size = [args.slice_train_start, args.slice_train_stop]
    val_size = [args.slice_val_start, args.slice_val_stop]
    test_size = [args.slice_test_start, args.slice_test_stop]
    # actor hyper-parameters
    num_layer = args.layers
    rho = args.rho
    # training hyper-parameters
    lr = args.lr
    early_stop = args.early_stop
    num_epoch = args.epochs
    batch_size = args.bsz
    num_sample = args.samples
    num_admm_step = args.admm_steps
    # testing hyper-parameters
    num_failure = args.failures

    # ========== init teal env, actor, model
    teal_env = TealEnv(
        obj=obj,
        topo=topo,
        problems=problems,
        num_path=num_path,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        rho=rho,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        num_failure=num_failure,
        device=device)
    teal_actor = TealActor(
        teal_env=teal_env,
        num_layer=num_layer,
        model_dir=MODEL_DIR,
        model_save=model_save,
        device=device)
    teal = Teal(
        teal_env=teal_env,
        teal_actor=teal_actor,
        lr=lr,
        early_stop=early_stop)

    # ========== train and test
    teal.train(
        num_epoch=num_epoch,
        batch_size=batch_size,
        num_sample=num_sample)
    teal.test(
        num_admm_step=num_admm_step,
        output_header=HEADERS,
        output_csv=output_csv,
        output_dir=TOP_DIR)

    return


if __name__ == '__main__':

    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    args, output_csv, problems = get_args_and_problems(OUTPUT_CSV_TEMPLATE)

    if args.dry_run:
        print("Problems to run:")
        for problem in problems:
            print(problem)
    else:
        benchmark(problems, output_csv, args)
