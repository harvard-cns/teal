from collections import defaultdict
from glob import iglob

import argparse
import os
import sys

sys.path.append("..")

from lib.config import TOPOLOGIES_DIR, TM_DIR

PROBLEM_NAMES = [
    'B4.json',
    'UsCarrier.json',
    'Kdl.json',
    'ASN2k.json',
]
TM_MODELS = [
    "real",
]
SCALE_FACTORS = [1.0]
OBJ_STRS = ["total_flow", "min_max_link_util"]

PATH_FORM_HYPERPARAMS = (4, True, "min-hop")

PROBLEM_NAMES_AND_TM_MODELS = [
    (prob_name, tm_model) for prob_name in PROBLEM_NAMES
    for tm_model in TM_MODELS
]

PROBLEMS = []
GROUPED_BY_PROBLEMS = defaultdict(list)
HOLDOUT_PROBLEMS = []
GROUPED_BY_HOLDOUT_PROBLEMS = defaultdict(list)

for problem_name in PROBLEM_NAMES:
    if problem_name.endswith(".graphml"):
        topo_fname = os.path.join(TOPOLOGIES_DIR, "topology-zoo", problem_name)
    else:
        topo_fname = os.path.join(TOPOLOGIES_DIR, problem_name)
    for model in TM_MODELS:
        for tm_fname in iglob(
            "{}/{}/{}*_traffic-matrix.pkl".format(TM_DIR, model, problem_name)
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]),\
                float(vals[3])
            GROUPED_BY_PROBLEMS[(problem_name, model, scale_factor)].append(
                (topo_fname, tm_fname)
            )
            PROBLEMS.append((problem_name, topo_fname, tm_fname))
        for tm_fname in iglob(
            "{}/holdout/{}/{}*_traffic-matrix.pkl".format(
                TM_DIR, model, problem_name
            )
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]),\
                float(vals[3])
            GROUPED_BY_HOLDOUT_PROBLEMS[(problem_name, model, scale_factor)]\
                .append(
                    (topo_fname, tm_fname)
            )
            HOLDOUT_PROBLEMS.append((problem_name, topo_fname, tm_fname))

GROUPED_BY_PROBLEMS = dict(GROUPED_BY_PROBLEMS)
for key, vals in GROUPED_BY_PROBLEMS.items():
    GROUPED_BY_PROBLEMS[key] = sorted(
        vals, key=lambda x: int(x[-1].split('_')[-3]))

GROUPED_BY_HOLDOUT_PROBLEMS = dict(GROUPED_BY_HOLDOUT_PROBLEMS)
for key, vals in GROUPED_BY_HOLDOUT_PROBLEMS.items():
    GROUPED_BY_HOLDOUT_PROBLEMS[key] = sorted(
        vals, key=lambda x: int(x[-1].split('_')[-3]))


def get_problems(args):
    if (args.topo, args.tm_model, args.scale_factor) not in GROUPED_BY_PROBLEMS:
        raise Exception('Traffic matrices not found')
    problems = []
    for topo_fname, tm_fname in GROUPED_BY_PROBLEMS[
            (args.topo, args.tm_model, args.scale_factor)]:
        problems.append((args.topo, topo_fname, tm_fname))
    return problems


def get_args_and_problems(formatted_fname_template, additional_args=[]):
    parser = argparse.ArgumentParser()

    # Problems arguments
    parser.add_argument(
        "--dry-run", dest="dry_run", default=False, action="store_true")
    parser.add_argument(
        "--obj", type=str, default='total_flow', choices=OBJ_STRS)
    parser.add_argument(
        "--tm-model", type=str, default='real', choices=TM_MODELS)
    parser.add_argument(
        "--topo", type=str, required=True, choices=PROBLEM_NAMES)
    parser.add_argument(
        "--scale-factor", type=float, default=1.0, choices=SCALE_FACTORS)
    parser.add_argument(
        '--devid', type=int, default=0, help='device id')
    parser.add_argument(
        '--model-save', type=bool, default=False, help='whether to save model')

    # env hyper-parameters
    parser.add_argument(
        '--slice-train-start', type=int, default=0)
    parser.add_argument(
        '--slice-train-stop', type=int, default=20)
    parser.add_argument(
        '--slice-val-start', type=int, default=20)
    parser.add_argument(
        '--slice-val-stop', type=int, default=28)
    parser.add_argument(
        '--slice-test-start', type=int, default=28)
    parser.add_argument(
        '--slice-test-stop', type=int, default=36)

    # actor hyper-parameters
    parser.add_argument(
        '--layers', type=int, default=6, help='number of flowGNN layers')
    parser.add_argument(
        '--rho', type=float, default=1.0, help='rho in ADMM')

    # training hyper-parameters
    parser.add_argument(
        '--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument(
        '--epochs', type=int, default=0, help='number of training epochs')
    parser.add_argument(
        '--bsz', type=int, default=20, help='batch size')
    parser.add_argument(
        '--samples', type=int, default=5, help='number of COMA samples')
    parser.add_argument(
        '--admm-steps', type=int, default=5, help='number of ADMM steps')
    parser.add_argument(
        '--early-stop', type=bool, default=False, help='whether to stop early')

    for add_arg in additional_args:
        name_or_flags, kwargs = add_arg[0], add_arg[1]
        parser.add_argument(name_or_flags, **kwargs)
    args = parser.parse_args()

    slice_str = "all"  # "slice_" + "_".join(str(i) for i in args.slices)
    formatted_fname_substr = formatted_fname_template.format(
        args.obj, slice_str)
    return args, formatted_fname_substr, get_problems(args)


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
