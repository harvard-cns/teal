# Teal: Traffic Engineering Accelerated by Learning

[Teal](https://dl.acm.org/doi/10.1145/3603269.3604857) is a learning-accelerated traffic engineering (TE) algorithm for cloud wide-area networks (WANs), published at ACM SIGCOMM '23.
By harnessing the parallel processing power of GPUs, Teal achieves unprecedented
acceleration of TE control, surpassing production TE solvers by several orders of magnitude
while retaining near-optimal flow allocations.

## Getting started

### Hardware requirements

- Linux OS (tested on Ubuntu 20.04, 22.04, and CentOS 7)
- A CPU instance with 16+ cores
- (Optional\*) A GPU instance with 24+ GB memory and CUDA installed

\*The baseline TE schemes only require a CPU to run. Teal runs on CPU as well, but its runtime will be significantly longer than on GPU.

### Cloning Teal with submodules
- `git clone https://github.com/harvard-cns/teal.git`
- `cd teal` and update git submodules with `git submodule update --init --recursive`

### Dependencies
- Run `conda env create -f environment.yml` to create a Conda environment with essential Python dependencies
    - [Miniconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) is required
- Run `conda activate teal` to activate the Conda environment. All the following steps related to Python (e.g., `pip` and `python` commands) **must be performed within this Conda environment** to ensure correct Python dependencies.
- Run `pip install -r requirements.txt` to install additional Python dependencies

#### Dependencies only required for baselines
- Install `make`
    - e.g., `sudo apt install build-essential` on Ubuntu
- Acquire a Gurobi license from [Gurobi](https://www.gurobi.com/solutions/licensing/) and activate it with `grbgetkey [gurobi-license]`
    - Run `gurobi_cl` to verify the activation

#### Dependencies only required for Teal
- If on a GPU instance, run `nvcc --version` to identify the installed version of CUDA
    - Note: when following the next steps to install `torch`, `torch-scatter`, and `torch-sparse`, it might be fine to select a version that supports a different CUDA version than the output of `nvcc`, provided that this CUDA version is supported by the GPU driver (as shown in `nvidia-smi`).
- Follow the [official instructions](https://pytorch.org/get-started/previous-versions/) to install PyTorch via pip based on the execution environment (CPU, or GPU with a specific version of CUDA).
    - *Example:* Install PyTorch 1.10.1 for CUDA 11.1 on a **GPU** instance:
        ```
        pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
        ```
        Run `python -c "import torch; print(torch.cuda.is_available())"` to verify the installation.
    - *Example:* Install PyTorch 1.10.1 on a **CPU** instance:
        ```
        pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
        ```
        Run `python -c "import torch; print(torch.__version__)"` to verify the installation.
- Install PyTorch extension libraries `torch-scatter` and `torch-sparse`:
    - First, identify the appropriate archive URL [here](https://data.pyg.org/whl/) based on PyTorch and CUDA versions. E.g., copy the link of `torch-1.10.1+cu111` for PyTorch 1.10.1 and CUDA 11.1.
    - Run `pip install --no-index torch-scatter torch-sparse -f [archive URL]`, replacing `[archive URL]` with the copied archive URL.
    - *Example:* On a **GPU** instance with PyTorch 1.10.1 and CUDA 11.1:
        ```
        pip install --no-index torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.1%2Bcu111.html`
        ```
    - *Example:* On a **CPU** instance with PyTorch 1.10.1:
        ```
       pip install --no-index torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.1%2Bcpu.html
        ```
    - Run `python -c "import torch_scatter; print(torch_scatter.__version__)"` and `python -c "import torch_sparse; print(torch_sparse.__version__)"` to verify the installation.
    - Troubleshooting: refer to the [Installation from Source section](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-source).

## Code structure
```
.
├── lib                     # source code for Teal (details in lib/README.md)
├── pop-ncflow-lptop        # submodule for baselines
│   ├── benchmarks          # test code for baselines
│   ├── ext                 # external code for baselines
│   └── lib                 # source code for baselines
├── run                     # test code for Teal
├── topologies              # network topologies with link capacity (e.g. `B4.json`)
│   └── paths               # paths in topologies (auto-generated if not existent)
└── traffic-matrices        # TE traffic matrices
    ├── real                # real traffic matrices from abilene.txt in Yates (https://github.com/cornell-netlab/yates)
    │                       # (e.g. `B4.json_real_0_1.0_traffic-matrix.pkl`)
    └── toy                 # toy traffic matrices (e.g. `ASN2k.json_toy_0_1.0_traffic-matrix.pkl`)
```

**Note:** As we are not allowed to share the proprietary traffic data from Microsoft WAN (or the Teal model trained on that data), we mapped the publicly accessible Yates traffic data to the B4 topology to facilitate code testing. For the other topologies (UsCarrier, Kdl, and ASN), we synthetically generated "toy" traffic matrices due to their larger sizes.

## Evaluating Teal
To evaluate Teal on the B4 topology:
```
$ cd ./run
$ python teal.py --obj total_flow --topo B4.json --epochs 3 --admm-steps 2
Loading paths from pickle file ~/teal/topologies/paths/path-form/B4.json-4-paths_edge-disjoint-True_dist-metric-min-hop-dict.pkl
path_dict size: 132
Creating model teal-models/B4.json_flowGNN-6_std-False.pt
Training epoch 0/3: 100%|█████████████████████████████████| 1/1 [00:01<00:00,  1.63s/it]
Training epoch 1/3: 100%|█████████████████████████████████| 1/1 [00:00<00:00,  2.45it/s]
Training epoch 2/3: 100%|█████████████████████████████████| 1/1 [00:00<00:00,  2.61it/s]
Testing: 100%|████████████████| 8/8 [00:00<00:00, 38.06it/s, runtime=0.0133, obj=0.9537]
```

To show explanations on the input parameters:
```
$ python teal.py --help
```

Results will be saved in
- `teal-total_flow-all.csv`: performance numbers
- `teal-logs`: directory with TE solution matrices
- `teal-models`: directory to save the trained models when `--model-save True`

Realistic traffic matrices are only available for B4 (please refer to the [note](#code-structure) above). For the other topologies — UsCarrier (`UsCarrier.json`), Kdl (`Kdl.json`), or ASN (`ASN2k.json`), use the "toy" traffic matrices we generated (taking UsCarrier as an example): 
```
$ python teal.py --obj total_flow --topo UsCarrier.json --tm-model toy --epochs 3 --admm-steps 2
```

## Evaluating baselines
Teal is compared with the following baselines:
- LP-all (`path_form.py`): LP-all solves the TE optimization problem for *all* demands using linear programming (implemented in Gurobi)
- LP-top (`top_form.py`): LP-top allocates the *top* α% (α=10 by default) of demands using an LP solver and assigns the remaining demands to the shortest paths
- NCFlow (`ncflow.py`): the NCFlow algorithm from the NSDI '21 paper: [*Contracting Wide-area Network Topologies to Solve Flow Problems Quickly*](https://www.usenix.org/conference/nsdi21/presentation/abuzaid)
- POP (`pop.py`): the POP algorithm from the SOSP '21 paper: [*Solving Large-Scale Granular Resource Allocation Problems Efficiently with POP*](https://dl.acm.org/doi/10.1145/3477132.3483588)

To evaluate the baselines on B4, run the following commands from the project root:
```
$ cd ./pop-ncflow-lptop/benchmarks
$ python path_form.py --obj total_flow --topos B4.json
$ python top_form.py --obj total_flow --topos B4.json
$ python ncflow.py --obj total_flow --topos B4.json
$ python pop.py --obj total_flow --topos B4.json --algo-cls PathFormulation --split-fractions 0.25 --num-subproblems 4
```
Results will be saved in
- `path-form-total_flow-all.csv`, `top-form-total_flow-all.csv`, `ncflow-total_flow-all.csv`, `pop-total_flow-all.csv`: performance numbers
- `path-form-logs`, `top-form-logs`, `ncflow-logs`, `pop-logs`: directory with TE solution matrices

To test on UsCarrier (`UsCarrier.json`), Kdl (`Kdl.json`), or ASN (`ASN2k.json`), specify the "toy" traffic matrices we generated (taking UsCarrier as an example): 
```
$ python path_form.py --obj total_flow --tm-models toy --topos UsCarrier.json
$ python top_form.py --obj total_flow --tm-models toy --topos UsCarrier.json
$ python ncflow.py --obj total_flow --tm-models toy --topos UsCarrier.json
$ python pop.py --obj total_flow --tm-models toy --topos UsCarrier.json --algo-cls PathFormulation --split-fractions 0.25 --num-subproblems 4
```

## Extending Teal

To add another TE implementation to this repo,

- If the implementation is based on linear programming or Gurobi, add test code to `./pop-ncflow-lptop/benchmarks/` and source code to `./pop-ncflow-lptop/lib/algorithms`. Code in `./pop-ncflow-lptop/lib` (e.g., `lp_solver.py`, `traffic_matrix.py`) and `./pop-ncflow-lptop/benchmarks` (e.g., `benchmark_helpers.py`) is reusable.
- If the implementation is based on machine learning, add test code to `./run/` and source code to `./lib/`. Code in `./lib/` (e.g., `teal_env.py`, `utils.py`) and `./run/` (e.g., `teal_helpers.py`) is reusable.


## Citation
If you use our code in your research, please cite our paper:
```
@inproceedings{teal,
    title={Teal: Learning-Accelerated Optimization of WAN Traffic Engineering},
    author={Xu, Zhiying and Yan, Francis Y. and Singh, Rachee and Chiu, Justin T. and Rush, Alexander M. and Yu, Minlan},
    booktitle={Proceedings of the ACM SIGCOMM 2023 Conference},
    pages={378--393},
    month=sep,
    year={2023}
}
```
