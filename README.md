# Teal

Teal (Traffic Engineering Accelerated by Learning) is a learning-based WAN traffic engineering (TE) algorithm that harnesses the parallel processing power of GPUs to accelerate TE control. Teal generates near-optimal flow allocations while being several orders of magnitude faster than production TE solvers.

## Getting started

### Hardware requirements

- Linux OS (tested on Ubuntu 20.04, 22.04, and CentOS 7)
- a CPU instance with 16+ cores
- (Optional\*) a GPU instance with 24+ GB memory and CUDA installed

\*Baselines only need to be evaluated on a CPU instance. Teal can be evaluated on a CPU instance as well, except that its runtime is significantly longer than the runtime on a GPU instance.


### Cloning Teal with submodule
- Run `git clone https://github.com/harvard-cns/teal.git`
- `cd teal` and then run `git submodule update --init --recursive`

### Dependencies
- Run `conda env create -f environment.yml` for a list of Python library dependencies and run `conda activate teal`
    - [Miniconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) is required
- Run `pip install -r requirements.txt` for additional non-conda dependencies
#### Baselines:
- `make` is required 
- Get a Gurobi license from [Gurobi](https://www.gurobi.com/solutions/licensing/) and activate with `grbgetkey [gurobi-license]`
    - Run `gurobi_cl` to check whether the activation is successful
    - Gurobi is only required for baselines but not by Teal
#### Teal
- If on a GPU instance, run `nvcc --version` for CUDA version
    - Installing a version of the below torch, torch-scatter, torch-sparse that supports a different CUDA version than what is shown by `nvcc` might still work, if this CUDA version is supported by your driver (as indicated by the highest supported CUDA version in `nvidia-smi`)
- Install torch with [pip installation command](https://pytorch.org/get-started/previous-versions/) based on CPU or GPU's CUDA version 
    - Examples:
        - e.g. install torch 1.10.1 for CUDA 11.1 with `pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`
        - e.g. install torch 1.10.1 for CPU with `pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html`
    - For a GPU instance, run `python -c "import torch; print(torch.cuda.is_available())"` to check if torch is installed with CUDA support; for a CPU instance, run `python -c "import torch; print(torch.__version__)"` to check if torch is successfully installed
- Select a [version link](https://data.pyg.org/whl/) based on torch version and CPU or GPU's CUDA version, and install torch-scatter, torch-sparse with `pip install --no-index torch-scatter torch-sparse -f [version-link]`
    - Examples:
        - e.g. install for torch 1.10.1 and CUDA 11.1 with `pip install --no-index torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.1%2Bcu111.html`
        - e.g. install for torch 1.10.1 and CPU with `pip install --no-index torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.1%2Bcpu.html`
    - Run `python -c "import torch_scatter; print(torch_scatter.__version__)"` and `python -c "import torch_sparse; print(torch_sparse.__version__)"` to check if torch-scatter and torch-sparse are successfully installed
    - Troubleshooting: refer to [Installation from Source](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-source) for troubles with installation.

### Code structures
```
.
├── lib                     # source code for Teal
├── pop-ncflow-lptop        # submodule for baselines
│   ├── benchmarks          # test code for baselines            
│   ├── ext                 # external code for baselines                      
│   └── lib                 # source code for baselines       
├── run                     # test code for Teal
├── topologies              # network topologies with link capacity (e.g. `B4.json`)
│   └── paths               # paths in Teal and baselines (paths will be generated automatically if not exists)
└── traffic-matrices        # traffic matrices
    └── real                # real traffic matrices from abilene.txt in Yates (https://github.com/cornell-netlab/yates)
                            # (e.g. `B4.json_real_0_1.0_traffic-matrix.pkl`)
```
## Using Teal

### How to run Teal

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
Results will be saved in
- `teal-logs`: directory for solution matrices
- `teal-models`: directory for the trained model when `--model-save True` 
- `teal-total_flow-all.csv`: csv file for the records of TE problems

### How to run baselines
- LP-all (`path_form`): LP-all solves the TE optimization problem for all demands using linear programming
- LP-top (`top_form.py`): LP-top allocates the top α% of demands using an LP solver and assigns the remaining demands to the shortest paths
- NCFlow (`ncflow.py`): NCFlow solution in paper [*Contracting Wide-area Network Topologies to Solve Flow Problems Quickly*](https://www.usenix.org/conference/nsdi21/presentation/abuzaid)
- POP (`pop.py`): POP solution in paper [*Solving Large-Scale Granular Resource Allocation Problems Efficiently with POP*](https://dl.acm.org/doi/10.1145/3477132.3483588)

To run baselines,
```
$ cd ./pop-ncflow-lptop/benchmarks
$ python path_form.py --obj total_flow --topos B4.json
$ python top_form.py --obj total_flow --topos B4.json
$ python ncflow.py --obj total_flow --topos B4.json
$ python pop.py --obj total_flow --topos B4.json --algo-cls PathFormulation --split-fractions 0.25 --num-subproblems 4 
```
Results will be saved in
- `path-form-logs`, `top-form-logs`, `ncflow-logs`, `pop-logs`: directory for solution directioraries, solution matrices
- `path-form-total_flow-all.csv`, `top-form-total_flow-all.csv`, `ncflow-total_flow-all.csv`, `pop-total_flow-all.csv`: csv files for the records of TE problems in each method respectively

## Extending Teal

To add another TE implementation in this repo, 

- If the implementation is based on Gurobi linear programming, add test code to `./pop-ncflow-lptop/benchmarks/` and source code to `./pop-ncflow-lptop/lib/algorithms`. Other code (e.g. `lp_solver.py`, `traffic_matrix.py`, etc in `./pop-ncflow-lptop/lib` and  `benchmark_helpers.py` in `./pop-ncflow-lptop/benchmarks/`) are reusable.
- If the implementation is based on learning, add test code to `./run/` and source code to `./lib/`. Other code (e.g. `teal_env.py`, `utils.py`, etc in `./lib/` and  `teal_helpers.py` in `./run/`) are reusable.
