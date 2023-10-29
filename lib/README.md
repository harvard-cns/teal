## Code structure
```
./lib
├── init.py                 # package initialization when imported
├── config.py               # directory configurations
├── teal_env.py             # traffic engineering environment
├── teal_model.py           # Teal model for training, validation, and testing
├── FlowGNN.py              # FlowGNN: extracts flow features
├── teal_actor.py           # multi-agent RL: maps flow features to initial allocations
├── ADMM.py                 # ADMM: fine-tunes allocations to satisfy constraints 
├── graph_utils.py          # utility functions for graph processing
├── path_util.py            # utility functions for path processing
└── util.py                 # utility functions for Teal, e.g. printing results
```
