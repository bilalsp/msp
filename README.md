<!-- Add banner here -->
<img src="img/banner_dyer.png" width="70%" height="40%">

# Machine Scheduling Problem Using Deep Reinforcement Learning
<!-- Add buttons here -->
![GitHub last commit](https://img.shields.io/github/last-commit/bilalsp/msp)
![GitHub pull requests](https://img.shields.io/github/issues-pr/bilalsp/msp)
![GitHub issues](https://img.shields.io/github/issues-raw/bilalsp/msp)
![GitHub stars](https://img.shields.io/github/stars/bilalsp/msp)

<!-- Describe your project in brief -->
In this research project, we propose a sophisticated approach for 
`Machine Scheduling Problem (MSP)` using deep reinforcement learning, where
we address the MSP problem especially for the Fabric Industry. However,
the presented method is not only limited to it.

The complete documentation is available on <a href='https://msp.readthedocs.io/en/latest/'>Read the Docs</a>.

**NOTE: This research has been conducted at <a href='https://limos.fr'>LIMOS</a>
in Clermont-Ferrand, France as a part of research internship.**

# Table of contents
- [Description](#Description)
- [Usage](#Usage)
- [Infrastructure](#Infrastructure)
- [Conclusion](#Conclusion)

## Description
In this research, we design Deep Learning model to sovle MSP problem 
(NP-hard combinatorial optimization problem) where the goal is to find a schedule 
from combinatorial search space such that it minimizes makespan (i.e., an objective) under certain constraints.

To define the MSP problem, we employ graph structure to model the 
various interaction between machines and jobs. We cast
the MSP as a Graph-to-Sequence machine learning problem that follows the standard
encoder-decoder architecture consisting of graph encoder, and a
recurrent decoder. We train the model using a policy-gradient reinforcement learning
algorithm, namely REINFORCE algorithm. Furthermore, we also present different novel
masking strategies to filter out the invalid action at any timestep during the decoding.

**To access the full research document, send the Request Email to mohammedbilalansari.official@gmail.com**

## Usage
The `msp` package is based on low-level API of tensorflow. To install it and all its dependencies, you need to download a code repository and run a `make build` command. Make sure you have `make` utility install on your system.
```
make build
```

To run the program on your system. Pass the following command with appropriate configuration file and 
action. Configuration files are available inside the bin folder. Possible values of actions are 
`run_solver`, `train_model`, `test_model`, or `train_and_test_model`. Refer flag.py file inside bin folder
for more details about flags.
```
python bin/main.py --config_path=bin/configs/exact_solver_msp_5_2_copy.yml --action='train_model'
```

To run it on HPC (High Performance Computing) cluster managed by slurm.
```
export config_path=bin/configs/rl_model_msp_5_2.yml
export action=train_model
sbatch slurm.sh export=config_path,action
```

Another way to use the package is programmatically

Sample code to run the ExactSolver on synthetic data.

```python
# Run the exact solver
from msp.solvers import ExactSolver
from msp.datasets import make_sparse_data

n_instances = 640
dataset = make_sparse_data(n_instances, seed=2021) 
solver = ExactSolver()

for msp_batch_instance in dataset.batch(32):
    best_schedule, makespan = solver(msp_batch_instance)
    # store all best_schedule in an appropriate way
```

Sample code to create the instanace of MSP model.

```python
# create a model instanace
from msp.models import MSPModel
from msp.models.encoders import GGCNEncoder
from msp.models.decoders import AttentionDecoder

encoder_params = {'units': 32}
decoder_params = {'units': 32}
model = MSPModel(GGCNEncoder, encoder_params, AttentionDecoder, decoder_params)
```

## Infrastructure
We run all experiments on NVIDIA Tesla V100 32GB GPU of <a href='https://hpc.hse.ru/en/'> HPC (high-performance computing) </a> cluster of the HSE University.


## Conclusion
In this research, we proposed a sophisticated approach for Machine Scheduling Problem(MSP) using deep reinforcement learning, where we formulated MSP as a Graph-to-Sequence machine learning problem. We also presented novel masking strategies to filter out the invalid action at any timestep. Our approach outperforms both exhaustive search and random searchmethods in terms of 
associated inference time with a narrow objective gap.
