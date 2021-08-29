.. _settingup:

Setting up MSP
==============
The `msp` package is based on low-level API of tensorflow. To install it and all its 
dependencies, you need to `download <https://github.com/bilalsp/msp>`_ a github code 
repository and run a `make build` command. 
Make sure you have `make` utility install on your system.

.. code-block:: bash
    
    make build

There are two ways to use this package:
   1. Command Line
   2. Programmatically

1. Command Line
----------------
To run the program on your system. Pass the following command with appropriate configuration file and 
action. Configuration files are available inside the bin folder. Possible values of actions are 
`run_solver`, `train_model`, `test_model`, or `train_and_test_model`. Refer flag.py file inside bin folder
for more details about flags.

.. code-block:: bash

    python bin/main.py --config_path=bin/configs/exact_solver_msp_5_2_copy.yml --action='train_model'

To run it on HPC (High Performance Computing) cluster managed by slurm.

.. code-block:: bash

    export config_path=bin/configs/rl_model_msp_5_2.yml
    export action=train_model
    sbatch slurm.sh export=config_path,action


2. Programmatically
-------------------
Sample code to run the ExactSolver on synthetic data.

.. code-block:: python

    # Run the exact solver
    from msp.solvers import ExactSolver
    from msp.datasets import make_sparse_data

    n_instances = 640
    dataset = make_sparse_data(n_instances, seed=2021) 
    solver = ExactSolver()

    for msp_batch_instance in dataset.batch(32):
        best_schedule, makespan = solver(msp_batch_instance)
        # store all best_schedule in an appropriate way


Sample code to create the instanace of MSP model.

.. code-block:: python

    # create a model instanace
    from msp.models import MSPModel
    from msp.models.encoders import GGCNEncoder
    from msp.models.decoders import AttentionDecoder

    encoder_params = {'units': 32}
    decoder_params = {'units': 32}
    model = MSPModel(GGCNEncoder, encoder_params, AttentionDecoder, decoder_params)
    
