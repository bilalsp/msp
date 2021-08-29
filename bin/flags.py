"""
All flags required to run the experiments.
"""
from absl import flags
from absl.flags import FLAGS

import tensorflow as tf


############################################################################
# ...........................Common Config..................................
############################################################################
flags.DEFINE_string("action", None,
                    """Provides the action to run the experiment. 
                    Possible values are `run_solver`, `train_model`, `test_model`,
                    or `train_and_test_model`.""")
flags.mark_flag_as_required('action')
flags.DEFINE_string("config_path", None,
                    """Path to a YAML configuration files defining FLAG
                    values. Setting a key in these files is equivalent to 
                    setting the FLAG value with the same name.""")
flags.DEFINE_integer("experiment_id", None,
                    """Unique number for experiment id. If not provided, timestamp
                    will be considered.""")
flags.DEFINE_integer('batch_size', 64,
                    """Batch size used by solver or model""")
flags.DEFINE_string("output_dir", None,
                    """The directory to write solver checkpoints and summaries
                    to. If None, a local temporary directory is created.""")
flags.DEFINE_integer("seed", None,
                    """Random seed for sample data generation, and used by
                    `RandomSolver` if applicable. Setting this value allows 
                    consistency between reruns.""")

############################################################################
# ................................SOLVER Config.............................
############################################################################
flags.DEFINE_string('solver', None,
                    """"Name of the Solver class.
                    Can be either a fully-qualified name, or the name
                    of a class defined in `msp.solvers`.""")
flags.DEFINE_string('solver_params',"{}",
                    "YAML configuration string for the solver parameters.")
flags.DEFINE_string('data_params', "{}",
                    """YAML configuration string for the sample data 
                    parameters, defined in msp.dataset.make_sparse_data""")

############################################################################
# ....................TRAIN and TEST MODEL Config..........................
############################################################################
flags.DEFINE_string('model', None,
                    """"Name of the model class.
                    Can be either a fully-qualified name, or the name
                    of a class defined in `msp.models`.""")
flags.DEFINE_string('model_params',"{}",
                    "YAML configuration string for the model parameters.")
flags.DEFINE_string('train_data_params', "{}",
                    """YAML configuration string for the sample data 
                    parameters, defined in msp.dataset.make_sparse_data""")
flags.DEFINE_string('val_data_params', "{}",
                    """YAML configuration string for the sample data 
                    parameters, defined in msp.dataset.make_sparse_data""")
flags.DEFINE_string('test_data_params', "{}",
                    """YAML configuration string for the sample data 
                    parameters, defined in msp.dataset.make_sparse_data""")
flags.DEFINE_integer('epochs', 100,
                    """Number of epochs for model training.""")
flags.DEFINE_float('lr', 0.0001,
                   "Learning rate for optimizer.")
flags.DEFINE_float('tol', 0.001,
                   "tolerance for baseline update in REINFORCE algorithm.")
flags.DEFINE_boolean('is_retrain', False,
                     "Retrain the already trained model")
