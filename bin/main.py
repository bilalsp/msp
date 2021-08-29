"""
Entry point script to run any experiment.
"""
import os
import sys
from datetime import datetime
from absl import app, logging
from absl.flags import FLAGS

import yaml
import tensorflow as tf
import tensorflow.io.gfile as gfile

from msp.utils.configs import load_yaml_flag

import flags
import solvers
import models


def run_experiment():
    if FLAGS.action == 'run_solver':
        solvers.run()
    elif FLAGS.action == 'train_model':
        models.train()
    elif FLAGS.action == 'test_model':
        models.test()
    elif FLAGS.action == 'train_and_test_model':
        models.train()
        models.test()
    else:
        sys.exit("Received invalid action.....") 

def main(argv):
    # Parse YAML FLAGS
    FLAGS.solver_params = load_yaml_flag(FLAGS.solver_params)
    FLAGS.data_params =  load_yaml_flag(FLAGS.data_params)
    
    if FLAGS.config_path:
        config_path = os.path.abspath(FLAGS.config_path.strip())
        logging.info("Loading config from %s", config_path)
        with gfile.GFile(config_path) as config_file:
            config_flags = yaml.safe_load(config_file)

    # Set flags based on config_flags
    for flag_key, flag_value in config_flags.items():
        if hasattr(FLAGS, flag_key):
            setattr(FLAGS, flag_key, flag_value)
        else:
            logging.warning("Ignoring config flag: %s", flag_key)

    if not FLAGS.output_dir:
        FLAGS.output_dir = tempfile.mkdtemp()

    if not FLAGS.experiment_id:
        FLAGS.experiment_id = datetime.now().strftime("%d-%m-%Y-%f")

    logging.info("Final Config:\n%s", yaml.dump(FLAGS.flag_values_dict()))
    logging.info('_'*10+'EXPERIMENT HAS STARTED'+'_'*10+'\n')
    run_experiment()
    logging.info('_'*10+'EXPERIMENT ENDED'+'_'*10+'\n')


if __name__ == "__main__":
    # logging.set_verbosity(logging.WARN)
    logging.set_verbosity(logging.INFO)  
    app.run(main)
