"""
Script to run the solvers (`Exact` and `Random`) of MSP problem.
"""
import time
import os
import errno 
from pydoc import locate
from absl.flags import FLAGS
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow.io.gfile as gfile

from msp.datasets import make_sparse_data
from msp import solvers


def pre_run():
    # Create directory for experiment if does not exist.
    output_dir = os.path.abspath(FLAGS.output_dir.strip())
    output_dir = output_dir + '/experiment_'+ str(FLAGS.experiment_id)
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return output_dir

def post_run(solver, tf_best_schedules, tf_best_makespans, results, output_dir):
    # save tf_best_schedules and tf_best_makespans ---> as numpy array
    filename = output_dir+'/schedulesWithMakespan.npy'
    with gfile.GFile(filename, "wb") as f:
        np.save(f, tf_best_schedules.numpy())
        np.save(f, tf_best_makespans.numpy())

    # save results --> as a numpy array
    filename = output_dir+'/results.npy'
    with gfile.GFile(filename, "wb") as f:
        np.save(filename, np.array(results))
    
    # save solver 
    if FLAGS.solver == 'RandomSolver':
        filenme = output_dir + '/saved_model'
        tf.saved_model.save(solver, filenme)

def run():
    logging.info('Running solver....')
    output_dir = pre_run()

    n_instances = FLAGS.data_params.pop('n_instances')
    batch_size = FLAGS.batch_size
    seed = FLAGS.data_params.pop('seed', FLAGS.seed)
    dataset = make_sparse_data(n_instances, seed=seed, **FLAGS.data_params)
    dataset = dataset.batch(batch_size)

    # find the solver class
    solver_class = locate(FLAGS.solver) or getattr(solvers, FLAGS.solver)
    seed = FLAGS.solver_params.pop('seed', FLAGS.seed)
    
    # restore solver if exist..
    solver_dir = output_dir + '/saved_model'
    if os.path.exists(solver_dir):
        solver = tf.saved_model.load(solver_dir)
        logging.info('Loaded existing solver.')
    else:
        solver = solver_class(seed=seed, **FLAGS.solver_params)
    
    cum_makespan_mean = 0
    results = []
    best_schedules, best_makespans = [], []

    start_time = time.perf_counter()
    for idx, msp_problem in enumerate(dataset):
        t1 = time.perf_counter()
        best_schedule, best_makespan = solver(msp_problem)
        t2 = time.perf_counter()

        best_schedules.append(best_schedule)
        best_makespans.append(best_makespan)
        
        makespan_mean = tf.reduce_mean(best_makespan) 
        cum_makespan_mean = ((idx*batch_size)*cum_makespan_mean + batch_size*makespan_mean)/((idx+1)*batch_size)
        cum_time = t2 - start_time
        results.append([idx, makespan_mean, cum_makespan_mean, t2-t1, cum_time])
        logging.info("Batch_ID:{}, makespan_mean: {}".format(idx, makespan_mean.numpy()))

    end_time = time.perf_counter()

    # convert to tensor
    tf_best_schedules = tf.stack(best_schedules)
    tf_best_makespans = tf.stack(best_makespans)

    post_run(solver, tf_best_schedules, tf_best_makespans, results, output_dir)
