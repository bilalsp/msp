"""
Script to train and test the model of MSP problem.
"""
import time
import os
import sys
import re
import errno 
from datetime import datetime
from copy import deepcopy
from pydoc import locate
from absl.flags import FLAGS
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow.io.gfile as gfile

from msp.datasets import make_sparse_data
from msp import models
from msp.models import encoders
from msp.models import decoders
from msp.rl_algorithm import ReinforceAlgorithm
from msp.utils.metric import MeanMakespan
from msp.utils.objective import compute_makespan


def _create_experiment_dir():
    # Create directory for experiment if does not exist.
    output_dir = os.path.abspath(FLAGS.output_dir.strip())
    experiment_dir = output_dir + '/experiment_'+ str(FLAGS.experiment_id)
    try:
        os.makedirs(experiment_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return experiment_dir

def check_weights(weight_before_update, weight_after_update):
    """Check whether weights have changed or not"""
    variables = []
    for w1, w2 in zip(weight_before_update, weight_after_update):
        try:
            tf.assert_equal(tf.reduce_all(tf.equal(w1, w2)), 
            False, 
            message='Weights are same')
        except Exception as e:
            variables.append((w1.name, w2.name))
    if len(variables) > 0:
        logging.error(
            'Total Variables: {}, Non updated variable count: {}'.format(
                len(weight_before_update), len(variables)))
        logging.error('List of non updated variables: '+ str(variables))

def _create_or_restore_models(experiment_dir, single_batch_instance, restore=False):
    # find the model class
    model_params = FLAGS.model_params
    model_class = locate(FLAGS.model) or getattr(models, FLAGS.model)
    encoder_class = locate(model_params['encoder']) or getattr(encoders, model_params['encoder'])
    decoder_class = locate(model_params['decoder']) or getattr(decoders, model_params['decoder'])
    
    model_train = model_class(
        encoder_class,
        model_params['encoder_params'],
        decoder_class,
        model_params['decoder_params']
    )
    
    model_baseline = model_class(
        encoder_class,
        model_params['encoder_params'],
        decoder_class,
        model_params['decoder_params']
    )
    
    epoch_when_last_save = 0
    # perform single forward pass, required to track initial and updated weights
    model_train(single_batch_instance) 
    model_baseline(single_batch_instance)
    
    if restore:
        try:
            # restore model_train
            checkpoint_dir = experiment_dir + "/"+ "/saved_model/train"
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info(
                "Loading the previously saved weights of model_train from " + latest_checkpoint) 
            status = model_train.load_weights(latest_checkpoint)
            status.assert_consumed() # would throw an exception if architecture is different
            # restore model_baseline
            checkpoint_dir = experiment_dir + "/saved_model/baseline"
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info(
                "Loading the previously saved weights of model_baseline from " + latest_checkpoint)
            status = model_baseline.load_weights(latest_checkpoint)
            status.assert_consumed() 
        except Exception as e:
            logging.error('Trained model does not exist...', str(e))
            sys.exit() 
        if FLAGS.is_retrain:
            # epoch when model were saved last time (required for model retraining)
            epoch_when_last_save = int(
                re.findall(r'cp-(.*).ckpt', os.path.basename(latest_checkpoint))[0])
            logging.info('epoch_when_last_save: '+str(epoch_when_last_save))
            if FLAGS.epochs <= epoch_when_last_save:
                logging.error(
                    'epoch should be greater than `epoch_when_last_save` to retrain the model.')
                sys.exit() 

    return model_train, model_baseline, epoch_when_last_save

def train():
    logging.info('Retraining the model....' if FLAGS.is_retrain else 'Training the model....')
    experiment_dir = _create_experiment_dir()

    # train dataset
    train_data_params = FLAGS.train_data_params
    seed = train_data_params.pop('seed', None)
    n_instances = train_data_params.pop('n_instances')
    train_dataset = make_sparse_data(n_instances, seed=seed, **train_data_params)
    train_dataset = train_dataset.batch(FLAGS.batch_size)

    # validation dataset
    val_data_params = FLAGS.val_data_params
    seed = train_data_params.pop('seed', FLAGS.seed)
    n_instances = val_data_params.pop('n_instances')
    val_dataset = make_sparse_data(n_instances, seed=seed, **val_data_params)
    val_dataset = val_dataset.batch(FLAGS.batch_size)

    # Baseline model and Training model `as per REINFORCE Algorithm`
    single_batch_instance = list(train_dataset.take(1))[0]
    model_train, model_baseline, epoch_when_last_save = _create_or_restore_models(
        experiment_dir, single_batch_instance, restore=FLAGS.is_retrain)
    
    # Initial Weights of models
    model_train_initial_weights = deepcopy(model_train.trainable_weights)
    model_baseline_initial_weights = deepcopy(model_baseline.trainable_weights) 
    
    # Define Callbacks
    callbacks = [
        tf.keras.callbacks.CSVLogger(
            experiment_dir+'/training_log-{}.log'.format(datetime.now().strftime("%d-%m-%Y-%f"))),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=experiment_dir+'/saved_model/train'+'/cp-{epoch:04d}.ckpt',
            verbose=1,
            save_weights_only=True,
            save_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=experiment_dir+'/saved_model/baseline'+'/cp-{epoch:04d}.ckpt',
            verbose=1,
            save_weights_only=True,
            save_freq='epoch')
    ]

    # optimizer to update model weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)

    # metrics
    train_metric = MeanMakespan()
    val_metric = MeanMakespan()

    reinforce_algorithm = ReinforceAlgorithm(
        compute_makespan, optimizer, train_metric, val_metric, tol=FLAGS.tol)

    # Run Reinforce Algorithm
    reinforce_algorithm.run(
        model_train, 
        model_baseline, 
        train_dataset, 
        val_dataset, 
        epochs=FLAGS.epochs,
        callbacks=callbacks,
        epoch_when_last_save=epoch_when_last_save)

    # check weights after training
    check_weights(model_train_initial_weights, model_train.trainable_weights)
    check_weights(model_baseline_initial_weights, model_baseline.trainable_weights)
    
def test():
    logging.info('Testing the model....')
    experiment_dir = _create_experiment_dir()
    
    # test dataset
    test_data_params = FLAGS.test_data_params
    seed = test_data_params.pop('seed', FLAGS.seed)
    n_instances = test_data_params.pop('n_instances')
    test_dataset = make_sparse_data(n_instances, seed=seed, **test_data_params)
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    # trained model
    single_batch_instance = list(test_dataset.take(1))[0]
    model_train, _, _ = _create_or_restore_models(experiment_dir, 
        single_batch_instance, restore=True)

    batch_size = FLAGS.batch_size
    cum_makespan_mean = 0
    results = []
    schedules, makespans = [], []

    start_time = time.perf_counter()
    for idx, msp_problem in enumerate(test_dataset):
        t1 = time.perf_counter()
        schedule, _ = model_train(msp_problem)
        makespan = compute_makespan(msp_problem, schedule)  
        t2 = time.perf_counter()

        schedules.append(schedule)
        makespans.append(makespan)
        
        makespan_mean = tf.reduce_mean(makespan) 
        cum_makespan_mean = (
            (idx*batch_size)*cum_makespan_mean + batch_size*makespan_mean)/((idx+1)*batch_size)
        cum_time = t2 - start_time
        results.append([idx, makespan_mean, cum_makespan_mean, t2-t1, cum_time])
        logging.info("Batch_ID:{}, makespan_mean: {}".format(idx, makespan_mean.numpy()))

    end_time = time.perf_counter()

    # convert to tensor
    tf_schedules = tf.stack(schedules)
    tf_makespans = tf.stack(makespans)

    # save tf_schedules and tf_makespans ---> as numpy array
    filename = experiment_dir + '/schedulesWithMakespan.npy'
    with gfile.GFile(filename, "wb") as f:
        np.save(f, tf_schedules.numpy())
        np.save(f, tf_makespans.numpy())

    # save results --> as a numpy array
    filename = experiment_dir + '/results.npy'
    with gfile.GFile(filename, "wb") as f:
        np.save(filename, np.array(results))
