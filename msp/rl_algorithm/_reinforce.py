"""
The :mod:`mps.rl_algorithm._reinforce` module defines reinforcement learning
algorithm to train the model.

Reference:
    Kool, Wouter, H. V. Hoof and M. Welling. 
    “Attention, Learn to Solve Routing Problems!” ICLR (2019).
"""
import time

import numpy as np
import tensorflow as tf


class ReinforceAlgorithm:

    def __init__(self, objective_func, optimizer, train_metric, val_metric, 
                 tol=1e-3):
        """REINFORCE Algorithm to train the model.
        
        Args:
            objective_func: objective function of the problem
            optimizer: optimizer to update model weights
            train_metric: metric to record model performance on training dataset
            val_metric:metric to record model performance on validation dataset
            tol: tolerance to update the baseline model 
        """
        self.optimizer = optimizer
        self.objective_func = objective_func
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.tol = tol

    def run(self, model_train, model_baseline, train_dataset, val_dataset, 
            epochs=1, verbose=1, callbacks=None, epoch_when_last_save=0):
        """Run the REINFORCE Algorithm on model_train and baseline.

        Args:
            model_train: model for the training
            model_baseline: model baseline for comparison as per `REINFORCE ALOGRITHM`
            train_dataset: dataset for training
            val_dataset: dataset for validation
            epochs: number of epochs
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
            callbacks: List of callbacks to apply during training.
            epoch_when_last_save: Used when retraining a model
        """
        # Container that configures and calls `tf.keras.Callback`s.
        baseline_callbacks = tf.keras.callbacks.CallbackList(
            callbacks[-1],
            add_history=True,
            add_progbar=False,
            model=model_baseline,
            verbose=verbose,
            epochs=epochs)
        callbacks = tf.keras.callbacks.CallbackList(
            callbacks[:-1],
            add_history=True,
            add_progbar=verbose != 0,
            model=model_train,
            verbose=verbose,
            epochs=epochs,
            steps=len(list(enumerate(train_dataset))))

        callbacks.on_train_begin()
        baseline_callbacks.on_train_begin()
        train_cum_time, val_cum_time = 0, 0

        for epoch in range(epoch_when_last_save, epochs):
            # print('Epoch {}/{}'.format(epoch+1, epochs))
            logs = {}
            callbacks.on_epoch_begin(epoch)
            baseline_callbacks.on_epoch_begin(epoch)

            ############## Model Training ##############
            start_time = time.perf_counter()
            train_losses = self.train_model_for_one_epoch(
                model_train, 
                model_baseline, 
                train_dataset,
                callbacks,
                verbose=verbose
            )
            train_metric_result = self.train_metric.result()
            train_time =  time.perf_counter() - start_time
            train_cum_time += train_time

            ############## Model Validation ##############
            start_time = time.perf_counter()
            val_losses = self.evaluate(model_train, model_baseline, val_dataset)
            val_metric_result = self.val_metric.result()
            val_time = time.perf_counter() - start_time
            val_cum_time += val_time

            # update baseline model if train model is better
            update_baseline = val_metric_result['train'] + self.tol < val_metric_result['baseline']
            if update_baseline:
                model_baseline.set_weights(model_train.get_weights())
            

            logs['train_time'] = train_time
            logs['cum_train_time'] = train_cum_time
            logs['train_loss'] = np.mean(train_losses)
            logs['makespan_train_data_by_train_model'] = train_metric_result['train']
            logs['makespan_train_data_by_baseline_model'] = train_metric_result['baseline']
            logs['val_time'] = val_time
            logs['cum_val_time'] = val_cum_time
            logs['val_loss'] = np.mean(val_losses)
            logs['makespan_val_data_by_train_model'] = val_metric_result['train']
            logs['makespan_val_data_by_baseline_model'] = val_metric_result['baseline']
            logs['is_baseline_updated'] = update_baseline

            # Reset states of all metrics
            self.train_metric.reset_states()
            self.val_metric.reset_states()

            baseline_callbacks.on_epoch_end(epoch, logs)
            callbacks.on_epoch_end(epoch, logs)

        baseline_callbacks.on_train_end()
        callbacks.on_train_end()

    def train_model_for_one_epoch(self, model_train, model_baseline, 
                                 train_dataset, callbacks, verbose=1):
        """Computes the loss then updates the weights and metrics for one epoch.
    
        Args:
            model_train: model for the training
            model_baseline: model baseline for comparison as per `REINFORCE ALOGRITHM`
            train_dataset: dataset for training
            callbacks: List of callbacks to apply during training.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
        """
        losses = []

        # Iterate over all the batches of the dataset.
        for step, batch_train in enumerate(train_dataset):
            callbacks.on_train_batch_begin(step)

            objective_train, objective_baseline, loss_value = self.apply_gradient(
                model_train,
                model_baseline,
                batch_train
            )
            losses.append(loss_value)
            self.train_metric.update_state(objective_train, objective_baseline)

            logs_ = {
                'train_loss': float(loss_value), 
                'makespan_train': self.train_metric.result()['train'],
                'makespan_baseline': self.train_metric.result()['baseline']
                }
            callbacks.on_train_batch_end(step+1, logs=logs_)
    
        return losses

    def apply_gradient(self, model_train, model_baseline, batch_train):
        """Apply the gradients to the trainable model weights.

        Args:
            model_train: model for the training
            model_baseline: model baseline for comparison as per `REINFORCE ALOGRITHM`
            batch_train: input mini batch data
        """
        # with no gradient 
        schedules_baseline, _ = model_baseline(batch_train, training=False)

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # B x V x 2, B x 1
            schedules_train, sum_log_probs = model_train(batch_train, training=True)
            
            # compute problem objective 
            objective_train = self.objective_func(batch_train, schedules_train)
            objective_baseline = self.objective_func(batch_train, schedules_baseline)

            # Compute the loss value for this minibatch.
            loss_value = tf.reduce_mean((objective_train - objective_baseline) * sum_log_probs)
        
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model_train.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, model_train.trainable_weights))

        return objective_train, objective_baseline, loss_value

    def evaluate(self, model_train, model_baseline, val_dataset, verbose=1,
                 callbacks=None):
        """Perform model evaluation.

        Args:
            model_train: model for the training
            model_baseline: model baseline for comparison as per `REINFORCE ALOGRITHM`
            val_dataset: dataset for validation
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
            callbacks: List of callbacks to apply during evaluation.
        """
        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
            callbacks = tf.keras.callbacks.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=model_train,
                verbose=verbose,
                epochs=1,
                steps=len(list(enumerate(val_dataset))))

        losses = []
        callbacks.on_test_begin()

        # Iterate over the batches of the dataset.
        for step, batch_val in enumerate(val_dataset):
            callbacks.on_test_batch_begin(step)
            t = time.perf_counter()
            
            schedules_train, sum_log_probs = model_train(batch_val, training=False)
            schedules_baseline, _ = model_baseline(batch_val, training=False)
        
            # compute problem objective
            objective_train = self.objective_func(batch_val, schedules_train)
            objective_baseline = self.objective_func(batch_val, schedules_baseline)
            
            # Compute the loss value for this minibatch.
            loss_value = tf.reduce_mean((objective_train - objective_baseline) * sum_log_probs)
            losses.append(loss_value)

            # update validation metrics
            self.val_metric.update_state(objective_train, objective_baseline)

            logs_ = {
                'val_loss': float(loss_value), 
                'makespan_train': self.val_metric.result()['train'],
                'makespan_baseline': self.val_metric.result()['baseline']
                }
            callbacks.on_test_batch_end(step+1, logs=logs_)
        callbacks.on_test_end()

        return losses
