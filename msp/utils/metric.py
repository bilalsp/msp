"""
The :mod:`mps.utils.metric` module defines tensorflow based metric to track the
model performance.
"""
import collections

import tensorflow as tf


class MeanMakespan(tf.keras.metrics.Metric):

    def __init__(self, name='mean_makespan', **kwargs):
        super(MeanMakespan, self).__init__(name=name, **kwargs)
        self.mean_makespan_train = self.add_weight(
            name='mean_makespan_train', initializer='zeros')
        self.mean_makespan_baseline = self.add_weight(
            name='mean_makespan_baseline', initializer='zeros')
        self.step = self.add_weight(
            name='iter_step', initializer='zeros')

    def update_state(self, makespan_train, makespan_baseline):
        """
        Accumulates statistics for the metric
        
        Args:
            makespan_train: makespan values for batch data by train model
            makespan_baseline: makespan values for batch data by baseline model
        """
        self.mean_makespan_train.assign_add(tf.reduce_mean(makespan_train))
        self.mean_makespan_baseline.assign_add(tf.reduce_mean(makespan_baseline))
        self.step.assign_add(1)

    def result(self):
        """Computes and returns the metric value tensor."""
        return dict(
            train = self.mean_makespan_train/self.step,
            baseline = self.mean_makespan_baseline/self.step
        )

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.mean_makespan_baseline.assign(0)
        self.mean_makespan_train.assign(0)
        self.step.assign(0)
