"""
The :mod:`mps.solvers._exact_solver` module defines exact solver.

Note: ExactSolver performs exhaustive search which is not tractable as the 
space of possible solutions increase for a large instance. Thus, it is 
applicable only for small size instance.
"""
import copy
import time
from collections import deque
from itertools import zip_longest
from queue import Queue

import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape

from msp.datasets import make_sparse_data
from msp.graphs import MSPSparseGraph
from msp.utils import MSPEnv
from msp.utils.objective import compute_makespan


class ExactSolver(tf.Module):

    def __init__(self, return_all_schedules=False, **kwargs):
        super(ExactSolver, self).__init__(name=kwargs.get('name', None))
        self.return_all_schedules = return_all_schedules 
        self.msp_env = MSPEnv()
        self.is_build = False       
    
    def build(self, input_shape):
        batch_size, num_node, num_node = input_shape.adj_matrix
        self.best_schedules = tf.Variable(
            initial_value=tf.zeros((batch_size, num_node, 2), dtype=tf.int64),
            trainable=False) 
        self.makespans = tf.Variable(
            initial_value=tf.constant(1e10, shape=(batch_size,1)),
            trainable=False)
        
        # ExactSolver run serially. So, remove batch dimension from input shape
        # before building an environment
        self.msp_env.build(self._remove_batch_dims(input_shape))
        self.is_build = True

    def __call__(self, inputs):
        # Create variables on first call.
        if not self.is_build:
            self.build(inputs.shape)

        # reintialize variables on each call.
        self.reset(inputs.shape)

        for idx, instance in enumerate(inputs.unbatch()):
            best_schedule, makespan = self._get_best_schedule(instance) 
            self.best_schedules.scatter_nd_update([[idx]], best_schedule)
            self.makespans.scatter_nd_update([[idx]], makespan)

        return self.best_schedules, self.makespans

    def reset(self, input_shape):
        batch_size, num_node, num_node = input_shape.adj_matrix
        best_schedules_shape = (batch_size, num_node, 2)
        makespans_shape = (batch_size, 1)
        self.best_schedules.assign(
            tf.zeros(best_schedules_shape, dtype=self.best_schedules.dtype)) 
        self.makespans.assign(
            tf.constant(1e10, shape=makespans_shape, dtype=self.makespans.dtype))

    def _get_best_schedule(self, instance):
        """Return best schedule out of all possible schedule."""
        env_stack = deque()
        schedule_stack = deque()
        num_node = instance.num_node

        time_step = self.msp_env.reset()
        all_possible_nodes = deque(
            self._mask_to_possible_nodes(time_step.mask)[tf.newaxis,0,:,:])
        env_stack.append((all_possible_nodes, self.msp_env))
        
        if self.return_all_schedules:
            all_generated_schedules = tf.cast(
                tf.reshape((), (0, num_node, 2)), dtype=tf.int64)

        best_schedule = tf.Variable(tf.zeros((1, num_node, 2), dtype=tf.int64)) 
        makespan_of_best_schedule = tf.Variable(tf.constant(1e10, shape=(1,1)))
     
        while env_stack:
            prev_all_possible_nodes, prev_env = env_stack.pop()

            if prev_all_possible_nodes:
                selected_node = prev_all_possible_nodes.pop()
                env_stack.append((prev_all_possible_nodes, prev_env))
              
                new_env = copy.deepcopy(prev_env)
                action = {'inputs': instance, 'selected_node': selected_node}

                time_step = new_env.step(action)

                schedule_stack.append(
                    [
                        tf.squeeze(selected_node).numpy(), 
                        tf.squeeze(time_step.mrg_machine).numpy()
                    ]
                )
                
                if time_step.is_last():
                    schedule = tf.constant(
                        schedule_stack, shape=(1, num_node, 2), dtype=tf.int64)
                    if self.return_all_schedules:
                        all_generated_schedules = tf.concat(
                            [all_generated_schedules, schedule], axis=0)
                    
                    makespan = compute_makespan(instance, schedule)

                    if tf.less(makespan, makespan_of_best_schedule):
                        best_schedule.assign(schedule)
                        makespan_of_best_schedule.assign(makespan)

                    schedule_stack.pop() # backtrack
                else:
                    all_possible_nodes = deque(
                        self._mask_to_possible_nodes(time_step.mask))
                    env_stack.append((all_possible_nodes, new_env))
            else:
                if schedule_stack:
                    schedule_stack.pop()

        return best_schedule, makespan_of_best_schedule

 
    def _mask_to_possible_nodes(self, mask):
        """Get all possible_nodes which can be visited at time step `t` 
        based on mask"""
        bool_mask = tf.not_equal(tf.squeeze(mask), tf.negative(1e10))
        possible_nodes = tf.expand_dims(tf.where(bool_mask), axis=-1)
        return possible_nodes

    def _remove_batch_dims(self, input_shape):
        return MSPSparseGraph(
            TensorShape([1]).concatenate(input_shape.adj_matrix[1:]),
            TensorShape([1]).concatenate(input_shape.node_features[1:]),
            TensorShape([1]).concatenate(input_shape.edge_features[1:]),
            TensorShape([1]).concatenate(input_shape.job_assignment[1:]),
        )

