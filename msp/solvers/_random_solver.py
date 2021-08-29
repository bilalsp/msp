"""
The :mod:`mps.solvers._random_solver` module defines random solver.

Note: Random solver return best schedule from pre-defined search space decided
based on parameter `best_out_of`. 
"""
import functools

import tensorflow as tf
import tensorflow_probability as tfp

from msp.utils import MSPEnv
from msp.utils.objective import compute_makespan


class RandomSolver(tf.Module):

    def __init__(self, best_out_of=100, seed=None, **kwargs):
        super(RandomSolver, self).__init__(**kwargs)
        self.msp_env = MSPEnv()
        self.best_out_of = best_out_of
        self.seed = seed
        self.is_build = False
        if seed:
            self.rand_gen = tf.random.experimental.Generator.from_seed(seed, alg="philox")
        else:
            self.rand_gen = tf.random.Generator.from_non_deterministic_state()

    def build(self, input_shape):  
        batch_size, num_node, num_node = input_shape.adj_matrix
        self.best_schedules = tf.Variable(
            initial_value=tf.zeros((batch_size, num_node, 2), dtype=tf.int64),
            trainable=False) 
        self.best_makespans = tf.Variable(
            initial_value=tf.constant(1e10, shape=(batch_size,1)),
            trainable=False)
        self.msp_env.build(input_shape)
        self.is_build = True

    def __call__(self, inputs):
        # Create variables on first call.
        if not self.is_build:
            self.build(inputs.shape)

        # reintialize variables on each call.
        self.reset(inputs.shape)

        for _ in range(self.best_out_of):
            # randomly generates schedules 
            schedules, makespans = self._gen_rand_schedules(inputs)
            # update best schedule and makespan
            self.update(schedules, makespans)
            
        return self.best_schedules, self.best_makespans

    def reset(self, input_shape):
        batch_size, num_node, num_node = input_shape.adj_matrix
        best_schedules_shape = (batch_size, num_node, 2)
        best_makespans_shape = (batch_size, 1)
        self.best_schedules.assign(
            tf.zeros(best_schedules_shape, dtype=self.best_schedules.dtype)) 
        self.best_makespans.assign(
            tf.constant(1e10, shape=best_makespans_shape, dtype=self.best_makespans.dtype))

    def update(self, schedules, makespans):
        self.best_schedules.assign(
                tf.where(
                    tf.less(makespans, self.best_makespans)[:,:,tf.newaxis],
                    schedules,
                    self.best_schedules
            ))
        self.best_makespans.assign(
            tf.where(
                tf.less(makespans, self.best_makespans),
                makespans,
                self.best_makespans
        ))

    @tf.function
    def _gen_rand_schedules(self, inputs):
        """Generates a random schedule for a given input."""
        schedules = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
        time_step = self.msp_env.reset()

        step = 0
        while not time_step.is_last():
            selected_node = self._select_node(time_step.mask)
            actions = {'inputs': inputs, 'selected_node': selected_node}
            time_step = self.msp_env.step(actions)
            schedules = schedules.write(step, tf.stack([selected_node, time_step.mrg_machine], axis=-1))
            step += 1
        
        # TensorArray --> Tensor
        schedules = tf.transpose(schedules.stack(), perm=[1,0,2,3])
        schedules = tf.squeeze(schedules)

        # B x V x 2
        schedules = tf.concat(schedules, axis=1)
        return schedules, compute_makespan(inputs, schedules)

    def _select_node(self, mask):
        """Randomly select a node based on mask."""
        rand_logits = self.rand_gen.normal(mask.shape) + mask
        rand_probs = tf.nn.softmax(rand_logits, axis=-1) 
        dist = tfp.distributions.Categorical(probs=rand_probs, dtype=tf.int64)
        selected_node = tf.squeeze(dist.sample(1, seed=self.seed), axis=0)
        return selected_node
        
     