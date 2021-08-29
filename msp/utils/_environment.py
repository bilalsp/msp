"""
The :mod:`mps.utils._environment` module defines RL-Environment.
"""
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from msp.utils._state import MSPState


class StepType(object):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST = np.asarray(0, dtype=np.int32)
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = np.asarray(1, dtype=np.int32)
  # Denotes the last `TimeStep` in a sequence.
  LAST = np.asarray(2, dtype=np.int32)

  def __new__(cls, value):
    """Add ability to create StepType constants from a value."""
    if value == cls.FIRST:
      return cls.FIRST
    if value == cls.MID:
      return cls.MID
    if value == cls.LAST:
      return cls.LAST

    raise ValueError('No known conversion for `%r` into a StepType' % value)


class TimeStep(
    NamedTuple('TimeStep', [('step_type', tf.TensorSpec),
                            ('mask', tf.TensorSpec),
                            ('mrg_machine', tf.TensorSpec)])):

    def is_first(self) -> tf.bool:
        return tf.equal(self.step_type, StepType.FIRST)
        
    def is_mid(self) -> tf.bool:
        return tf.equal(self.step_type, StepType.MID)

    def is_last(self) -> tf.bool:
        return tf.equal(self.step_type, StepType.LAST)


class MSPEnv(tf.Module):

    def __init__(self, **kwargs):
        self.is_build = False
        self._state = MSPState()
        super(MSPEnv, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.input_shape = input_shape
        self._state.build(input_shape)
        self.is_build = True

    def reset(self):
        """Returns the current `TimeStep` after resetting the Environment."""
        assert self.is_build, 'build the environment.'
        self._state.reset()
        time_step =  self.current_time_step()
        return time_step

    def current_time_step(self):
        """Returns the current `TimeStep`."""
        assert self.is_build, 'build the environment.'
        step_count = self._state.get_step_count()
        mask = self._state.get_mask()
        step_type = self._convert_to_step_type(step_count)
        return TimeStep(step_type, mask, self._state.mrg_machine)

    def step(self, actions):
        """Applies the action and returns the new `TimeStep`."""
        assert self.is_build, 'build the environment.'
        msp_inputs = actions.get('inputs')
        selected_node = actions.get('selected_node')
        self._state.update(msp_inputs, selected_node)
        return self.current_time_step()

    def _convert_to_step_type(self, step_count):
        batch_size, num_node, num_machine = self.input_shape.job_assignment
        first = lambda: tf.constant(0, dtype=tf.int32)
        mid = lambda: tf.constant(1, dtype=tf.int32)
        last = lambda: tf.constant(2, dtype=tf.int32)
        step_type = tf.case(
            [(tf.equal(step_count, tf.constant(1, dtype=tf.int32)), first),
             (tf.equal(step_count, tf.constant(num_node+1)), last)],
            exclusive=True, strict=True, default=mid)  
        return step_type
