"""
The :mod:`mps.utils._state` module defines RL-Environment state.
"""
import copy

import tensorflow as tf


class MSPState(tf.Module):

    def __init__(self):
        self.is_build = False

    def build(self, input_shape):
        """Create variables on first call."""
        self.input_shape = input_shape
        batch_size, num_node, num_machine = self.input_shape.job_assignment

        with tf.name_scope('tracking_vars'):
            # B x 1
            self._first_node = tf.Variable(
                initial_value=tf.zeros((batch_size, 1), dtype=tf.int64),
                trainable=False, 
                name='first_node'
            )
            
            # B x 1
            self._last_node = tf.Variable(
                initial_value=tf.zeros((batch_size, 1),  dtype=tf.int64),
                trainable=False, 
                name='last_node'
            )

            # B x V x 1
            self._visited_t = tf.Variable(
                initial_value=tf.zeros((batch_size, num_node, 1)),  
                trainable=False, 
                name='visited_nodes'
            )

            # B x n_machines x 1
            self._visited_mt = tf.Variable(
                initial_value=tf.zeros((batch_size, num_machine, 1)),
                trainable=False, 
                name='generated_job_subsequence_for_machine'
            )

            # B x 1
            self.mrg_machine = tf.Variable(
                initial_value=tf.zeros((batch_size,1), dtype=tf.int64),
                trainable=False, 
                name='most_recently_generated_machine'
            )

            # B x 1 x V
            self.mask = tf.Variable(
                initial_value=self._create_initial_mask(input_shape),
                trainable=False, 
                name='mask'
            )
            
            self.step_count = tf.Variable(
                initial_value=1, 
                trainable=False, 
                name='decoding_step_counter'
            )

        self.is_build = True

    @property
    def first_node(self):
        return self._first_node

    @property
    def last_node(self):
        return self._last_node

    def get_mask(self):
        return self.mask
    
    def get_step_count(self):
        return self.step_count

    def __call__(self, inputs, selected_node):
        """Call updates the current state."""
        if not self.is_built:
            self.build(tf.shape(inputs))
        self.update(inputs, selected_node)

    def update(self, inputs, selected_node):
        """ """
        if not self.is_build:
            self.build(tf.shape(inputs))

        batch_size = tf.shape(selected_node)[0]
        num_job = inputs.num_job
        node_type = inputs.node_features[:, :, -1]
        
        # B x 1
        type_of_selected_node = tf.cast(
            tf.gather(node_type, selected_node, batch_dims=1),
            dtype=tf.int64
        )

        # Update first node
        is_first_step = tf.equal(self.get_step_count(), 1)
        true_fn = lambda: selected_node
        false_fn = lambda: self._first_node
        self._first_node.assign(tf.cond(is_first_step, true_fn, false_fn))

        # Update last node
        self._last_node.assign(selected_node)

        # Update visited_mt
        is_not_first_step = tf.greater(self.get_step_count(), 1)
        def true_fn():
            temp = tf.zeros(tf.shape(self._visited_t))
            temp = tf.tensor_scatter_nd_update(
                tf.squeeze(temp, axis=-1),
                tf.concat([
                    tf.range(batch_size, dtype=tf.int64)[:, tf.newaxis],
                    self.mrg_machine * (1 - type_of_selected_node)
                ], axis=1),
                tf.ones((batch_size,), dtype=temp.dtype)
            ) 
            return temp[:, num_job:, tf.newaxis]
        false_fn = lambda : self._visited_mt
        self._visited_mt.assign_add(tf.cond(is_not_first_step, true_fn, false_fn))

        # Update visited nodes        
        self._visited_t.scatter_nd_update(
            tf.concat([
                tf.reshape(tf.range(batch_size, dtype=tf.int64), tf.shape(selected_node)),
                selected_node,
                tf.reshape(tf.zeros(batch_size, dtype=tf.int64), tf.shape(selected_node)),
            ], axis=1),
            tf.ones((batch_size,))
        )

        # Update most recently generated machine (mrg)
        self.mrg_machine.assign(
            tf.where(
                tf.cast(1 - type_of_selected_node, dtype=tf.bool), 
                selected_node, 
                self.mrg_machine
            )
        )
        
        # Update mask
        is_still_decoding = self.step_count < inputs.num_node
        true_fn = lambda: self._compute_mask(inputs)
        false_fn = lambda: self.mask
        self.mask.assign(tf.cond(is_still_decoding, true_fn, false_fn))

        # Update step count
        self.step_count.assign_add(1)

    def reset(self):
        assert self.is_build, "build the state module...."
        self._first_node.assign(tf.zeros(tf.shape(self._first_node), dtype=tf.int64))
        self._last_node.assign(tf.zeros(tf.shape(self._last_node),  dtype=tf.int64))
        self._visited_t.assign(tf.zeros(tf.shape(self._visited_t)))
        self._visited_mt.assign(tf.zeros(tf.shape(self._visited_mt)))
        self.mrg_machine.assign(tf.zeros(tf.shape(self.mrg_machine), dtype=tf.int64))
        self.mask.assign(self._create_initial_mask(self.input_shape))
        self.step_count.assign(1)

    # ##########################################################################
    # ...........................PRIVATE METHODS................................
    # ##########################################################################

    def _create_initial_mask(self, input_shape):
        batch_size, num_node, num_machine = input_shape.job_assignment
        large_negative_constant = tf.negative(1e10)
        # At timestep t=1
        job_type = tf.ones((batch_size, 1, num_node-num_machine), dtype=tf.float32)
        machine_type = tf.zeros((batch_size, 1, num_machine), dtype=tf.float32)
        node_type = tf.concat([job_type, machine_type], axis=-1)
        return tf.negative(1e10)*node_type

    def _compute_mask(self, inputs):
        """ """
        batch_size, num_node, num_machine = inputs.job_assignment.shape
        large_negative_constant = tf.negative(1e10)

        # B x 1 x V
        node_type = tf.transpose(inputs.node_features[:, :, -1:], perm=[0, 2, 1])
        mask_D_t = self._deadlock_prevention_mask(inputs, node_type)            
        mask_N_t = self._eligible_neighbor_nodes(inputs, node_type)
        mask_V_t = self._unvisited_mask()
        mask_t_prime = mask_D_t * mask_N_t * mask_V_t

        # B x 1 x V
        mask_t = large_negative_constant * (1 - mask_t_prime)
        
        is_mask_invalid = tf.reduce_any(
            tf.reduce_all(tf.equal(mask_t, large_negative_constant), axis=-1))

        tf.assert_equal(is_mask_invalid, False, message='invalid_mask')
        return mask_t

    def _deadlock_prevention_mask(self, inputs, node_type):
        """ """
        # B x 1 x 1
        is_deadlock  = self._is_deadlock(inputs)

        # B x 1 x V
        mask_D_t = tf.where(is_deadlock, node_type, tf.ones(tf.shape(node_type)))

        return mask_D_t

    def _eligible_neighbor_nodes(self, inputs, node_type):
        """ """
        adj_matrix = inputs.adj_matrix
        job_assignment = inputs.job_assignment
        num_job = inputs.num_job

        # B x 1
        mrg_machine_prime = tf.math.mod(self.mrg_machine, num_job)

        # B x 1 x V
        neighbours = tf.gather_nd(
            adj_matrix,
            indices=self.last_node[:,:,tf.newaxis],
            batch_dims=1 
        )

        # B x 1 x V
        mask_N_t = tf.multiply(
            neighbours,
            tf.add(
                1 - node_type,
                tf.gather(
                    tf.transpose(job_assignment, perm=[0, 2, 1]), 
                    indices=mrg_machine_prime, 
                    batch_dims=1
                )
            )
        )

        return mask_N_t

    def _unvisited_mask(self):
        """ """
        # B x 1 x V
        mask_V_t = 1 - tf.transpose(self._visited_t, perm=[0, 2, 1])
        return mask_V_t
            
    def _is_deadlock(self, inputs):
        """ """
        alpha = inputs.job_assignment
        num_job = inputs.num_job

        # B x n_machines x V
        alpha_prime = tf.transpose(alpha, perm=[0, 2, 1]) * (1 - self._visited_mt)
        # B x 1
        mrg_machine_prime = tf.math.mod(self.mrg_machine, num_job)

        # B x 1 x V
        alpha_j_prime = tf.gather(alpha_prime, indices=mrg_machine_prime, 
                                  batch_dims=1)
    
        # B x 1 x 1
        delta_t = tf.matmul(
            tf.cast(
                tf.greater(
                    alpha_j_prime,
                    tf.subtract(
                        tf.reduce_sum(alpha_prime, axis=-2, keepdims=True), 
                        alpha_j_prime
                    )
                ),
                dtype=tf.float32
            ),
            1 - self._visited_t
        )

        is_dead_lock = tf.greater(delta_t, 0)

        return is_dead_lock

