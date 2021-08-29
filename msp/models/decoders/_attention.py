"""
The :mod:`mps.models._attention` module defines attention based 
decoder architecture.
"""
from collections import namedtuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model

from msp.layers import ContextEmbedding, MHALayer, SHALayer
from msp.utils import MSPEnv


class AttentionDecoder(Model):

    def __init__(self, 
                 units,
                 *args, 
                 use_bias=False,
                 n_heads=8,
                 aggregation_graph='mean',
                 tanh_clipping=10,
                 **kwargs):
        super(AttentionDecoder, self).__init__(*args,  **kwargs)

        self.aggregation_graph = aggregation_graph
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping

        assert units % n_heads == 0, \
               "Number of heads should be multiple of hidden dimensions."

        self.context_embed = ContextEmbedding(
            units, use_bias=use_bias, name='context_embed_layer')
        self.mha_layer = MHALayer(
            n_heads, units//n_heads, use_bias=use_bias, name='mha_layer')
        self.sha_layer = SHALayer(units, tanh_clipping=tanh_clipping)
        self.msp_env = MSPEnv()

        ############# For pre-computations ####################
        # 1. projection for graph embedding to build fixed context
        self.W_CG = tf.keras.layers.Dense(
            units, use_bias=use_bias, name='precomputation_W_CG')
        
        # 2. projections for MHA (keys and values, KV) and SHA (keys, K)
        self.W_KVK = tf.keras.layers.Dense(
            3*units, use_bias=use_bias, name='precomputation_W_KVK')
        
        self.is_build = False

    def call(self, inputs, training=None):
        """ """
        if not self.is_build:
            self.msp_env.build(inputs.shape)
            self.is_build = True

        schedules, log_probs = [], []
        time_step = self.msp_env.reset()

        # pre-computations (i.e., independent of timestep)
        precompute_res = self._precompute(inputs)
    
        for _ in range(inputs.num_node):

            state_t = self.msp_env._state

            # B x 1 x H
            query_t = self.context_embed([
                inputs, precompute_res.fixed_context, state_t])

            # B x 1 x V
            mask_t = time_step.mask  

            # B x 1 x H
            query_prime_t = self.mha_layer([
                query_t,
                precompute_res.mha_keys,
                precompute_res.mha_values,
                mask_t
            ])

            # B x 1 x V
            log_p = self.sha_layer([
                query_prime_t,
                precompute_res.sha_keys,
                mask_t
            ])

            # B x 1 (each)
            selected_node, prob_selected_node = self._select_node(log_p, mask_t, training)
            
            # update the state for next timestep t=t+1
            actions = {'inputs': inputs, 'selected_node': selected_node}
            time_step = self.msp_env.step(actions)
            # state_t.update(selected_node)

            schedules.append(tf.stack([selected_node, state_t.mrg_machine], axis=-1))
            log_probs.append(tf.math.log(prob_selected_node))
                
        # B x 1
        sum_log_probs = tf.reduce_sum(tf.concat(log_probs, axis=-1), axis=-1)
        
        # B x V x 2
        schedules = tf.concat(schedules, axis=1)

        return schedules, sum_log_probs

    def _precompute(self, inputs):
        """Precompute keys and values for efficiency."""
        _precompute_res = namedtuple(
            'precompute',
            ['fixed_context', 'mha_keys', 'mha_values', 'sha_keys']
        )

        # B x 1 x H
        graph_embed = self._get_graph_embed(inputs.node_embed)

        # precomputation for context embedding
        # B x 1 x H
        fixed_context = self.W_CG(graph_embed)
        
        # precomputation for MHA and SHA 
        # B x 1 x V x H (each)
        mha_keys, mha_values, sha_keys = tf.split(
            self.W_KVK(tf.expand_dims(inputs.node_embed, axis=-3)),
            num_or_size_splits=3,
            axis=-1
        )

        results = _precompute_res(
            fixed_context=fixed_context,
            mha_keys=mha_keys, 
            mha_values=mha_values, 
            sha_keys=sha_keys
        )

        return results

    def _get_graph_embed(self, node_embed):
        if self.aggregation_graph == "sum":
            graph_embed = tf.reduce_sum(node_embed, axis=-2)
        elif self.aggregation_graph == "max":
            graph_embed = tf.reduce_max(node_embed, axis=-2)
        elif self.aggregation_graph == "mean":
            graph_embed = tf.reduce_mean(node_embed, axis=-2)
        else:  # dissable graph embedding
            graph_embed = tf.reduce_sum(node_embed, axis=-2) * 0.0

        return tf.expand_dims(graph_embed, axis=-2)

    def _select_node(self, log_p, mask, training):
        """Select the node based on log-probabilities return by model"""
        tf.assert_equal(tf.reduce_any(tf.math.is_nan(log_p)), False,
                        message="Log probabilities over the nodes should be defined.")

        # B x 1 x V
        probs = tf.exp(log_p)

        tf.assert_equal(tf.reduce_all(probs == probs), True, 
                        message='Probs should not contain any nans')

        is_decoding_correct = lambda selected_node: tf.reduce_all(
            tf.equal(
                    tf.gather_nd(
                        tf.squeeze(mask, axis=1), selected_node, batch_dims=1),
                    0
            )
        )
    
        if training:
            dist = tfp.distributions.Categorical(probs=probs, dtype=tf.int64)
            selected_node_temp = tf.squeeze(dist.sample(1), axis=0)
            # Check if sampling was correct 
            is_decoding_incorrect = lambda selected_node: tf.logical_not(
                is_decoding_correct(selected_node)
            )
            def body(selected_node_temp):
                tf.print('Sampled bad values, resampling!')
                return [tf.squeeze(dist.sample(1), axis=0)]
            selected_node = tf.while_loop(
                is_decoding_incorrect,
                body,
                [selected_node_temp],
                parallel_iterations=1
            )[0]
        else:
            selected_node = tf.math.argmax(probs, axis=-1)
            tf.assert_equal(is_decoding_correct(selected_node), True,
                            message="Greedy decoding: infeasible action has maximum probability")

        # B x 1
        prob_selected_node = tf.gather(tf.squeeze(probs), selected_node, batch_dims=1)

        return selected_node, prob_selected_node
