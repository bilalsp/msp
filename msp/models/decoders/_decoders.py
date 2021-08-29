"""
https://www.tensorflow.org/text/tutorials/transformer
"""

"""
###################
# Masking Pending .........................
##################
"""
import math

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model

from msp.layers import GGCNLayer
from msp.graphs import MSPSparseGraph
from msp.solutions import MSPState


from typing import NamedTuple


from tensorflow import Tensor
class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: Tensor
    context_node_projected: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    def __getitem__(self, key):
        if tf.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionDecoder(Model):

    def __init__(self, 
                 units,
                 *args, 
                 activation='relu',
                 aggregation_graph='mean',
                 n_heads=2, # make it 8
                 mask_inner=True,
                 tanh_clipping=10,
                 decode_type='sampling',
                 extra_logging=False,
                 **kwargs):
        """ """
        super(AttentionDecoder, self).__init__(*args, **kwargs)
        self.aggregation_graph = aggregation_graph
        self.n_heads = n_heads
        self.mask_inner = mask_inner
        self.tanh_clipping = tanh_clipping
        self.decode_type = decode_type
        self.extra_logging = extra_logging

        embedding_dim = units
        
        self.W_placeholder = self.add_weight(shape=(2*embedding_dim,),
                                initializer='random_uniform', #Placeholder should be in range of activations (think)
                                name='W_placeholder',
                                trainable=True)

        graph_embed_shape = tf.TensorShape((None, units))
        self.fixed_context_layer = tf.keras.layers.Dense(units, use_bias=False)
        self.fixed_context_layer.build(graph_embed_shape)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        project_node_embeddings_shape = tf.TensorShape((None, None, None, units))
        self.project_node_embeddings = tf.keras.layers.Dense(3*units, use_bias=False)
        self.project_node_embeddings.build(project_node_embeddings_shape)

        #
        # Embedding of first and last node
        step_context_dim = 2*units
        project_step_context_shape = tf.TensorShape((None, None, step_context_dim))
        self.project_step_context = tf.keras.layers.Dense(embedding_dim, use_bias=False)
        self.project_step_context.build(project_step_context_shape)

        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim

        project_out_shape = tf.TensorShape((None, None, 1, embedding_dim))
        self.project_out = tf.keras.layers.Dense(embedding_dim, use_bias=False)
        self.project_out.build(project_out_shape)


        # self.context_layer = tf.keras.layers.Dense(units, use_bias=False)
        # self.mha_layer = None
        

        # dynamic router


        
    def call(self, inputs, training=False, return_pi=False):
        """ """
        state = MSPState(inputs)

        node_embedding = inputs.node_embed

        # AttentionModelFixed(node_embedding, fixed_context, *fixed_attention_node_data)
        fixed = self._precompute(node_embedding)


        # for i in range(num_steps):
            # i == 0 should be machine 
            # AttentionCell(inputs, states)

        outputs = []
        sequences = []

        # i = 0
        while not state.all_finished():
            # B x 1 x V
            # Get log probabilities of next action
            log_p, mask = self._get_log_p(fixed, state)
            
            selected = self._select_node(
                    tf.squeeze(tf.exp(log_p), axis=-2), tf.squeeze(mask, axis=-2)) # Squeeze out steps dimension

            state.update(selected)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            
            # if i == 1:
            #     break
            # i+=1
        
        _log_p, pi = tf.stack(outputs, axis=1), tf.stack(sequences, axis=1)

        if self.extra_logging:
            self.log_p_batch = _log_p
            self.log_p_sel_batch = tf.gather(tf.squeeze(_log_p,axis=-2), pi, batch_dims=1)

        # # Get predicted costs
        # cost, mask = self.problem.get_costs(nodes, pi)
        mask = None

        ###################################################
        # Need Clarity #############################################################
        # loglikelihood 
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        ## Just for checking
        return_pi = True    
        if return_pi:
            return state.makespan, ll, pi

        return state.makespan, ll

        

        



    def _precompute(self, node_embedding, num_steps=1):

        graph_embed = self._get_graph_embed(node_embedding)

        fixed_context = self.fixed_context_layer(graph_embed)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = tf.expand_dims(fixed_context, axis=-2)

        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed  = tf.split(
            self.project_node_embeddings(tf.expand_dims(node_embedding, axis=-3)),
            num_or_size_splits=3,
            axis=-1
        )

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed
        )
        return AttentionModelFixed(node_embedding, fixed_context, *fixed_attention_node_data)

    def _get_graph_embed(self, node_embedding):
        """ """
        if self.aggregation_graph == "sum":
            graph_embed = tf.reduce_sum(node_embedding, axis=-2)
        elif self.aggregation_graph == "max":
            graph_embed = tf.reduce_max(node_embedding, axis=-2)
        elif self.aggregation_graph == "mean":
            graph_embed = tf.reduce_mean(node_embedding, axis=-2)
        else:  # Default: dissable graph embedding
            graph_embed = tf.reduce_sum(node_embedding, axis=-2) * 0.0

        return graph_embed

    def _make_heads(self, v, num_steps=None):

        assert num_steps is None or v.shape[1] == 1 or v.shape[1] == num_steps
        batch_size, _, num_nodes, embed_dims = v.shape
        num_steps = num_steps if num_steps else 1
        head_dims = embed_dims//self.n_heads

        # M x B x N x V x (H/M)
        return tf.transpose(
            tf.broadcast_to(
                tf.reshape(v, shape=[batch_size, v.shape[1], num_nodes, self.n_heads, head_dims]),
                shape=[batch_size, num_steps, num_nodes, self.n_heads, head_dims]
            ),
            perm=[3, 0, 1, 2, 4]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        # Compute query = context node embedding
        
        # B x 1 x H
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
        
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask, for masking next action based on previous actions
        mask = state.get_mask()
        graph_mask = state.get_graph_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, graph_mask)

        # [B x N x V]
        # log-softmax activation function so that we get log probabilities over actions
        if normalize:
            log_p = tf.nn.log_softmax(log_p/1.0, axis=-1)

        assert not tf.reduce_any(tf.math.is_nan(log_p)), "Log probabilities over the nodes should be defined"

        return log_p, mask


    def _get_parallel_step_context(self, node_embedding, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once 
        (for efficient evaluation of the model)
        """
        # last_node at time t
        last_node = state.get_current_node()
        batch_size, num_steps = last_node.shape

        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if state.i.numpy()[0] == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                # B x 1 x 2H
                return tf.broadcast_to(self.W_placeholder[None, None, :], 
                                       shape=[batch_size, 1, self.W_placeholder.shape[-1]])
            else:
                return tf.concat(
                    [
                        tf.gather(node_embedding,state.first_node,batch_dims=1), 
                        tf.gather(node_embedding,last_node,batch_dims=1)  
                    ],
                    axis=-1
                )
                
                # print('$'*20)
                # node_embedding = torch.from_numpy(node_embedding.numpy())
                # f = torch.from_numpy(state.first_node.numpy())
                # l = torch.from_numpy(state.last_node.numpy())
                # GG = node_embedding\
                # .gather(
                #     1,
                #     torch.cat((f, l), 1)[:, :, None].expand(batch_size, 2, node_embedding.size(-1))
                # ).view(batch_size, 1, -1)
                # print(GG)
                # print('$'*20)
                # ##############################################
                # # PENDING
                # ##############################################
                # pass

    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, graph_mask=None):
        batch_size, num_steps, embed_dim = query.shape
        query_size = key_size = val_size = embed_dim // self.n_heads

        # M x B x N x 1 x (H/M)
        # Compute the glimpse, rearrange dimensions to (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = tf.transpose(
            tf.reshape(
                query, # B x 1 x H
                shape=[batch_size, num_steps, self.n_heads, 1, query_size]
            ),
            perm=[2, 0, 1, 3, 4]
        )

        # [M x B x N x 1 x (H/M)] X [M x B x N x (H/M) x V] = [M x B x N x 1 x V]
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = tf.matmul(glimpse_Q, tf.transpose(glimpse_K, [0,1,2,4,3])) / math.sqrt(query_size)
        
        mask_temp = tf.cast(tf.broadcast_to(mask[None, :, :, None, :], shape=compatibility.shape), dtype=tf.double)
        compatibility = tf.cast(compatibility, dtype=tf.double) + (mask_temp * -1e9)

        graph_mask_temp = tf.cast(tf.broadcast_to(graph_mask[None, :, :, None, :], shape=compatibility.shape), dtype=tf.double)
        compatibility = tf.cast(compatibility, dtype=tf.double) + (graph_mask_temp * -1e9)

        compatibility = tf.cast(compatibility, dtype=tf.float32)        

        # compatibility[tf.broadcast_to(mask[None, :, :, None, :], shape=compatibility.shape)] = -1e10
        # compatibility[tf.broadcast_to(graph_mask[None, :, :, None, :], shape=compatibility.shape)] = -1e10

        # attention weights a(c,j): 
        attention_weights = tf.nn.softmax(compatibility, axis=-1)


        # [M x B x N x 1 x V] x [M x B x N x V x (H/M)] = [M x B x N x 1 x (H/M)]
        heads = tf.matmul(attention_weights, glimpse_V)
       
        # B x N x 1 x H
        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            tf.reshape(
                tf.transpose(heads, perm=[1, 2, 3, 0, 4]),
                shape=[batch_size, num_steps, 1, self.n_heads*val_size]
            )
        )

        # B x N x 1 x H
        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse


        # [B x N x 1 x H] x [B x 1 x H x V] = [B x N x 1 x V] --> [B x N x V] (Squeeze) 
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = tf.squeeze(tf.matmul(final_Q, tf.transpose(logit_K, perm=[0,1,3,2])),
                            axis=-2) / math.sqrt(final_Q.shape[-1])

        logits = logits + ( tf.cast(graph_mask, dtype=tf.float32) * -1e9)
        logits = tf.math.tanh(logits) * self.tanh_clipping
        logits = logits + ( tf.cast(mask, dtype=tf.float32) * -1e9)

        # logits[graph_mask] = -1e10 
        # logits = torch.tanh(logits) * self.tanh_clipping
        # logits[mask] = -1e10
        
        return logits, tf.squeeze(glimpse, axis=-2)

    
    def _select_node(self, probs, mask):
        assert tf.reduce_all(probs == probs) == True, "Probs should not contain any nans"

        if self.decode_type == "greedy":
            selected = tf.math.argmax(probs, axis=1)
            assert not tf.reduce_any(tf.cast(tf.gather_nd(mask, tf.expand_dims(selected, axis=-1), batch_dims=1), dtype=tf.bool)), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            dist = tfp.distributions.Multinomial(total_count=1, probs=probs)
            selected = tf.argmax(dist.sample(), axis=1)

            # Check if sampling went OK
            while tf.reduce_any(tf.cast(tf.gather_nd(mask, tf.expand_dims(selected, axis=-1), batch_dims=1), dtype=tf.bool)):
                print('Sampled bad values, resampling!')
                selected = tf.argmax(dist.sample(), axis=1)

        else:
            assert False, "Unknown decode type"
        return selected

    
    def _calc_log_likelihood(self, _log_p, a, mask):
        
        # Get log_p corresponding to selected actions
        batch_size, steps_count = a.shape
        indices = tf.concat([
        tf.expand_dims(tf.broadcast_to(tf.range(steps_count, dtype=tf.int64), shape=a.shape), axis=-1),
        tf.expand_dims(a, axis=-1)],
        axis=-1
        )
        log_p = tf.gather_nd(_log_p, indices, batch_dims=1)
        

        # _log_p = torch.from_numpy(_log_p.numpy())
        # a = torch.from_numpy(a.numpy())
        # AA = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        # print(AA)
        # print(log_p)
        # print('DONE')
        

        # Get log_p corresponding to selected actions
        # log_p = tf.gather(tf.squeeze(_log_p,axis=-2), a, batch_dims=1) #_log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        # Why??????
        # assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return tf.reduce_sum(log_p, axis=1) # log_p.sum(1)

       

