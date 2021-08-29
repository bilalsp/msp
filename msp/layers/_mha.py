"""
The :mod:`mps.layers._mha` module defines `Multi-Head Attention` layer
inherited from tensorflow layer. Required precomputed keys, values, and query 
for efficiency.
"""
import tensorflow as tf
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Layer


class MHALayer(Layer):

    def __init__(self, 
                 n_heads,
                 key_dim,
                 value_dim=None,
                 dropout=0.0,
                 use_bias=False,
                 **kwargs):
        """Multi-Head Attention Layer."""
        super(MHALayer, self).__init__(**kwargs)
        self._n_heads = n_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias


    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        keys_shape = input_shape[1]
        batch_size, _, num_nodes, embed_dim = keys_shape

        W_out_shape = tf.TensorShape((None, 1, embed_dim))

        with tf.name_scope('W_output'):
            self.W_out = tf.keras.layers.Dense(embed_dim, use_bias=self._use_bias)
            self.W_out.build(W_out_shape)

        super().build(input_shape)
        

    def call(self, inputs, training=None):
        """ """
        # projected query, keys, and values
        query, keys, values, mask = inputs

        batch_size = tf.shape(keys)[0]
        num_node = tf.shape(keys)[1]
        embed_dim = tf.shape(keys)[-1]

        # M x B x 1 x (H/M)
        q = self._make_heads(query, self._key_dim)

        # M x B x V x (H/M)
        K = self._make_heads(tf.squeeze(keys, axis=1), self._key_dim)
        V = self._make_heads(tf.squeeze(values, axis=1), self._value_dim)

        # [M x B x 1 x (H/M)] X [M x B x (H/M) x V] = [M x B x 1 x V]
        # Batch matrix multiplication to compute compatibilities
        attn_weights  = tf.divide(
            tf.matmul(q, tf.transpose(K, [0,1,3,2])),
            self._key_dim**0.5
        )  

        if mask is not None:
            # B x 1 x V
            mask = tf.broadcast_to(mask, shape=tf.shape(attn_weights))
            attn_weights = attn_weights + mask

        # M x B x 1 x V
        attn_weights = softmax(attn_weights, axis=-1)   

        # [M x B x 1 x V] x [M x B x V x (H/M)] = [M x B x 1 x (H/M)]
        heads = tf.matmul(attn_weights, V)

        # B x 1 x H
        outputs = self.W_out(
            tf.reshape(
                tf.transpose(heads, perm=[1, 2, 0, 3]) ,
                shape=(batch_size, 1, self._n_heads*self._value_dim)
            )
        )
        return outputs

    def _make_heads(self, v, dim):
        """ """
        # in case of query num_nodes=1 
        batch_size = tf.shape(v)[0]
        num_nodes = tf.shape(v)[1]
        embed_dims = tf.shape(v)[2]

        # M x B x N x V x (H/M)
        return tf.transpose(
            tf.reshape(
                v,  # B x V x H
                shape=[batch_size, num_nodes, self._n_heads, dim]
            ),
            perm=[2, 0, 1, 3]    
        )