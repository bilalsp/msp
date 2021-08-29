"""
The :mod:`mps.layers._sha` module defines `Single-Head Attention` layer
inherited from tensorflow layer. Required precomputed keys and query. Note that
it just return log-probabilites.
"""
import tensorflow as tf
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Layer


class SHALayer(Layer):

    def __init__(self, 
                 key_dim,
                 tanh_clipping=10,
                 use_bias=False,
                 **kwargs):
        """Single-Head Attention Layer."""
        super(SHALayer, self).__init__(**kwargs)
        self._key_dim = key_dim
        self.tanh_clipping = tanh_clipping
        self._use_bias = use_bias

    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        super().build(input_shape)

    def call(self, inputs, training=None):
        # projected query, and keys
        query, keys, mask = inputs
        batch_size, _, num_nodes, embed_dim = keys.shape

        # [B x 1 x H] X [B x H x V] = [B x 1 x V]
        # Batch matrix multiplication to compute logits
        logits = tf.divide(
            tf.matmul(query, tf.transpose(tf.squeeze(keys, axis=1), [0,2,1])),
            self._key_dim**0.5
        ) 
        logits = tf.math.tanh(logits) * self.tanh_clipping + mask

        # log probabilities
        log_p = tf.nn.log_softmax(logits, axis=-1)

        return log_p