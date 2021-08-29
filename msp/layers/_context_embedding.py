"""
The :mod:`mps.layers._context_embedding` module defines ContextEmbedding layer
inherited from tensorflow layer.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer


class ContextEmbedding(Layer):

    def __init__(self, units, use_bias=False, **kwargs):
        """ """
        super(ContextEmbedding, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        with tf.name_scope('W_placeholder'):
            # learnable first and last node at timestep t=1
            self.W_context_placeholder = self.add_weight(
                shape=(2*self.units,),
                initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                trainable=True,
                name='W_context_placeholder'
            )
        
        with tf.name_scope('W_step_context'):
            # projections for first and last node
            W_step_context_shape = tf.TensorShape((None, None, 2*self.units))
            self.W_step_context = tf.keras.layers.Dense(
                self.units, use_bias=self.use_bias)
            self.W_step_context.build(W_step_context_shape)

        super().build(input_shape)


    def call(self, inputs):
        """ """
        node_embed = inputs[0].node_embed
        fixed_context = inputs[1]
        state = inputs[2]
        
        batch_size = tf.shape(node_embed)[0]
        
        # first and last node embedding
        first_step_first_last_embed = lambda : tf.broadcast_to(
            self.W_context_placeholder[None, None, :],
            shape=(batch_size, 1, tf.shape(self.W_context_placeholder)[0])
        )
        other_step_first_last_embed = lambda : tf.concat(
            [
                tf.gather(node_embed, state.first_node, batch_dims=1), 
                tf.gather(node_embed, state.last_node, batch_dims=1)      
            ],
            axis=-1
        )

        is_first_step = tf.equal(state.get_step_count(), 1)
        
        # B x 1 x 2H
        first_last_embed = tf.cond(is_first_step, 
                                   first_step_first_last_embed, 
                                   other_step_first_last_embed)
        
        # B x 1 x H
        context_embed = fixed_context + self.W_step_context(first_last_embed)
        return context_embed

