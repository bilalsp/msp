"""
The :mod:`mps.models._models` module defines encoder-decoder architecture to 
solve MSP problem.
"""
import tensorflow as tf
from tensorflow.keras import Model


class MSPModel(Model):

    def __init__(self, 
                 encoder_class,
                 encoder_params,
                 decoder_class,
                 decoder_params,
                 *args, 
                 **kwargs):
        """ """
        super(MSPModel, self).__init__(*args,  **kwargs)

        self.encoder = encoder_class(
            encoder_params.get('units', 128),
            encoder_params.get('layers', 3),
            name='ggcn_encoder'
        )

        self.decoder = decoder_class(
            decoder_params.get('units', 128),
            use_bias = decoder_params.get('use_bias', False),
            n_heads = decoder_params.get('n_heads', 8),
            aggregation_graph = decoder_params.get('aggregation_graph', 'mean'),
            tanh_clipping = decoder_params.get('tanh_clipping', 10),
            name='attention_decoder'        # **kwargs <---- use pop instead of get 
        )

    @tf.function
    def call(self, inputs, training=None):
        """ """
        outputs = self.encoder(inputs, training)
        outputs = self.decoder(outputs, training)
        return outputs