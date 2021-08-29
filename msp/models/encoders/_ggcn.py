"""
The :mod:`mps.models._ggcn` module defines encoder architecture based
on `Gated Graph Convolution` layer.
"""
import tensorflow as tf
from tensorflow.keras import Model

from msp.graphs import MSPEmbedGraph
from msp.layers import GGCNLayer


class GGCNEncoder(Model):

    def __init__(self, 
                 units,
                 layers,
                 *args, 
                 activation='relu',
                 **kwargs):
        """ """
        super(GGCNEncoder, self).__init__(*args,  **kwargs)
        self.initial_layer_1 = tf.keras.layers.Dense(
            units, name='initial_layer_for_node_features')
        self.initial_layer_2 = tf.keras.layers.Dense(
            units, name='initial_layer_for_edge_features')
        self.ggcn_layers = [GGCNLayer(units, activation=activation)
                            for _ in range(layers)]
        
    def call(self, inputs, training=None):
        """ """
        node_features_t = self.initial_layer_1(inputs.node_features)
        edge_features_t = self.initial_layer_2(inputs.edge_features)
        
        outputs = MSPEmbedGraph(**inputs._asdict(), 
                                node_embed=node_features_t,
                                edge_embed=edge_features_t)

        for layer in self.ggcn_layers:
            outputs = layer(outputs, training)

        return outputs