"""

"""
import tensorflow as tf
from tensorflow.keras import Model

from msp.layers import GGCNLayer
from msp.graphs import MSPSparseGraph


class GGCNEncoder(Model):

    def __init__(self, 
                 units,
                 layers,
                 *args, 
                 activation='relu',
                 **kwargs):
        """ """
        super(GGCNEncoder, self).__init__()
        self.initial_layer_1 = tf.keras.layers.Dense(units)
        self.initial_layer_2 = tf.keras.layers.Dense(units)
        self.ggcn_layers = [GGCNLayer(units, activation=activation)
                            for _ in range(layers)]
        
    def call(self, inputs, training=False):
        """ """
        node_features_t = self.initial_layer_1(inputs.node_features)
        edge_features_t = self.initial_layer_2(inputs.edge_features)
        
        outputs = MSPSparseGraph(inputs.adj_matrix, 
                                 node_features_t, 
                                 edge_features_t,
                                 inputs.alpha)

        for layer in self.ggcn_layers:
            outputs = layer(outputs)

        return outputs
