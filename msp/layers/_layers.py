"""

"""
import tensorflow as tf
from tensorflow.keras.layers import Layer

from msp.graphs import MSPSparseGraph


class GGCNLayer(Layer):

    # @validate_hyperparams
    def __init__(self, 
                 units,
                 *args, 
                 activation='relu', 
                 use_bias=True, 
                 normalization='batch',
                 aggregation='mean',
                 **kwargs):
        """ """
        super(GGCNLayer, self).__init__(*args, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.normalization= normalization
        self.aggregation = aggregation

    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        node_features_shape = input_shape.node_features
        edge_featues_shape = input_shape.edge_features
        embedded_shape = tf.TensorShape((None, self.units))

        with tf.name_scope('node'):
            with tf.name_scope('U'):
                self.U = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.U.build(node_features_shape)

            with tf.name_scope('V'):
                self.V = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.V.build(node_features_shape)

            with tf.name_scope('norm'):
                self.norm_h = {
                    "batch": tf.keras.layers.BatchNormalization(),
                    "layer": tf.keras.layers.LayerNormalization()
                }.get(self.normalization, None)
                if self.norm_h:
                    self.norm_h.build(embedded_shape)

        with tf.name_scope('edge'):
            with tf.name_scope('A'):
                self.A = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.A.build(tf.TensorShape((None, node_features_shape[-1])))
            
            with tf.name_scope('B'):
                self.B = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.B.build(node_features_shape)

            with tf.name_scope('C'):
                self.C = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.C.build(edge_featues_shape)

            with tf.name_scope('norm'):
                self.norm_e = {
                    'batch': tf.keras.layers.BatchNormalization(),
                    'layer': tf.keras.layers.LayerNormalization(axis=-2)
                }.get(self.normalization, None)
                if self.norm_e:
                    self.norm_e.build(embedded_shape)
    
        super().build(input_shape)
 
    def call(self, inputs):
        """ """
        adj_matrix = inputs.adj_matrix
        h = inputs.node_features
        e = inputs.edge_features

        # Edges Featuers
        Ah = self.A(h)
        Bh = self.B(h)
        Ce = self.C(e)
        e = self._update_edges(e, [Ah, Bh, Ce])

        edge_gates = tf.sigmoid(e)

        # Nodes Features
        Uh = self.U(h)
        Vh = self.V(h)
        h = self._update_nodes(
            h,
            [Uh, self._aggregate(Vh, edge_gates, adj_matrix)]
        )

        outputs = MSPSparseGraph(adj_matrix, h, e, inputs.alpha)
        return outputs
        
    def _update_edges(self, e, transformations:list):
        """Update edges features"""
        Ah, Bh, Ce  = transformations
        e_new = tf.expand_dims(Ah, axis=1) + tf.expand_dims(Bh, axis=2) + Ce
        # Normalization
        batch_size, num_nodes, num_nodes, hidden_dim = e_new.shape
        if self.norm_e:
            e_new = tf.reshape(
                self.norm_e(
                    tf.reshape(e_new, [batch_size*num_nodes*num_nodes, hidden_dim])
                ), e_new.shape
            )
        # Activation
        e_new = self.activation(e_new)
        # Skip/residual Connection
        e_new = e + e_new     # (---------------------Think about Add Layer-------------------------------)
        return e_new

    def _update_nodes(self, h, transformations:list):
        """ """
        Uh, aggregated_messages = transformations
        h_new = tf.math.add_n([Uh, aggregated_messages])
        # Normalization
        batch_size, num_nodes, hidden_dim = h_new.shape
        if self.norm_h:
            h_new = tf.reshape(
                self.norm_h(
                    tf.reshape(h_new, [batch_size*num_nodes, hidden_dim])
                ), h_new.shape
            )
        # Activation
        h_new = self.activation(h_new)
        # Skip/residual Connection
        h_new = h + h_new       # (---------------------Think about Add Layer-------------------------------)
        return h_new

    def _aggregate(self, Vh, edge_gates, adj_matrix):
        """ """
        # Reshape as edge_gates
        Vh = tf.broadcast_to(
            tf.expand_dims(Vh, axis=1),
            edge_gates.shape
        )
        # Gating mechanism
        Vh = edge_gates * Vh
        
        # Apply graph structure      #----------------------Fix it-------------------------------
        # mask = tf.broadcast_to(tf.expand_dims(adj_matrix,axis=-1), Vh.shape)
        # Vh[~mask] = 0

        # message aggregation
        if self.aggregation == 'mean':
            total_messages = tf.cast(
                tf.expand_dims(
                    tf.math.reduce_sum(adj_matrix, axis=-1),
                    axis=-1
                ),
                Vh.dtype
            )
            return tf.math.reduce_sum(Vh, axis=2) / total_messages
        
        elif self.aggregation == 'sum':
            return tf.math.reduce_sum(Vh, axis=2)
