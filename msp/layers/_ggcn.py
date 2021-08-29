"""
The :mod:`mps.layers._ggcn` module defines `Gated Graph Convolution Net` layer
inherited from tensorflow layer.

Reference:
    - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. 
    `Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020`.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer

from msp.graphs import MSPEmbedGraph


class GGCNLayer(Layer):

    def __init__(self, 
                 units,
                 *args, 
                 activation='relu', 
                 use_bias=True, 
                 normalization='batch',
                 aggregation='mean',
                 **kwargs):
        """Gated Graph Convolution Layer.
        
        Args:
            units: Number of hidden dimensions.
            activation: Activation function to use.
            use_bise: Boolean, whether the layer uses a bias vector.
            normalization: Type of normalization `batch` or `layer`.
            aggregation: Method to aggreate the messaages from neighbor nodes.

        Input shape:
            Instance of `MSPEmbedGraph` consisting of atleast following tensor:
            Tensor `adj_matrix` with shape: `(batch_size, num_node, num_node)`.
            Tensor `node_features` with shape: `(batch_size, num_node, input_dim)`.
            Tensor `edge_features` with shape: `(batch_size, num_node, num_node, input_dim)`.


        Output shape:
            Instance of `MSPEmbedGraph` consisting of atleast following tensor:
            Tensor `adj_matrix` with shape: `(batch_size, num_node, num_node)`.
            Tensor `node_features` with shape: `(batch_size, num_node, units)`.
            Tensor `edge_features` with shape: `(batch_size, num_node, num_node, units)`.
        """
        super(GGCNLayer, self).__init__(*args, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.normalization= normalization
        self.aggregation = aggregation

    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        node_embed_shape = input_shape.node_embed
        edge_embed_shape = input_shape.edge_embed

        with tf.name_scope('node'):
            with tf.name_scope('U'):
                self.U = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.U.build(node_embed_shape)

            with tf.name_scope('V'):
                self.V = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.V.build(node_embed_shape)

            with tf.name_scope('norm'):
                self.norm_h = {
                    "batch": tf.keras.layers.BatchNormalization(),
                    "layer": tf.keras.layers.LayerNormalization()
                }.get(self.normalization, None)
                if self.norm_h:
                    self.norm_h.build(node_embed_shape)

        with tf.name_scope('edge'):
            with tf.name_scope('A'):
                self.A = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.A.build(edge_embed_shape)
            
            with tf.name_scope('B'):
                self.B = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.B.build(node_embed_shape)

            with tf.name_scope('C'):
                self.C = tf.keras.layers.Dense(self.units, use_bias=self.use_bias)
                self.C.build(node_embed_shape)

            with tf.name_scope('norm'):
                self.norm_e = {
                    'batch': tf.keras.layers.BatchNormalization(),
                    'layer': tf.keras.layers.LayerNormalization(axis=-1)
                }.get(self.normalization, None)
                if self.norm_e:
                    self.norm_e.build(edge_embed_shape)
    
        super().build(input_shape)

    def call(self, inputs, training=None):
        """ """
        adj_matrix = inputs.adj_matrix
        h = inputs.node_embed
        e = inputs.edge_embed

        # Edges Featuers
        Ae = self.A(e)
        Bh = self.B(h)
        Ch = self.C(h)

        e = self._update_edges(e, [Ae, Bh, Ch], training)
        
        edge_gates = tf.sigmoid(e)

        # Nodes Features
        Uh = self.U(h)
        Vh = self.V(h)

        h = self._update_nodes(
            h,
            [Uh, self._aggregate(Vh, edge_gates, adj_matrix)],
            training
        )

        outputs = MSPEmbedGraph(
            *(   
                inputs.adj_matrix,
                inputs.node_features,
                inputs.edge_features,
                inputs.job_assignment
            ),
            node_embed = h,
            edge_embed = e
        )

        return outputs
        
    def _update_edges(self, e, transformations:list, training):
        """Update edges features"""
        Ae, Bh, Ch  = transformations
        e_new = Ae + tf.expand_dims(Bh, axis=1) + tf.expand_dims(Ch, axis=2)

        # normalization
        if self.norm_e:
            e_new = self.norm_e(e_new, training)
        
        # activation
        e_new = self.activation(e_new)

        # skip/residual Connection
        e_new = e + e_new    
        return e_new

    def _update_nodes(self, h, transformations:list, training):
        """Update node feature."""
        Uh, aggregated_messages = transformations
        h_new = tf.math.add_n([Uh, aggregated_messages])
        
        # Normalization
        if self.norm_h:
            h_new = self.norm_h(h_new, training)
    
        # Activation
        h_new = self.activation(h_new)
        
        # Skip/residual Connection
        h_new = h + h_new   
        return h_new

    def _aggregate(self, Vh, edge_gates, adj_matrix):
        """Aggregate neighbors messages."""
        # Reshape as edge_gates
        Vh = tf.broadcast_to(tf.expand_dims(Vh, axis=1), tf.shape(edge_gates))
        #Vh = tf.broadcast_to(tf.expand_dims(Vh, axis=1), edge_gates.shape)
        
        # Gating mechanism
        Vh = edge_gates * Vh    
        
        # Apply graph structure
        neighbor_mask = tf.broadcast_to(tf.expand_dims(adj_matrix, axis=-1), tf.shape(Vh))
        # neighbor_mask = tf.broadcast_to(tf.expand_dims(adj_matrix, axis=-1), Vh.shape)
        Vh = Vh * neighbor_mask

        # message aggregation
        if self.aggregation == 'mean':           
            return tf.divide(
                tf.math.reduce_sum(Vh, axis=2), 
                tf.math.reduce_sum(neighbor_mask, axis=2))

        elif self.aggregation == 'sum':
            return tf.math.reduce_sum(Vh, axis=2)
        
        else:
            return tf.reduce_max(Vh, axis=2)
