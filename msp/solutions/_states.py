"""

"""
import tensorflow as tf


class MSPState:

    def __init__(self, inputs):
        """ """
        self.adj_matrix = inputs.adj_matrix
        self.edge_features = inputs.edge_features
        self.node_features = inputs.node_features
        self.node_embed = inputs.node_embed
        
        batch_size, num_nodes, node_embed_dims = self.node_embed.shape
        
        self._first_node = tf.zeros((batch_size,1), dtype=tf.int64)
        self._last_node = self._first_node
        self._visited = tf.zeros((batch_size,1,num_nodes), dtype=tf.uint8)
        self._makespan = tf.zeros((batch_size,1))
    
        self.i = tf.zeros(1, dtype=tf.int64) # # Vector with length num_steps
        self.ids = tf.range(5, dtype=tf.int64)[:, None] #  # Add steps dimension
        #self._step_num = tf.zeros(1, dtype=tf.int64)


    @property
    def first_node(self):
        return self._first_node

    @first_node.setter
    def first_node(self, first_node):
        self._first_node = first_node

    @property
    def last_node(self):
        return self._last_node

    @last_node.setter
    def last_node(self, last_node):
        self._last_node = last_node


    @property
    def makespan(self):
        return self._makespan

    def all_finished(self):
        # Exactly n steps
        return self.i.numpy()[0] >= self.node_embed.shape[-2]

    def get_current_node(self):
        return self._last_node

    def get_mask(self):
        return self._visited

    def get_graph_mask(self):
        batch_size, num_nodes, _ = self.node_embed.shape
        if self.i.numpy()[0] == 0:
            return tf.zeros((batch_size,1,num_nodes), dtype=tf.bool)
        else:
            return ~tf.cast(
                tf.gather_nd(self.adj_matrix, 
                             indices=tf.expand_dims(self.last_node, axis=-1), 
                             batch_dims=1),
                dtype=tf.bool
            )

    def update(self, selected):

        # update state
        cur_node = selected[:, tf.newaxis] 
        
        if self.i.numpy()[0] == 0:
            setup_time = tf.zeros(cur_node.shape)
        else:
            setup_time = tf.gather_nd(
                self.edge_features,
                tf.concat([self.last_node, cur_node], axis=1),
                batch_dims=1
            )[:,0:1]

        processing_time = tf.gather_nd(
            self.node_features,
            cur_node,
            batch_dims=1
        )[:, 0:1]

        self._makespan = self._makespan + setup_time + processing_time

        self._first_node = cur_node if self.i.numpy()[0] == 0 else self._first_node

        batch_size, _, _ = self._visited.shape
        self._visited = tf.tensor_scatter_nd_update(
            tf.squeeze(self._visited, axis=-2),
            tf.concat([tf.reshape(tf.range(batch_size, dtype=tf.int64),cur_node.shape), cur_node], axis=1),
            tf.ones((batch_size,), dtype=self._visited.dtype)
        )[:,tf.newaxis,:]

        self.i = self.i + 1




