"""
Generate samples of synthetic data sets.
"""
import itertools
import functools

import numpy as np
import tensorflow as tf

from ..utils import _is_connected, check_random_state
from ..graphs import MSPGraph
from ..graphs import MSPSparseGraph

def make_data(n_samples, node_feat_size=5, edge_feat_size=3, msp_size=None,
              msp_rand_size=None, random_state=None) -> tf.data.Dataset:
    """Generate a random MSP dataset.
     
    Parameters
    ----------
    n_samples : int
        The number of samples.

    node_feat_size : int,  default=5
        The number of features for each node.

    edge_feat_size : int, default=3
        The number of features for each edge.

    msp_size : tuple, default=None
        The tuple of (n_machines, n_jobs) where n_machines is number of machines,
        and n_jobs is the number of jobs.

    msp_rand_size: tuple, default=None
        The random number of nodes for each sample in a generated dataset.

    random_state : int, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output.

    Returns
    -------
    dataset : tf.data.Dataset
    """

    rand_gen = check_random_state(random_state)

    def _generator(n_samples):

        for _ in range(n_samples):
            
            if msp_size is not None:
                n_machines, n_jobs = msp_size
            elif msp_rand_size is not None:
                n_machines, n_jobs = rand_gen.randint(*msp_rand_size, size=2)
            else:
                n_machines, n_jobs = rand_gen.randint(*(1,1000), size=2)

            # random node features: [n_nodes, node_feat_size]
            node_features_shape = (n_jobs+n_machines, node_feat_size)
            node_features = rand_gen.rand(*node_features_shape)
            node_features[:,-1] = 1
            node_features[n_jobs:,:] = 0

            # random job allocation: [n_nodes, n_machines]
            alpha_shape = (n_jobs+n_machines, n_machines)
            alpha = rand_gen.choice([0, 1], size=alpha_shape, p=[2./5, 3./5])
            alpha = alpha + np.logical_not(alpha.sum(axis=1))\
                              .reshape((-1,1)).astype('int')
            alpha[n_jobs:,:] = 0

            # edge index: [2, n_edges]
            edge_index = np.array(list(
                filter(
                    functools.partial(_is_connected, node_features, alpha),
                    itertools.combinations(range(n_jobs+n_machines), 2)
                )
            )).T

            # edge features: [n_edges, edge_feat_size]
            edge_features_shape = (edge_index.shape[1], edge_feat_size)
            edge_features = rand_gen.rand(*edge_features_shape)
            edge_type = np.logical_xor(
                list(map(lambda edge: node_features[edge[0]][-1], edge_index.T)),
                list(map(lambda edge: node_features[edge[1]][-1], edge_index.T))
            ).reshape((-1,1))
            edge_features = edge_features * np.invert(edge_type).astype('int')
            edge_features[:,-1:] = edge_type         
            
            node_features = tf.convert_to_tensor(node_features)
            edge_features = tf.convert_to_tensor(edge_features)
            edge_index = tf.convert_to_tensor(edge_index)
            alpha = tf.convert_to_tensor(alpha)

            msp = MSPGraph(node_features, edge_features, edge_index, alpha)
            
            yield msp.to_dict()

    output_signature = {
        'node_features': tf.TensorSpec(shape=None, dtype=tf.float64),
        'edge_features': tf.TensorSpec(shape=None, dtype=tf.float64),
        'edge_index': tf.TensorSpec(shape=None, dtype=tf.float64),
        'alpha': tf.TensorSpec(shape=None, dtype=tf.float64)
    }

    dataset = tf.data.Dataset.from_generator(_generator, args=(n_samples,),
                                            output_signature=output_signature)
    return dataset



def make_sparse_data(n_samples, node_feat_size=5, edge_feat_size=3, msp_size=None,
              msp_rand_size=None, random_state=None) -> tf.data.Dataset:
    """Generate a random MSP dataset.
     
    """

    rand_gen = check_random_state(random_state)

    def sparse_repr(graph):
        edge_index = tf.cast(tf.transpose(
            tf.concat([
                graph['edge_index'],
                tf.scatter_nd(
                    tf.constant([[1],[0]]),
                    graph['edge_index'],
                    graph['edge_index'].shape
                )
            ], axis=-1)
        ), tf.int32)

        edge_features = tf.cast(
            tf.tile(
                graph['edge_features'], 
                tf.constant([2,1], tf.int32)
            ), 
        tf.float32)
      
        num_nodes = graph['alpha'].shape[0]
        num_edge_features = edge_features.shape[-1]
        adjMatrix_shape = (num_nodes, num_nodes, num_edge_features)  

        adjMatrix = tf.tensor_scatter_nd_update(
                tf.zeros(adjMatrix_shape),
                edge_index, 
                edge_features  
        )
        return adjMatrix


    def _generator(n_samples):

        for _ in range(n_samples):
            if msp_size is not None:
                n_machines, n_jobs = msp_size
            elif msp_rand_size is not None:
                n_machines, n_jobs = rand_gen.randint(*msp_rand_size, size=2)
            else:
                n_machines, n_jobs = rand_gen.randint(*(1,1000), size=2)

            # random node features: [n_nodes, node_feat_size]
            node_features_shape = (n_jobs+n_machines, node_feat_size)
            node_features = rand_gen.rand(*node_features_shape)
            node_features[:,-1] = 1
            node_features[n_jobs:,:] = 0

            # random job allocation: [n_nodes, n_machines]
            alpha_shape = (n_jobs+n_machines, n_machines)
            alpha = rand_gen.choice([0, 1], size=alpha_shape, p=[2./5, 3./5])
            alpha = alpha + np.logical_not(alpha.sum(axis=1))\
                              .reshape((-1,1)).astype('int')
            alpha[n_jobs:,:] = 0

            # edge index: [2, n_edges]
            edge_index = np.array(list(
                filter(
                    functools.partial(_is_connected, node_features, alpha),
                    itertools.combinations(range(n_jobs+n_machines), 2)
                )
            )).T

            # edge features: [n_edges, edge_feat_size]
            edge_features_shape = (edge_index.shape[1], edge_feat_size)
            edge_features = rand_gen.rand(*edge_features_shape)
            edge_type = np.logical_xor(
                list(map(lambda edge: node_features[edge[0]][-1], edge_index.T)),
                list(map(lambda edge: node_features[edge[1]][-1], edge_index.T))
            ).reshape((-1,1))
            edge_features = edge_features * np.invert(edge_type).astype('int')
            edge_features[:,-1:] = edge_type         
             
            node_features = tf.convert_to_tensor(node_features)
            edge_features = tf.convert_to_tensor(edge_features)
            edge_index = tf.convert_to_tensor(edge_index)
            alpha = tf.convert_to_tensor(alpha)

            msp = MSPGraph(node_features, edge_features, edge_index, alpha)
            edge_features = sparse_repr(msp.to_dict())

            adj_matrix = tf.sparse.SparseTensor(
                indices= tf.cast(tf.transpose(
                    tf.concat([
                        edge_index,
                        tf.scatter_nd(
                            tf.constant([[1],[0]]),
                            edge_index,
                            edge_index.shape
                        )
                    ], axis=-1)
                ), tf.int64),
                values = tf.ones([2*edge_index.shape[-1]]),
                dense_shape = [node_features_shape[0], node_features_shape[0]]
            )
            adj_matrix = tf.sparse.to_dense(tf.sparse.reorder(adj_matrix))

            yield MSPSparseGraph(adj_matrix, node_features, edge_features, alpha)
    
    output_signature = MSPSparseGraph(
        adj_matrix = tf.TensorSpec(shape=None, dtype=tf.float32), 
        node_features = tf.TensorSpec(shape=None, dtype=tf.float64), 
        edge_features = tf.TensorSpec(shape=None, dtype=tf.float64), 
        alpha = tf.TensorSpec(shape=None, dtype=tf.float64)
    )

    dataset = tf.data.Dataset.from_generator(_generator, args=(n_samples,),
                                            output_signature=output_signature)
    return dataset


