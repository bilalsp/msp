"""
Tests for gated graph convolution layer.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl.testing import parameterized
from keras import keras_parameterized
import tensorflow as tf

from msp.layers import GGCNLayer
from msp.graphs import MSPEmbedGraph


class GGCNLayerTest(keras_parameterized.TestCase):

    @parameterized.named_parameters(
        (
            'trainable_variables_norm_batch',
            tf.TensorShape(dims=(None, 30, 30)),
            tf.TensorShape(dims=(None, 30, 5)),
            tf.TensorShape(dims=(None, 30, 30, 3)),
            tf.TensorShape(dims=(None, 30, 10)),
            tf.TensorShape(dims=(None, 30, 32)),
            tf.TensorShape(dims=(None, 30, 30, 32)),
            64,
            {'use_bias': False}
        ),
        (
            'trainable_variables_norm_layer',
            tf.TensorShape(dims=(None, 50, 50)),
            tf.TensorShape(dims=(None, 50, 5)),
            tf.TensorShape(dims=(None, 50, 50, 3)),
            tf.TensorShape(dims=(None, 50, 20)),
            tf.TensorShape(dims=(None, 50, 128)),
            tf.TensorShape(dims=(None, 50, 50, 128)),
            128,
            {'normalization': 'layer'}
        )
    )
    def test_trainable_variables(self, adj_matrix_shape, node_features_shape, 
                                 edge_features_shape, job_assignment_shape, 
                                 node_embed_shape, edge_embed_shape, hidden_dim, 
                                 kwargs):
        """ """
        ggcn_layer = GGCNLayer(hidden_dim, **kwargs)

        input_shape = MSPEmbedGraph(
            adj_matrix=adj_matrix_shape,
            node_features=node_features_shape,
            edge_features=edge_features_shape,
            job_assignment=job_assignment_shape,
            node_embed=node_embed_shape,
            edge_embed=edge_embed_shape
        )

        ggcn_layer.build(input_shape)

        trainable_vars = ggcn_layer.trainable_variables
        all_vars = ggcn_layer.variables

        # batch normalization has `moving_mean` and `moving_variance` untrainable 
        # variables for node and edge
        if kwargs.get('normalization', 'batch') == 'batch':
            self.assertEqual(len(all_vars), len(trainable_vars) + 4)
        else:
            self.assertEqual(len(all_vars), len(trainable_vars))
        
        vars_shape = {
            'node_kernel': (node_embed_shape[-1], hidden_dim),
            'edge_kernel': (edge_embed_shape[-1], hidden_dim),
            'node_norm': (node_embed_shape[-1],),
            'edge_norm': (edge_embed_shape[-1],),
            'bias': (hidden_dim,)
        }

        for var in trainable_vars:
            
            get_var_name = lambda: (
                (var.name.split('/')[0] + '_' if 'bias' not in var.name else '') + \
                (var.name.split('/')[-1][:-2] if 'norm' not in var.name else 'norm')
            )

            self.assertTupleEqual(tuple(var.shape), vars_shape[get_var_name()])
