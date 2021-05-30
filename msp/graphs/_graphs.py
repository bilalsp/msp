"""
Create MSPGraph using node_features, edge_features, edge_index, alpha matrices
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.python.framework import dtypes


class MSPGraph(nx.Graph):

    def __init__(self, node_features, edge_features, edge_index, alpha, *args, 
                 **kwargs):
        """MSP(Machine Scheduling Problem) Graph.

        Parameters
        ----------
        node_features : ndarray of shape (n_nodes,node_feat_size), dtype=float
            Features for each node in a graph.
        
        edge_features : ndarray of shape (n_edges, edge_feat_size), dtype=float
            Features for each edge in a graph.

        edge_index:  ndarray of shape (2, n_edges), dtype=int
            Graph connectivity 
        
        alpha: ndarray of shape (n_nodes,n_machines), dtype=int
            Job allocation matrix.
        """
        super().__init__()
        node_feat_name = kwargs.get('node_feat_name', None)
        edge_feat_name = kwargs.get('edge_feat_name', None)
        self._make_nx(node_features, edge_features, edge_index, alpha,
                      node_feat_name, edge_feat_name)

    def _make_nx(self, node_features, edge_features, edge_index, alpha,
                 node_feat_name=None, edge_feat_name=None):
        """ """
        self.graph['alpha'] = ','.join(map(str, alpha.numpy().flatten()))
        self.graph['m'] = alpha.shape[1]

        nodes = range(node_features.shape[0])
        
        if node_feat_name is None:
            node_feat_name = ['nfeature_'+str(i+1) for i in nodes]
        if edge_feat_name is None:
            edge_feat_name = ['efeature_'+str(i+1) for i in range(edge_features.shape[1])]

        node_for_adding = zip(
            nodes,
            map(lambda vec: dict(zip(node_feat_name,vec.numpy())), node_features)
        )
        
        edge_for_adding = map(
            lambda e_indx,attr: tuple(e_indx)+(attr,), 
            np.array(edge_index, dtype='int').T, 
            map(lambda vec: dict(zip(edge_feat_name,vec.numpy())), edge_features)
        )

        self.add_nodes_from(node_for_adding)
        self.add_edges_from(edge_for_adding)
    
    @property
    def node_features(self) -> np.ndarray:
        """Node feature matrix of MSP"""
        node_features = tf.stack(list(map(
            lambda _dict: [*_dict.values()],
            self.nodes(data=True)._nodes.values()
        )))
        return node_features

    @property
    def edge_features(self) -> np.ndarray:
        """Edge feature matrix of MSP"""
        edge_features = tf.stack(list(map(
            lambda e: [*e[-1].values()],
            list(self.edges(data=True))
        )))
        return edge_features

    @property
    def edge_index(self) -> np.ndarray:
        """Edge index of MSP graph"""
        edge_index = tf.constant(self.edges)
        return tf.transpose(edge_index)
        
    @property
    def alpha(self) -> np.ndarray:
        """Job allocation matrix of MSP"""
        m = self.graph['m']
        alpha = tf.constant(np.array(self.graph['alpha'].split(','), dtype='float')\
                  .reshape(self.number_of_nodes(),m))
        return alpha

    def to_dict(self) -> dict:
        """Convert MSPGraph object to dictionary"""
        _dict = {
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'edge_index': self.edge_index,
            'alpha': self.alpha,
        }
        return _dict

    @staticmethod
    def from_dict(_dict:dict) -> MSPGraph:
        """Create MSPGraph object from dictionary"""
        msp = MSPGraph(**_dict)
        return msp

    def writeAs(self, path, format='graphml') -> None:
        """Write MSPGraph into Gephi compatible format"""
        if format == 'graphml':
            nx.write_graphml_lxml(self, path, named_key_ids=True)
        elif format == 'gexf':
            nx.write_gexf(self, path)
        else:
            raise TypeError("Unsupported format.")