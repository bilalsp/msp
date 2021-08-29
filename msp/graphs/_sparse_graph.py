"""
The :mod:`msp.graphs._sparse_graph` module defines named tuple 
for MPS graph.
"""
from typing import NamedTuple, Callable

from tensorflow import Tensor


class MSPSparseGraph(NamedTuple):
    """Named Tuple defining MSP Sparse Graph"""
    adj_matrix: Tensor
    node_features: Tensor
    edge_features: Tensor
    job_assignment: Tensor

    @property
    def num_node(self):
        """Get number of nodes in a graph"""
        return self.job_assignment.shape[-2]

    @property
    def num_machine(self):
        """Get number of machine nodes in a graph"""
        return self.job_assignment.shape[-1]

    @property
    def num_job(self):
        """Get number of job nodes in a graph"""
        return self.num_node - self.num_machine

    @property
    def msp_size(self):
        """Get size of the msp instance"""
        return (self.num_job, self.num_machine)

    @property
    def batch_size(self):
        """Get batch size of the batch instance"""
        assert len(self.job_assignment.shape) == 3, "It's not a batch instance."
        return self.job_assignment.shape[0]

    @property
    def unbatch(self, keepdims=True) -> Callable:
        """Unbatch the batch instance
        
        Args: 
            keepdims: if false batch dimension will be removed
        
        Returns:
            python generator
        """
        def _generator():
            for batch_i in range(self.batch_size):
                if keepdims:
                    batch_i = slice(batch_i, batch_i+1)
                yield MSPSparseGraph(
                    self.adj_matrix[batch_i, :, :],
                    self.node_features[batch_i, :, :],
                    self.edge_features[batch_i, :, :, :],
                    self.job_assignment[batch_i, :, :]
                )
        return _generator

    @property
    def shape(self):
        """Get shape of the msp instance"""
        graph_shape = MSPSparseGraph(
            adj_matrix=self.adj_matrix.shape,
            node_features=self.node_features.shape,
            edge_features=self.edge_features.shape,
            job_assignment=self.job_assignment.shape
        )
        return graph_shape


class MSPEmbedGraph(
    NamedTuple(
        'MSPEmbedGraph',
        [
            *MSPSparseGraph._field_types.items(),
            ('node_embed', Tensor),
            ('edge_embed', Tensor)
        ]
    ), MSPSparseGraph):
    """Named Tuple defining Embedded MSP Graph"""
    pass