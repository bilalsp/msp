"""

"""
from typing import NamedTuple

from tensorflow import Tensor


class MSPSparseGraph(NamedTuple):
    """ """
    adj_matrix: Tensor
    node_features: Tensor
    edge_features: Tensor
    alpha: Tensor

    @property
    def num_node(self):
        return self.adj_matrix.shape[-2]


class MSPEmbedGraph(
    NamedTuple(
        'MSPEmbedGraph',
        [
            *MSPSparseGraph._field_types.items(),
            ('node_embed', Tensor),
            ('edge_embed', Tensor)
        ]
    ), MSPSparseGraph):
    """ """
    pass
