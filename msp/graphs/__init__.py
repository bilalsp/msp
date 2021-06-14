"""
The :mod:`msp.graphs` module includes methods to create a MSPGraph object
which has been built on top of the networkx graph.
"""

from ._graphs import MSPGraph
from ._sparse_graphs import MSPSparseGraph

__all__ = ['MSPGraph', 'MSPSparseGraph']

