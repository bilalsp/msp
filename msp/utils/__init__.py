"""
The :mod:`msp.utils` module includes various utilities.
"""
import numbers

import numpy as np


__all__ = ['_is_connected', 'check_random_state']


def _is_connected(node_features, alpha, pair) -> bool:
    """Check whether the give pair of nodes are connected.

    Parameters
    ----------
    node_features : ndarray of shape (n_nodes,node_feat_size), dtype=float
        Features for each node in a graph.
    alpha: ndarray of shape (n_nodes,n_machines), dtype=int
        Job allocation matrix.
    pair : tuple or list
        Pair of node which need to be checked.

    Returns
    -------
    boolean
    """

    node_i, node_j = pair

    if (
        node_i == node_j or
        not any([node_features[node_i,-1],node_features[node_j,-1]]) or
        (
            node_features[node_i,-1] and
            node_features[node_j,-1] and
            np.dot(alpha[node_i], alpha[node_j]) == 0
        )
    ):
        return False

    return True


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    
    Parameters
    ----------
    seed : None, int or instance of RandomState
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

