"""
The :mod:`msp.datasets` module includes methods to generate synthetic data sets.
"""

from ._base import load_data, load_sample_data
from ._samples_generator import make_data

__all__ = ['load_sample_data', 'load_data', 'make_data']



