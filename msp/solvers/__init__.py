"""
The :mod:`mps.solvers` module defines two different solver namely
`ExactSolver` and `RandomSolver` to make a comparision with 
proposed deep learning model.
"""
from msp.solvers._exact_solver import ExactSolver
from msp.solvers._random_solver import RandomSolver

__all__ = ['ExactSolver', 'RandomSolver']