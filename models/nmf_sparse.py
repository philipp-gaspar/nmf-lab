import sys
import numpy as np


class NMF_Sparse_Base(object):
    """
    Base class for Sparse NMF algorithms.
    """
    default_max_iter = 100

    def __init__(self):
        # NMF_Sparse_Base is a Base class and cannot be instantiated
        raise NotImplementedError()

    def initializer(self, A, X):
        return A, X

    def iter_solver(self, Y, A, X, j, it):
        raise NotImplementedError()

class NMF_Sparse(NMF_Sparse_Base):
    """
    NMF algorithm with sparsity constraints.

    Two possible cost functions:
        - 'euclidean'
        - 'kullback-leibler'
    """
    def __init__(self, default_max_iter=100):
        self.eps = 1e-16

    def initializer(self, A, X):
