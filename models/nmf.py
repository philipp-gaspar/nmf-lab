import numpy as np

import json

class NMF(object):
    """
    Base class for NMF algorithms.
    """

    def __init__(self):
        """
        """
        self.default_max_iter = 100

    def run(self,
            Y,
            j,
            init=None,
            max_iter=self.default_max_iter,
            verbose=False):
        """
        Run a particular NMF algorithm.

        Arguments:
        ---------
            Y (numpy.array): data matrix, shape (i, t)
            j (int): target lower rank

            optionals:
            init:
            max_iter:
            verbose:

        Returns:
        -------
        A: factor matrix, shape (i, j)
        B: coefficient matrix, shape (t, j)
        """

        self.info = {'j': j,
                     'A_dim_1': A.shape[0],
                     'A_dim_2:' A.shape[1],
                     'A_type': str(A.__class__),
                     'max_iter': max_iter,
                     'verbose': verbose}

        # first implement the uniform random itialization
        self.info['init'] = 'uniform_random'
        A = np.random.rand(Y.shape[0], j)
        B = np.random.rand(Y.shape[1], j)

        if verbose:
            print "[NMF] Running: "
            print json.dumps(self.info, indent=4, sort_keys=True)
