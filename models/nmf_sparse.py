import sys
import numpy as np

import json
import time
from collections import namedtuple

class NMF_SPARSE(object):
    """
    Base class for Sparse NMF algorithms
    """

    def __init__(self):
        # NMF_SPARSE is a Base class and cannot be instantiated
        raise NotImplementedError()

    def run(self, V, r, alpha=0.0, init=None, max_iter=100, verbose=False):
        """
        Run a particular Sparse NMF algorithm.
        NMF factorization go as: V = WH

        Parameters
        ----------

        Returns
        -------
        results : dict
            Dictionary with the results from the algorithm.
                W - factor matrix, shape (n, r)
                H - coefficient matrix, shape (r, m)
        """
        if verbose:
            self.info = {'r': r,
                         'V_dim_1': V.shape[0],
                         'V_dim_2': V.shape[1],
                         'V_type': str(V.__class__),
                         'alpha': alpha,
                         'max_iter': max_iter,
                         'verbose': verbose}
            print('[NMF] Running: ')
            print(json.dumps(self.info, indent=4, sort_keys=True))

        # Initialization of factor matrices
        if init != None:
            W = init[0].copy()
            H = init[1].copy()
            self.info['init'] = 'user-provided'
        else:
            W = np.abs(np.random.randn(V.shape[0], r))
            H = np.abs(np.random.randn(r, V.shape[1]))
            self.info['init'] = 'normally-random'

        start = time.time()

        # Algorithm specific initialization
        # NEED CHANGE
        W, H = self.initializer(W, H)

        # Dictionary to save the results for each iteration
        self.results = {'n_components': r,
                        'alpha': alpha,
                        'iter': [],
                        'error': [],
                        'sparse_error': [],
                        'total_error': [],
                        'sparseness_W': [],
                        'sparseness_H': []}

        # Iteration process
        for it in range(1, max_iter+1):
            W, H, errors_dict = self.iter_solver(V, W, H, alpha)
            self.results['iter'].append(it)
            self.results['error'].append(errors_dict['error'])
            self.results['sparse_error'].append(errors_dict['sparse_error'])
            self.results['total_error'].append(errors_dict['total_error'])

            # NEED IMPLEMENTATION
            # CALCULATE SPARSENESS ON W AND H

        # Normalize matrices W and H in order to not modify their product
        W, H = scale_factor_matrices(W, H, by_norm='1')

        self.results['W'] = W
        self.results['H'] = H

        # Final info
        if verbose:
            self.final = {}
            self.final['iterations'] = it
            self.final['elapsed'] = time.time() - start
            print('[NMF] Completed: ')
            print(json.dumps(self.final, indent=4, sort_keys=True))

        return self.results

    def iter_solver(self, V, W, H, alpha):
        raise NotImplementedError

    def initializer(self, W, H):
        return W, H

class nmf_sparse_euc(NMF_SPARSE):
    """
    Implements NMF using the Euclidean Distance Metric.

    REFS:
    [1]: "Algorithms for Non-Negative Matrix Factorization", D. Lee
    and S. Seung, NIPS (2001)
    [2]: "Sparse Coding and NMF", J. Eggert and E. Korner,
    Neural Networks (2004)
    [3]: "Wind Noise Reduction Using Non-Negative Sparse Coding", M. Schmidt,
    J. Larsen and F. Hsiao, IEEE MLSP (2007)
    """
    def __init__(self):
        self.eps = 1e-16

    def initializer(self, W, H):
        norm_vec = column_norm(W, by_norm='1')
        W = W / norm_vec[None, :]

        return W, H

    def iter_solver(self, V, W, H, alpha):
        # preallocate matrix of ones
        ones = np.ones(V.shape[0], V.shape[0])

        # Update H
        numerator = H * (W.T.dot(V))
        denominator = ((W.T.dot(W)).dot(H)) + alpha
        H = numerator / np.maximum(denominator, self.eps)

        # Update W
        HHT = H.dot(H.T)
        aux_1 = ones.dot(W.dot(HHT) * W)
        numerator = W * V.dot(H.T) + aux_1
        aux_2 = ones.dot(V.dot(H.T) * W)
        denominator = W.dot(HHT) + aux_2
        W = numerator / np.maximum(denominator, self.eps)

        # normalize columns of W
        norm_vec = column_norm(W, by_norm='1')
        W = W / norm_vec[None, :]

        # Calculate errors
        V_hat = W.dot(H)
        error = np.sum(np.power(V - V_hat, 2)) # frobenius error
        sparse_error = np.sum(H)
        total_error = error + (alpha * sparse_error)

        # create a dictonary with errors
        errors_dict {'error': error,
                     'sparse_error': sparse_error,
                     'total_error': total_error}

        return W, H, errors_dict
