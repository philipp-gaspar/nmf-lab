import sys
import numpy as np

import json
import time

from ..utils.matrix_utils import column_norm, scale_factor_matrices

class NMF_SPARSE(object):
    """
    Base class for Sparse NMF algorithms
    """

    def __init__(self):
        # NMF_SPARSE is a Base class and cannot be instantiated
        raise NotImplementedError()

    def run(self, V, r, alpha, init=None, max_iter=100, verbose=False):
        """
        Run a particular Sparse NMF algorithm.
        NMF factorization go as: V = WH

        Parameters
        ----------
        V : numpy.array
            Data matrix, shape (n, m).
                - rows -> features
                - columns -> measurements
        r : int
            Nunber of components.
        alpha : float
            Sparseness parameter.

        Optionals
        ---------
        init : list
            List with initial W and H matrices.
        max_iter : int
            Maximum number of iterations.
        verbose : boolean
            Verbose variable.

        Returns
        -------
        results : dict
            Dictionary with the results from the algorithm.
                W - factor matrix, shape (n, r)
                H - coefficient matrix, shape (r, m)
        """
        self.info = {'r': r,
                     'V_dim_1': V.shape[0],
                     'V_dim_2': V.shape[1],
                     'V_type': str(V.__class__),
                     'alpha': alpha,
                     'max_iter': max_iter,
                     'verbose': verbose}
        if verbose:
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
        W, H = self.initializer(W, H)

        # Dictionary to save the results for each iteration
        self.results = {'n_components': r,
                        'alpha': alpha,
                        'iter': [],
                        'error': [],
                        'sparse_error': [],
                        'total_error': [],
                        'sparseness_W': []}

        # Iteration process
        for it in range(1, max_iter+1):
            W, H, errors_dict = self.iter_solver(V, W, H, alpha)
            self.results['iter'].append(it)
            self.results['error'].append(errors_dict['error'])
            self.results['sparse_error'].append(errors_dict['sparse_error'])
            self.results['total_error'].append(errors_dict['total_error'])

            # CALCULATE SPARSENESS ON W
            sparseness_W = self.sparseness_on_basis(W)
            self.results['sparseness_W'].append(sparseness_W)

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

    def run_repeat(self, V, r, alpha, num_trials,
                   max_iter=100, verbose='False'):
        """
        Run an NMF algorithm several times with random initial values and
        return the best result in terms of the total error function choosen.

        Parameters
        ----------
        V : numpy.array
            Data matrix, shape (n, m).
                - rows -> features
                - columns -> measurements
        r : int
            Number of components.
        alpha : float
            Sparseness parameter.
        num_trials : int
            Number of different trials/initializations for the NMF algorithm.

        Optionals
        ---------
        max_iter : int
            Maximum number of iterations.
        verbose : boolean
            Verbose variable.

        Returns
        -------
        best_model : dict
            Dictionary with the results from the best model:
                W - factor matrix, shape (n, r)
                H - coefficient matrix, shape (r, m)
        """
        if verbose:
            print('[NMF] Running %i trials.' % (num_trials))

        for trial in range(num_trials):
            actual_model = self.run(V, r, alpha, init=None, max_iter=max_iter, \
            verbose=False)
            if trial == 0:
                best_model = actual_model
            else:
                if actual_model['total_error'][-1] < best_model['total_error'][-1]:
                    best_model = actual_model

        if verbose:
            print('[NMF] Best result has error = %1.3f' % best_model['total_error'][-1])

        return best_model

    def sparseness_on_basis(self, W):
        """
        Define the sparseness of a matrix W based on the relationship
        between the L1 and L2 norm [1].

        Parameter
        ----------
        W : numpy.array
            Basis matrix found by NMF algorithm.
                - rows -> features
                - columns -> components

        Returns
        -------
        sparseness : list
            A list with the sparseness for each base vector (columns of W).

        REF:
        [1] "Non-Negative Matrix Factorization with Sparseness Constraints",
        P. O. Hoyer, Journal of Machine Learning Research (2004)
        """
        eps = 1e-16
        n = W.shape[0] # number of features / dimention of component vector
        r = W.shape[1] # numer of components
        norm_factor = (np.sqrt(n) - 1) + eps
        sparseness = []

        for comp in range(r):
            numerator = np.sqrt(n) - np.sum(abs(comp))
            denominator = np.sqrt(np.sum(np.power(comp, 2))) + eps
            result = (numerator / denominator)
            sparseness.append(result/norm_factor)

        return sparseness

    def iter_solver(self, V, W, H, alpha):
        raise NotImplementedError

    def initializer(self, W, H):
        return W, H

class nmf_sparse_euc(NMF_SPARSE):
    """
    Implements NMF using the Euclidean Distance Metric [1] with sparsity
    constraints proposed in [2, 3].

    NMF factorization goes as:
    V = WH; s. t. W>=0, H>=0

    The cost function is:
    np.sum(np.power(V - W.dot(H), 2)) - (alpha * np.sum(H))


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
        ones = np.ones([V.shape[0], V.shape[0]])

        # Update H (This is the original multiplicative update rule)
        numerator = H * (W.T.dot(V))
        denominator = ((W.T.dot(W)).dot(H)) + alpha
        H = numerator / np.maximum(denominator, self.eps)

        # Update W
        HHT = H.dot(H.T)
        aux_1 = W * ones.dot(W.dot(HHT) * W)
        numerator = W * V.dot(H.T) + aux_1
        aux_2 = W * ones.dot(V.dot(H.T) * W)
        denominator = W.dot(HHT) + aux_2
        W = numerator / np.maximum(denominator, self.eps)

        # normalize columns of W
        norm_vec = column_norm(W, by_norm='1')
        W = W / norm_vec[None, :]

        # Calculate errors
        V_hat = W.dot(H) # reconstruction
        error = np.sum(np.power(V - V_hat, 2)) # frobenius error
        sparse_error = np.sum(H)
        total_error = error + (alpha * sparse_error)

        # create a dictonary with errors
        errors_dict = {'error': error,
                       'sparse_error': sparse_error,
                       'total_error': total_error}

        return W, H, errors_dict
