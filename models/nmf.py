import sys
import numpy as np

import json
import time

from ..utils.matrix_utils import column_norm, scale_factor_matrices
from ..utils.metrics import frobenius_norm, kullback_leibler_divergence

class NMF(object):
    """
    Base class for NMF algorithms.
    """
    default_max_iter = 100

    def __init__(self):
        # NMF is a Base class and cannot be instantiated
        raise NotImplementedError()

    def run(self, V, r,
            init=None,
            max_iter=default_max_iter,
            cost_function='frobenius',
            verbose=False):
        """
        Run a particular NMF algorithm.
        NMF factorization go as: V = WH

        Parameters
        ----------
        V : numpy.array
            Data matrix, shape (n, m).
                - rows -> features
                - columns -> measurements
        r : int
            Number of components.

        Optionals
        ---------
        init : list
            List with inital W and H matrices.
        max_iter : int
            Maximum number of iterations.
        cost_function : str
            Name of cost function used in the iteration process.
            Possible values are:
                - 'frobenius'
                - 'kullback-leibler'
                - 'itakura-saito'
        verbose : boolean
            Verbose variable.

        Returns
        -------
        results : dict
            Dictionary with the results from the algorithm:
                W - factor matrix, shape (n, r)
                H - coefficient matrix, shape (r, m)
        """
        self.cost_function = cost_function
        self.info = {'r': r,
                     'V_dim_1': V.shape[0],
                     'V_dim_2': V.shape[1],
                     'V_type': str(V.__class__),
                     'max_iter': max_iter,
                     'cost_function': self.cost_function,
                     'verbose': verbose}

        # Initialization of factor matrices
        if init != None:
            W = init[0].copy()
            H = init[1].copy()
            self.info['init'] = 'user_provided'
        else:
            # non-negative random matrices scaled
            # with sqrt(V.mean() / r)
            # same strategy used in the NMF package in Sklearn
            avg = np.sqrt(V.mean() / r)
            W = np.abs(avg * np.random.randn(V.shape[0], r))
            H = np.abs(avg * np.random.randn(r, V.shape[1]))
            self.info['init'] = 'normally-random'

        if verbose:
            print "[NMF] Running: "
            print json.dumps(self.info, indent=4, sort_keys=True)

        start = time.time()

        # Algorithm specific initialization
        W, H = self.initializer(W, H)

        # Dictionary to save the results for each iteration
        self.results = {'cost_function': self.cost_function,
                        'n_components': j,
                        'iter': [],
                        'error': []}

        # Iteration Process
        for i in range(1, self.info['max_iter'] + 1):
            W, H, error = self.iter_solver(V, W, H, r, i)
            self.results['iter'].append(i)
            self.results['error'].append(error)

        # Normalize matrices W and H in order to not
        # modify the product W.dot(H)
        W, H = scale_factor_matrices(W, H, by_norm='1')
        V_hat = W.dot(H)

        self.results['V_hat'] = V_hat
        self.results['W'] = W
        self.results['H'] = H

        # Final info
        self.final = {}
        self.final['iterations'] = i
        self.final['elapsed'] = time.time() - start
        if verbose:
            print "[NMF] Completed:"
            print json.dumps(self.final, indent=4, sort_keys=True)

        return self.results

    def run_repeat(self, V, r, num_trials,
                   max_iter=default_max_iter,
                   cost_function=self.cost_function,
                   verbose=False):
        """
        Run an NMF algorithm several times with random
        initial values and return the best results in terms
        of the error function choosen.

        Parameters
        ----------
        V : numpy.array
            Data matrix, shape (n, m).
                - rows -> features
                - columns -> measurements
        r : int
            Number of components.
        num_trials : int
            Number of different trials/initializations for the NMF algorithm.

        Optionals
        ---------
        max_iter : int
            Maximum number of iterations.
        cost_function : str
            Name of cost function used in the iteration process.
            Possible values are:
                - 'frobenius'
                - 'kullback-leibler'
                - 'itakura-saito
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
            print '[NMF] Running %i random trials.' % (num_trials)

        for trial in range(num_trials):

            actual_model = self.run(V, r,
                                    init=None,
                                    max_iter=max_iter,
                                    cost_function=cost_function,
                                    verbose=False)

            if trial == 0:
                best_model = actual_model
            else:
                if actual_model['error'][-1] < best_model['error'][-1]:
                    best_model = actual_model

        if verbose:
            print '[NMF] Best result has error = %1.3f' % best_model['error'][-1]

        return best_model

    def iter_solver(self, V, W, H, r, it):
        raise NotImplementedError

    def initializer(self, W, H):
        return W, H

# class NMF_HALS(NMF):
#     """
#     NMF algorithm: Hierarchical Alternating Least Squares
#
#     Improved and modified HALS NMF algorithm called:
#     FAST HALS for Large Scale NMF
#
#     REF [1]: Pseudo-Code in the book: Nonnegative Matrix and Tensor
#     Factorizations - A. Cichocki et All; Page 219; Algorithm 4.3
#     """
#     def __init__(self, default_max_iter=100):
#         self.eps = 1e-16
#
#     def initializer(self, A, B):
#         # Normalize columns of matrix A to l2 unity norm
#         norm_vec = column_norm(A, by_norm='2')
#         A = A / norm_vec[None, :]
#
#         return A, B
#
#     def iter_solver(self, Y, A, B, j, it):
#         # Check the cost function. FAST HALS algorithm only
#         # accepts Frobenius.
#         if self.cost_function != 'frobenius':
#             print "Cost function not valid for HALS algorithm."
#             print "Try: 'frobenius'."
#             sys.exit()
#
#         # Update B
#         W = Y.T.dot(A)
#         V = A.T.dot(A)
#
#         for jj in range(j):
#             b_jj = B[:, jj] + W[:, jj] - B.dot(V[:, jj])
#             B[:, jj] = np.maximum(b_jj, self.eps)
#
#         # Update A
#         P = Y.dot(B)
#         Q = B.T.dot(B)
#
#         for jj in range(j):
#             a_jj = A[:, jj] * Q[jj, jj] + P[:, jj] - A.dot(Q[:, jj])
#             A[:, jj] = np.maximum(a_jj, self.eps)
#             norm_value = np.sqrt(np.sum(A[:, jj] * A[:, jj]))
#             A[:, jj] = A[:, jj] / norm_value
#
#         # Calculate error
#         Y_hat = A.dot(B.T)
#         error = 0.5 * np.linalg.norm(Y-Y_hat, ord='fro')
#
#         return A, B, error

class NMF_MU(NMF):
    """
    NMF algorithm: Multiplicative Updating

    Three possible cost functions:
        - frobenius
        - kullback-leibler
        - itakura-saito

    REF [1]: Multiplicative Update Rules for Nonnegative Matrix Factorization
    with Co-occurrence Constraints - Steven K. Tjoa and K. J. Ray Liu
    """
    def __init__(self, default_max_iter=100):
        self.eps = 1e-16

    def initializer(self, W, H):
        # Normalize columns of matrix A to l1 unity norm
        norm_vec = column_norm(W, by_norm='1')
        W = W / norm_vec[None, :]

        return W, H

    def iter_solver(self, Y, A, B, j, it):
        # FROBENIUS
        if self.cost_function == 'frobenius':
            # Update W
            numerator = W * (V.dot(H.T))
            denominator = W.dot(H.dot(H.T)) + self.eps
            W = numerator / denominator

            # normalize columns of W
            norm_vec = column_norm(W, by_norm='1')
            W = W / norm_vec[None, :]

            # Update H
            numerator = H * (W.T.dot(V))
            denominator = W.T.dot(W.dot(H)) + self.eps
            H = numerator / denominator

            # Calculate error
            V_hat = W.dot(H)
            error = frobenius_norm(V, V_hat)

        # KULLBACK-LEIBLER
        elif self.cost_function == 'kullback-leibler':
            ones = np.ones([V.shape[0], V.shape[1]])\
            WH = W.dot(H) # reconstruction

            # Update W
            numerator = W * ((V / WH).dot(H.T))
            denominator = np.maximum(ones.dot(H.T), self.eps)
            W = numerator / denominator

            # normalize columns of W
            norm_vec = column_norm(W, by_norm='1')
            W = W / norm_vec[None, :]

            # update reconstruction
            WH = W.dot(H)

            # Update H
            numerator = H * (W.T.dot(V / WH))
            denominator = np.maximum(W.T.dot(ones), self.eps)
            H = numerator / denominator

            # Calculate error
            V_hat = W.dot(H)
            error = kullback_leibler_divergence(V, V_hat)

        # ITAKURA-SAITO
        elif self.cost_function == 'itakura-saito':
            X = B.T # to keep the original form
            ones = np.ones([Y.shape[0], Y.shape[1]])
            AX = A.dot(X) + self.eps # reconstruction

            # Upadate A
            numerator = A * (Y/np.power(AX, 2)).dot(X.T)
            denominator = (ones / AX).dot(X.T)
            A = numerator / denominator

            # normalize columns of A
            norm_vec = column_norm(A, by_norm='1')
            A = A / norm_vec[None, :]

            # update reconstruction
            AX = A.dot(X) + self.eps

            # Update B
            numerator = X * (A.T.dot(Y/np.power(AX, 2)))
            denominator = A.T.dot(ones / AX)
            X = numerator / denominator
            B = X.T

            # Calculate error
            Y_hat = A.dot(B.T) + self.eps
            error = ((Y/(Y_hat + self.eps)) - np.log((Y/Y_hat + self.eps)+self.eps) - 1).sum(axis=None)

        else:
            print "Not a valid cost function."
            print "Try: 'frobenius', 'kullback-leibler' or 'itakura-saito'."
            sys.exit()

        return W, H, error

# class LOCAL_NMF(NMF):
#     """
#     Local Non-Negative Matrix Factorization (LNMF)
#
#     This NMF algorithmm is aimed to learn spatially localized, parts-based
#     subspace representations of visual patterns.
#     An objective fuction (Kullback-Leibler) localization constrain, in addition
#     to the non-negativity constrain in the standard NMF.
#
#     REF [1]: Learning Spatially Localized, Parts-Based Representations -
#     S. Z. Li, X. Hou, H. Zhang and Q. Cheng - (2001).
#
#     NOTE
#     ----
#     Even though the stated objective function has weights alpha and
#     beta, these disappear in the simplifications made by the authors when
#     deriving the updates. Therefore, we report the regular Kullback-Leibler
#     divergence as the error here.
#     """
#     def __init__(self, default_max_iter=100):
#         self.eps = 1e-16
#
#     def initializer(self, A, B):
#         # Normalize columns of matrix A to l1 unity norm
#         norm_vec = column_norm(A, by_norm='1')
#         A = A / norm_vec[None, :]
#
#         return A, B
#
#     def iter_solver(self, Y, A, B, j, it):
#         # Check the cost function. LNMF only accepts Kullback-Leibler.
#         if self.cost_function != 'kullback-leibler':
#             print("Cost function not valid for LNMF algorithm.")
#             print("Try: 'kullback-leibler'.")
#             sys.exit()
#
#         X = B.T # to keep with the original form
#         ones = np.ones([Y.shape[0], Y.shape[1]])
#         AX = A.dot(X) + self.eps # initial reconstruction
#
#         # Update A
#         numerator = A * ((Y / AX).dot(X.T))
#         denominator = np.maximum(ones.dot(X.T), self.eps)
#         A = numerator / denominator
#
#         # normalize columns of A
#         norm_vec = column_norm(A, by_norm='1')
#         A = A / norm_vec[None, :]
#
#         # update reconstruction
#         AX = A.dot(X) + self.eps
#
#         # Update B
#         AX = np.maximum(AX, self.eps)
#         X = np.sqrt(X * (A.T.dot(Y / AX)))
#         B = X.T
#
#         # Calculate error
#         Y_hat = A.dot(B.T) + self.eps
#         error = ((Y * np.log((Y/Y_hat) + self.eps)) - Y + Y_hat).sum(axis=None)
#
#         return A, B, error
