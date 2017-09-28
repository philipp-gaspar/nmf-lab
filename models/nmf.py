import sys
import numpy as np

import json
import time

from ..utils.matrix_utils import column_norm, scale_factor_matrices

class NMF(object):
    """
    Base class for NMF algorithms.
    """
    default_max_iter = 100

    def __init__(self):
        # NMF is a Base class and cannot be instantiated
        raise NotImplementedError()

    def run(self, Y, j,
            init=None,
            max_iter=default_max_iter,
            cost_function='frobenius',
            verbose=False):
        """
        Run a particular NMF algorithm.

        Parameters
        ----------
        Y : numpy.array
            Data matrix, shape (i, t).
        j : int
            Target lower rank.

        Optionals
        ---------
        init : list
            List with inital A and B matrices.
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
                A - factor matrix, shape (i, j)
                B - coefficient matrix, shape (t, j)
        """
        self.cost_function = cost_function
        self.info = {'j': j,
                     'Y_dim_1': Y.shape[0],
                     'Y_dim_2': Y.shape[1],
                     'Y_type': str(Y.__class__),
                     'max_iter': max_iter,
                     'cost_function': self.cost_function,
                     'verbose': verbose}

        # Initialization of factor matrices
        if init != None:
            A = init[0].copy()
            B = init[1].copy()
            self.info['init'] = 'user_provided'
        else:
            # non-negative random matries like used scaled
            # with sqrt(Y.mean() / n_components)
            # same strategy used in the NMF package in Sklearn
            avg = np.sqrt(Y.mean() / j)
            A = np.abs(avg * np.random.randn(Y.shape[0], j))
            B = np.abs(avg * np.random.randn(Y.shape[1], j))
            self.info['init'] = 'random'

        if verbose:
            print "[NMF] Running: "
            print json.dumps(self.info, indent=4, sort_keys=True)

        start = time.time()

        # Algorithm specific initialization
        A, B = self.initializer(A, B)

        # Dictionary to save the results for each iteration
        self.results = {'cost_function': self.cost_function,
                        'n_components': j,
                        'iter': [],
                        'error': []}

        # Iteration Process
        for i in range(1, self.info['max_iter'] + 1):
            A, B, error = self.iter_solver(Y, A, B, j, i)
            self.results['iter'].append(i)
            self.results['error'].append(error)

        # Normalize matrices A and B in order to not
        # modify the product A.dot(B.T)
        A, B = scale_factor_matrices(A, B, by_norm='1')
        Y_hat = A.dot(B.T)

        self.results['Y_hat'] = Y_hat
        self.results['A'] = A
        self.results['B'] = B

        # Final info
        self.final = {}
        self.final['iterations'] = i
        self.final['elapsed'] = time.time() - start
        if verbose:
            print "[NMF] Completed:"
            print json.dumps(self.final, indent=4, sort_keys=True)

        return self.results

    def run_repeat(self, Y, j, num_trials,
                   max_iter=default_max_iter,
                   cost_function='frobenius',
                   verbose=False):
        """
        Run an NMF algorithms several times with random
        initial values and return the best results in terms
        of the error function choosen.
        """
        if verbose:
            print '[NMF] Running %i random trials.' % (num_trials)

        for trial in range(num_trials):

            actual_model = self.run(Y, j,
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

    def iter_solver(self, Y, A, B, j, it):
        raise NotImplementedError

    def initializer(self, A, B):
        return A, B

class NMF_HALS(NMF):
    """
    NMF algorithm: Hierarchical Alternating Least Squares

    Improved and modified HALS NMF algorithm called:
    FAST HALS for Large Scale NMF

    REF [1]: Pseudo-Code in the book: Nonnegative Matrix and Tensor
    Factorizations by A. Cichocki et All; Page 219; Algorithm 4.3
    """
    def __init__(self, default_max_iter=100):
        self.eps = 1e-16

    def initializer(self, A, B):
        # Normalize columns of matrix A to l2 unity norm
        norm_vec = column_norm(A, by_norm='2')
        A = A / norm_vec[None, :]

        return A, B

    def iter_solver(self, Y, A, B, j, it):
        # Check the cost function. FAST HALS algorithm only
        # accepts Frobenius.
        if self.cost_function != 'frobenius':
            print "Cost function not valid for HALS algorithm."
            print "Try: 'frobenius'."
            sys.exit()

        # Update B
        W = Y.T.dot(A)
        V = A.T.dot(A)

        for jj in range(j):
            b_jj = B[:, jj] + W[:, jj] - B.dot(V[:, jj])
            B[:, jj] = np.maximum(b_jj, self.eps)

        # Update A
        P = Y.dot(B)
        Q = B.T.dot(B)

        for jj in range(j):
            a_jj = A[:, jj] * Q[jj, jj] + P[:, jj] - A.dot(Q[:, jj])
            A[:, jj] = np.maximum(a_jj, self.eps)
            norm_value = np.sqrt(np.sum(A[:, jj] * A[:, jj]))
            A[:, jj] = A[:, jj] / norm_value

        # Calculate error
        Y_hat = A.dot(B.T)
        error = 0.5 * np.linalg.norm(Y-Y_hat, ord='fro')

        return A, B, error

class NMF_MU(NMF):
    """
    NMF algorithm: Multiplicative Updating

    Three possible cost functions:
        - frobenius
        - kullback-leibler
        - itakura-saito

    REF [1]: Multiplicative Update Rules for Nonnegative Matrix Factorization
    with Co-occurrence Constraints by Steven K. Tjoa and K. J. Ray Liu
    """
    def __init__(self, default_max_iter=100):
        self.eps = 1e-16

    def initializer(self, A, B):
        # Normalize columns of matrix A to l1 unity norm
        norm_vec = column_norm(A, by_norm='1')
        A = A / norm_vec[None, :]

        return A, B

    def iter_solver(self, Y, A, B, j, it):
        # FROBENIUS
        if self.cost_function == 'frobenius':
            # Update A
            YB = Y.dot(B)
            numerator = A * YB
            denominator = A.dot(B.T.dot(B)) + self.eps
            A = numerator / denominator

            # normalize columns of A
            norm_vec = column_norm(A, by_norm='1')
            A = A / norm_vec[None, :]

            # Update B
            YtA = Y.T.dot(A)
            numerator = B * YtA
            denominator = B.dot(A.T.dot(A)) + self.eps
            B = numerator / denominator

            # Calculate error
            Y_hat = A.dot(B.T)
            error = 0.5 * np.linalg.norm(Y-Y_hat, ord='fro')

        # KULLBACK-LEIBLER
        elif self.cost_function == 'kullback-leibler':
            X = B.T # to keep with the original form
            ones = np.ones([Y.shape[0], Y.shape[1]])
            AX = A.dot(X) + self.eps # initial reconstruction

            # Update A
            numerator = A * ((Y / AX).dot(X.T))
            denominator = np.maximum(ones.dot(X.T), self.eps)
            A = numerator / denominator

            # normalize columns of A
            norm_vec = column_norm(A, by_norm='1')
            A = A / norm_vec[None, :]

            # update reconstruction
            AX = A.dot(X) + self.eps

            # Update B
            numerator = X * (A.T.dot(Y / AX))
            denominator = np.maximum(A.T.dot(ones), self.eps)
            X = numerator / denominator
            B = X.T

            # Calculate error
            Y_hat = A.dot(B.T) + self.eps
            error = ((Y * np.log((Y/Y_hat) + self.eps)) - Y + Y_hat).sum(axis=None)

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

        return A, B, error
