import sys
import numpy as np

import json
import time

from utils.matrix_utils import column_norm, scale_factor_matrices

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
            verbose=True):
        """
        Run a particular NMF algorithm.

        Args:
        ----
        Y (numpy.array): data matrix, shape (i, t)
        j (int): target lower rank

        optionals:
        init (list): list with inital A and B matrices
        max_iter (int): maximum number of iterations
        cost_function (str): name of cost function used in the iteration process.
        Possible values are: 'frobenius', 'kullback-leibler', 'itakura-saito'.
        verbose (boolean): verbose variable

        Returns:
        -------
        A: factor matrix, shape (i, j)
        B: coefficient matrix, shape (t, j)
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
            A = np.random.rand(Y.shape[0], j)
            B = np.random.rand(Y.shape[1], j)
            self.info['init'] = 'uniform_random'

        if verbose:
            print "[NMF] Running: "
            print json.dumps(self.info, indent=4, sort_keys=True)

        start = time.time()
        # Algorithm specific initialization
        A, B = self.initializer(A, B)

        # Dictionary to save the results for each iteration
        self.results = {'cost_function': self.cost_function,
                        'iter': [],
                        'rel_error': []}

        # Iteration Process
        for i in range(1, self.info['max_iter'] + 1):
            A, B, rel_error = self.iter_solver(Y, A, B, j, i)
            self.results['iter'].append(i)
            self.results['rel_error'].append(rel_error)

        # Normalize matrices A and B in order to not
        # modify the product A.dot(B.T)
        A, B = scale_factor_matrices(A, B)
        Y_hat = A.dot(B.T)
        self.results['Y_hat'] = Y_hat

        # Final info
        self.final = {}
        self.final['iterations'] = i
        self.final['elapsed'] = time.time() - start
        if verbose:
            print "[NMF] Completed:"
            print json.dumps(self.final, indent=4, sort_keys=True)

        return A, B

    def iter_solver(self, Y, A, B, j, it):
        raise NotImplementedError

    def initializer(self, A, B):
        return A, B

class NMF_HALS(NMF):
    """
    NMF algorithm: Hierarchical Alternating Least Squares

    Improved and modified HALS NMF algorithm called:
    FAST HALS for Large Scale NMF

    Pseudo-Code in the book: Nonnegative Matrix and Tensor Factorizations
    by A. Cichocki et All; Page 219; Algorithm 4.3
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
        """
        def __init__(self, default_max_iter=100):
            self.eps = 1e-16

        def iter_solver(self, Y, A, B, j, it):
            """
            """
            if self.cost_function == 'frobenius':
                # Update B
                YtA = Y.T.dot(A)
                numerator = B * YtA
                denominator = B.dot(A.T.dot(A)) + self.eps
                B = numerator / denominator

                # Update A
                YB = Y.dot(B)
                numerator = A * YB
                denominator = A.dot(B.T.dot(B)) + self.eps
                A = numerator / denominator

                # Calculate error
                Y_hat = A.dot(B.T)
                error = 0.5 * np.linalg.norm(Y-Y_hat, ord='fro')

            elif self.cost_function == 'kullback-leibler':
                # Update A
                numerator = A * (Y / (A.dot(B.T)).dot(B))
                ones = np.ones(Y.shape[0], Y.shape[1])
                denominator = ones.dot(B)
                A = numerator / denominator

                # Update B
                X = B.T # to keep with the original form
                AX = A.dot(X)
                numerator = X * (A.T.dot(Y / AX))
                denominator = np.maximum(A.T.dot(ones), self.eps)
                X = numerator / denominator
                B = X.T

                # Calculate error
                Y_hat = A.dot(B.T)
                error = 0.0 # EDIT TO-DO!!!

            else:
                print "Not a valid cost function."
                print "Try: 'frobenius'."
                sys.exit()

            return A, B, error
