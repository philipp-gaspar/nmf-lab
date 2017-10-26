import sys
import numpy as np

from nmf import NMF

from ..utils.matrix_utils import row_norm, scale_factor_matrices

class NMF_Sparseness_Constraints(NMF):
    """
    NMF algorithm with sparsity constraints.
    """
    def __init__(self, default_max_iter=100):
        self.eps = 1e-16

    def initializer(self, W, H):
        # normalize rows of matrix H to l2 norm.
        norm_vec = row_norm(H, by_norm='2')
        H = H / norm_vec[:, None]

        return W, H

    # NEED CONVERTION!!
    def iter_solver(self, Y, A, B, j, it,
                    s_A, s_B):
        X = B.T # to keep with the original form

        # data dimensions
        n = Y.shape[0]
        m = Y.shape[1]

        # NEED IMPLEMENTATION
        # if it == 0:
            # make initial matrices have correct sparseness

        # ---------
        # Update A
        # ---------
        # IF SPARSENESS ON A
        if s_A != 0:
            dA = (A.dot(X) - Y).dot(X.T) # gradient for A

            Y_hat = A.dot(X)
            error = 0.5 * np.linalg.norm(Y-Y_hat, ord='fro') # calculate error

            # make sure we decrease the objective function
            while True:

                # take the step in the direction of the negative gradient
                A_new = A - stepsize_A * dA

                # project each column of A according to the authors
                norm_vec = column_norm(A, by_norm='2')
                for col in range(A.shape[1]):
                    A_new[:, col] = project_func(A_new[:, col],
                                                 L1_a * norm_vec[col],
                                                 np.power(norm_vec[col], 2),
                                                 1)
                # calculate new error
                new_error = 0.5 * np.linalg(Y - (A_new.dot(X)), ord='fro')

                # if the objective decrease we can continue
                if new_error <= error:
                    break
                # else, decrease stepsize and try again
                stepsize_A = stepsize_A / 2

                # check for convergence
                if stepsize_A < 1e-200:
                    print('Algorithm converged.\n')
                    A = A_new
                    error = new_error
                    return A, B, error

            # slightly increase the step size
            stepsize_A = stepsize_A * 1.2
            A = A_new

        # IF A IS NOT SPARSE
        else:
            # update using standard NMF multiplicative rule
            A = A * (Y.dot(X.T)) / A.dot(X.dot(X.T)) + self.eps

        # normalize columns of W

        # ---------
        # UPDATE X
        # ---------
        # IF SPARSENESS ON X
        if s_X != 0:
            dX = A.T * (A.dot(X)-Y) # gradient for X

            Y_hat = A.dot(X)
            error = 0.5 * np.linalg.norm(Y-Y_hat, ord='fro') # calculate error

            # make sure we decrease the objective
            while True:

                # take step in the direction of the negative gradient
                X_new = X - stepsize_X * dX

                # project each row of X according to the authors
                for row in range(X.shape[0]):
                    X_new[row, :] = (project_func(X_new[row,:].T, Ls1, 1, 1)).T

                # calculate new error
                new_error = 0.5 * np.linalg(Y - (A.dot(X_new)), ord='fro')

                # if the objective decrease we can continue
                if new_error <= error:
                    break
                # else, decrease stepsize and try again
                stepsize_X = stepsize_X / 2

                # check for convergence
                if stepsize_X < 1e-200:
                    print('Algorithm converged.\n')
                    return A, B, new_error

            # slightly increase the stepsize
            stepsize_X = stepsize_X * 1.2
            X = X_new

        # IF X IS NOT SPARSE
        else:
            # update using standard NMF multiplicative rule
            X = X * (A.T.dot(Y)) / A.T.dot(A.dot(X)) + self.eps
