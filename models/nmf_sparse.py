import sys
import numpy as np

import json
import time

from ..utils.matrix_utils import row_norm, scale_factor_matrices

class NMF_Sparse_Base(object):
    """
    Base class for Sparse NMF algorithms.
    """
    default_max_iter = 100

    def __init__(self):
        # NMF_Sparse_Base is a class and cannot be instantiated
        raise NotImplementedError()

    def run(self, V, r,
            sW=0,
            sH=0,
            init=None,
            max_iter=default_max_iter,
            cost_function='frobenius',
            verbose=False):

        """
        Run a particular NMF Sparse algorithm.
        NMF factorization go as V = WH

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
        sW : float
            Sparseness of W matrix.
        sH : float
            Sparseness of H matrix.
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
        pass
    # self.cost_function = cost_function
    # self.info {'r': r,
    #            'V_dim_1': V.shape[0],
    #            'V_dim_2': V.shape[1],
    #            'V_type': str(V.__class__),
    #            'sparse_W': sW,
    #            'sparse_H': sH,
    #            'max_iter': max_iter,
    #            'cost_function': self.cost_function,
    #            'verbose': verbose}
    #
    # # Initialization of factor matrices
    # if init != None:
    #     W = init[0].copy()
    #     H = init[1].copy()
    #     self.info['init'] = 'user_provided'
    # else:
    #     avg = np.sqrt(V.mean() / r)
    #     W = np.abs(avg * np.random.randn(V.shape[0], r))
    #     H = np.abs(avg * np.random.randn(r, V.shape[1]))
    #     self.info['init'] = 'normally-random'
    #
    # if verbose:
    #     print "[NMF] Running: "
    #     print json.dumps(self.info, indent=4, sort_keys=True)
    #
    # start = time.time()
    #
    # # Algorithm specific initialization
    # W, H = self.initializer(W, H)
    #
    # # Dictionary to save the results for each iteration
    # self.results = {'cost_function': self.cost_function,
    #                 'n_components': r,
    #                 'iter': [],
    #                 'error': [],
    #                 'sparseness_W': [],
    #                 'sparseness_H': []}
    #
    # # Iteration Process
    # for i in range(1, self.info['max_iter'] + 1):
    #     W, H, error = self.iter_solver(V, W, H, r, i, sW, sH)
    #     self.results['iter'].append(i)
    #     self.results['error'].append(error)
    #
    #     # calculate sparseness on W and H


class NMF_Sparseness_Constraints():
    """
    NMF algorithm: Non-negative Matrix Factorizations with
    Sparseness Constrints.

    REF [1] Non-negative Matrix Factorizations with
    Sparseness Constrints - P. Hoyer - Journal of Machine Learning
    Research (2004)
    """
    def __init__(self, default_max_iter=100):
        self.eps = 1e-16

    def initializer(self, W, H):
        # normalize rows of matrix H to l2 norm.
        norm2 = np.sqrt(np.sum(np.power(H,2), axis=1))
        H = H / norm2[:, None]

        return W, H

    def run(self, V, r,
            sparse_W=0.0,
            sparse_H=0.0,
            init=None,
            max_iter=100,
            cost_function='frobenius',
            verbose=False):
        """
        """
        self.info = {'r': r,
                     'V_dim_1': V.shape[0],
                     'V_dim_2': V.shape[1],
                     'V_type': str(V.__class__),
                     'sparse_W': sparse_W,
                     'sparse_H': sparse_H,
                     'max_iter': max_iter,
                     'cost_function': cost_function,
                     'verbose': verbose}

        # Initialization of factor matrices
        if init != None:
            W = init[0].copy()
            H = init[1].copy()
            self.info['init'] = 'user-provided'
        else:
            W = abs(np.random.randn(V.shape[0], r))
            H = abs(np.random.randn(r, V.shape[1]))
            self.info['init'] = 'normally-random'

        if verbose:
            print('[NMF] Running: ')
            print(json.dumps(self.info, indent=4, sort_keys=True))

        start = time.time()

        # Algorithm Initialization
        W, H = self.initializer(W, H)

        # Dictionary to save the results for each iteration
        self.results = {'cost_function': cost_function,
                        'n_components': r,
                        'iter': [],
                        'error': [],
                        'sparseness_W': [],
                        'sparseness_H': [],
                        'step_size_W': [],
                        'step_size_H': []}

        # initial step sizes
        step_W = 1.0
        step_H = 1.0

        # Iteration process
        for it in range(1, max_iter+1):
            print('\n--> Iteration #%i' % it)
            W, H, error, step_W, step_W = self.iter_solver(V, W, H, r, it,
                                                           sparse_W, sparse_H,
                                                           step_W, step_H)
            self.results['iter'].append(it)
            self.results['error'].append(error)
            self.results['step_size_W'].append(step_W)
            self.results['step_size_H'].append(step_H)

        # Final factor matrices
        self.results['W'] = W
        self.results['H'] = H

        # Final info
        self.final = {'iterations': i,
                      'elapsed': time.time() - start}
        if verbose:
            print('[NMF] Completed:')
            print(json.dumps(self.final, indent=4, sort_keys=True))

        return self.results

    def project_func(self, s, k1, k2, non_negative):
        """
        Solves the following problem:
        Given a vector s, find the vector v having sum(abs(v))=k1
        and sum(v.^2)=k2 which is closest to s in the euclidean sense.

        If the binary flag is set, the vector v is additionally
        retricted to being non-negative (v>=0)

        Original code by Patrik O. Hoyer (2004)
        """
        print('\nStart Projection:')
        print('---------------')
        # problem dimension
        N = len(s)

        if not non_negative:
            is_negative = s<0
            s = abs(s)

        # start by projecting the point to the sum constrained hyperplane
        v = (s + (k1 - sum(s))) / N

        # initialize zero_coeff
        zero_coeff = np.array([])

        j = 0
        while True:
            print('>> j#%i' % j)
            midpoint = (np.ones(N)*k1) / (N - len(zero_coeff))
            if len(zero_coeff)>0:
                midpoint[zero_coeff] = 0

            print('Lenght Z')
            print(len(zero_coeff))

            w = v - midpoint  # vector
            # print('w:')
            # print(w)
            a = sum(np.power(w,2)) # scalar
            b = 2.0 * (w.dot(v)) # scalar
            c = sum(np.power(v,2)) - k2 # scalar

            b2 = np.power(b, 2)
            ac = a * c # scalar
            delta = b2 - (4*ac)

            alpha = (-b + np.sqrt(delta)) / (2*a)
            print('alpha = %1.2f' % alpha)
            v = midpoint + alpha*w

            # check if all components of v are non-negative
            if (v >= 0).all():
                print("We've found our solution.")
                used_iters = j+1
                break

            j = j+1

            # set negative values to zero, subtract appropriate amout from rest
            zero_coeff = np.where(v<=0)[0]
            v[zero_coeff] = 0
            print('v Update:')
            print(v)
            print('Z:')
            print(zero_coeff)
            print('Lenght Z Update:')
            print(len(zero_coeff))

            if (len(zero_coeff)==N):
                print('Divisao por zero!')

            c = (k1 - sum(v)) / (N - len(zero_coeff))
            v = (v + c)
            v[zero_coeff] = 0

            if j>3: break
        # if binary flag not set, return signs to the solution
        if not non_negative:
            v = ((-2*is_negative) + 1) * v;

        # check for problems
        if np.max(abs(np.imag(v))) > 1e-10:
            print('Somehow got imaginary values.')


        return v

    def iter_solver(self, V, W, H, r, it,
                    sparse_W=0.0,
                    sparse_H=0.0,
                    step_W=1.0,
                    step_H=1.0):

        # data dimensions
        n = V.shape[0]
        m = V.shape[1]

        # make initial matrices have correct sparseness
        # before the first iteration.
        if it == 1:
            print('Initial Matrices...')
            # check sparseness on W:
            if sparse_W != 0:
                L1_W = np.sqrt(n) - ((np.sqrt(n) - 1) * sparse_W)
                for col in range(W.shape[1]):
                    W[:, col] = self.project_func(W[:, col], L1_W, 1, 1)

            if sparse_H != 0:
                L1_H = np.sqrt(m) - ((np.sqrt(m) - 1) * sparse_H)
                for row in range(H.shape[0]):
                    H[row, :] = self.project_func(H[row, :], L1_H, 1, 1)

        print('>> Start Updates')
        # ---------
        # Update W
        # ---------
        # IF SPARSENESS ON W
        if sparse_W != 0:
            dW = ((W.dot(H)) - V).dot(H.T) # gradient for W

            V_hat = W.dot(H)
            error = 0.5 * np.linalg.norm(V-V_hat, ord='fro') # calculate error

            # make sure we decrease the objective function
            while True:

                # take the step in the direction of the negative gradient
                W_new = W - step_W * dW

                # project each column of W according to the authors
                norm_vec = np.sqrt(sum(np.power(W_new, 2)))
                for col in range(W_new.shape[1]):
                    W_new[:, col] = self.project_func(W_new[:, col], \
                        L1_W * norm_vec[col], np.power(norm_vec[col], 2), 1)

                # calculate new error
                new_error = 0.5 * np.linalg(V - (W_new.dot(H)), ord='fro')

                # if the objective decrease we can continue
                if new_error <= error:
                    break
                # else, decrease stepsize and try again
                step_W = step_W / 2
                print('.')

                # check for convergence
                if step_W < 1e-200:
                    print('Algorithm converged.\n')
                    W = W_new
                    error = new_error
                    return W, H, error, step_W, step_H

            # slightly increase the step size
            step_W = step_W * 1.2
            W = W_new

        # IF W IS NOT SPARSE
        else:
            # update using standard NMF multiplicative rule
            print('W not sparse')
            W = W * (V.dot(H.T)) / W.dot(H.dot(H.T)) + self.eps

        # normalize columns of W

        # ---------
        # UPDATE H
        # ---------
        # IF SPARSENESS ON H
        if sparse_H != 0:
            dH = W.T.dot((W.dot(H)) - V) # gradient for H

            V_hat = W.dot(H)
            error = 0.5 * np.linalg.norm(V-V_hat, ord='fro') # calculate error

            # make sure we decrease the objective
            while True:

                # take step in the direction of the negative gradient
                H_new = H - step_H * dH

                # project each row of H according to the authors
                for row in range(H_new.shape[0]):
                    H_new[row, :] = self.project_func(H_new[row,:], L1_H, 1, 1)

                # calculate new error
                new_error = 0.5 * np.linalg(V - (W.dot(H_new)), ord='fro')

                # if the objective decrease we can continue
                if new_error <= error:
                    break
                # else, decrease stepsize and try again
                step_H = step_H / 2
                print('.')

                # check for convergence
                if step_H < 1e-200:
                    print('Algorithm converged.\n')
                    H = H_new
                    error = new_error
                    return W, H, error, step_W, step_H

            # slightly increase the stepsize
            step_H = step_H * 1.2
            H = H_new

        # IF H IS NOT SPARSE
        else:
            # update using standard NMF multiplicative rule
            print('H is not sparse.')
            H = H * (W.T.dot(V)) / W.T.dot(W.dot(H)) + self.eps

        return W, H, error, step_W, step_H
