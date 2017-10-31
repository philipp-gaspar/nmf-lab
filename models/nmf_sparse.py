import sys
import numpy as np

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
    self.cost_function = cost_function
    self.info {'r': r,
               'V_dim_1': V.shape[0],
               'V_dim_2': V.shape[1],
               'V_type': str(V.__class__),
               'sparse_W': sW,
               'sparse_H': sH,
               'max_iter': max_iter,
               'cost_function': self.cost_function,
               'verbose': verbose}

    # Initialization of factor matrices
    if init != None:
        W = init[0].copy()
        H = init[1].copy()
        self.info['init'] = 'user_provided'
    else:
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
                    'n_components': r,
                    'iter': [],
                    'error': [],
                    'sparseness_W': [],
                    'sparseness_H': []}

    # Iteration Process
    for i in range(1, self.info['max_iter'] + 1):
        W, H, error = self.iter_solver(V, W, H, r, i, sW, sH)
        self.results['iter'].append(i)
        self.results['error'].append(error)

        # calculate sparseness on W and H


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
        self.default_max_iter = default_max_iter

    def initializer(self, W, H):
        # normalize rows of matrix H to l2 norm.
        norm_vec = row_norm(H, by_norm='2')
        H = H / norm_vec[:, None]

        return W, H

    def run(self, V, r,
            sparse_W=0.0,
            sparse_H=0.0,
            init=None,
            max_iter=self.default_max_iter,
            cost_function='frobenius',
            verbose=False):
        """
        """
        self.info = {'r': r,
                     'V_dim_1': V.shape[0],
                     'V_dim_2': V.shape[2],
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
            avg = np.sqrt(V.mean() / r)
            W = np.abs(avg * np.random.randn(V.shape[0], r))
            H = np.abs(avg * np.random.randn(r, V.shape[1]))
            self.info['init'] = 'normally-random'

        if verbose:
            print('[NMF] Running: ')
            print(json.dumps(self.info, indent=4, sort_keys=True))

        start = time.time()

        # Algorithm Initialization
        W, H = self.initializer(W, H)

        # Dictionary to save the results for each iteration
        self.results{'cost_function': cost_function,
                     'n_components': r,
                     'iter': [],
                     'error': [],
                     'sparseness_W': [],
                     'sparseness_H': [],
                     'step_size_W': [],
                     'step_size_H': []}

        # Iteration process
        for it in range(1, max_iter+1):
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
        self.final{'iterations': i,
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
        # Problem dimension
        N = len(s)

        # If non-negativivity flag not set, record signs and take abs
        if not non_negative:
            is_negative = s<0
            s = abs(s)

        # Start by projecting the point to the sum constraint hyperplane
        v = s + (k1 - sum(s)) / N

        # Initialize zerocoeff (initially, no elements are assumed zero)
        zero_coeff = []

        j = 0
        while True:
            # This does the proposed projction operator
            midpoint = (np.ones([N, 1]) * k1) / (N - len(zero_coeff))
            midpoint[zero_coeff] = 0
            w = v - midpoint
            a = sum(w*w)
            b = 2 * w.dot(v)
            c = sum(v*v) - k2
            alphap = (-b + np.real(np.sqrt(b*b - 4*a*c))) / 2*a
            v = alphap*w + v

            if v>=0:
                used_iters = j+1
                return v, used_iters





    def iter_solver(self, V, W, H, r, it,
                    sparse_W=0.0,
                    sparse_H=0.0,
                    step_W=1.0,
                    step_H=1.0):

        # data dimensions
        n = V.shape[0]
        m = V.shape[1]

        # NEED IMPLEMENTATION
        # make initial matrices have correct sparseness
        # before the first iteration.
        if it == 1:
            # check sparseness on W:
            if sparse_W != 0:
                L1a = np.sqrt(n) - (np.sqrt(n) - 1) * sparse_W
                # PROJECT
            if sparse_H != 0:
                L1s = np.sqrt(m) - (np.sqrt(m) - 1) * sparse_H
                # PROJECT

        # ---------
        # Update W
        # ---------
        # IF SPARSENESS ON W
        if sparseness_W != 0:
            dW = ((W.dot(H)) - V).dot(H.T) # gradient for W

            V_hat = W.dot(H)
            error = 0.5 * np.linalg.norm(V-V_hat, ord='fro') # calculate error

            # make sure we decrease the objective function
            while True:

                # take the step in the direction of the negative gradient
                W_new = W - step_W * dW

                # project each column of W according to the authors
                norm_vec = column_norm(W_new, by_norm='2')
                for col in range(W_new.shape[1]):
                    W_new[:, col] = project_func(W_new[:, col],
                                                 L1a * norm_vec[col],
                                                 np.power(norm_vec[col], 2),
                                                 1)
                # calculate new error
                new_error = 0.5 * np.linalg(V - (W_new.dot(H)), ord='fro')

                # if the objective decrease we can continue
                if new_error <= error:
                    break
                # else, decrease stepsize and try again
                stepsize_A = stepsize_A / 2
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
                    H_new[row, :] = (project_func(H_new[row,:].T,
                                                  Ls1,
                                                  1,
                                                  1)).T

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
            print('H id=s not sparse.')
            H = H * (W.T.dot(V)) / W.T.dot(W.dot(H)) + self.eps
