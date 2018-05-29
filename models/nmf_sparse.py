import sys
import numpy as np

import json
import time

import pickle

from ..utils.matrix_utils import column_norm, scale_factor_matrices
from ..utils.metrics import frobenius_norm, \
                            kullback_leibler_divergence, \
                            itakura_saito_divergence

class NMF_SPARSE(object):
    """
    Base class for Sparse NMF algorithms
    """

    def __init__(self):
        # NMF_SPARSE is a Base class and cannot be instantiated
        raise NotImplementedError()

    def run(self, V, r, alpha, sparse_W,
            norm='1', init=None, max_iter=100, verbose=False):
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
        sparse_W : boolean
            Apply sparsity on W matrix.

        Optionals
        ---------
        norm : str
            - '1' for L1 normalization
            - '2' for L2 normalization
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
                     'sparse_W': sparse_W,
                     'norm': norm,
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
        W, H = self.initializer(W, H, norm)

        # Dictionary to save the results for each iteration
        self.results = {'n_components': r,
                        'alpha': alpha,
                        'sparse_W': sparse_W,
                        'norm': norm,
                        'iter': [],
                        'original_error': [],
                        'sparse_error': [],
                        'total_error': [],
                        'sparseness_W': []}

        # Iteration process
        for it in range(1, max_iter+1):
            W, H, errors_dict = self.iter_solver(V, W, H,
                                                 alpha, sparse_W, norm)
            self.results['iter'].append(it)
            self.results['original_error'].append(errors_dict['original_error'])
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

    def run_repeat(self, V, r, alpha, sparse_W, norm, num_trials,
                   max_iter=100, verbose=False,
                   save_init=False, file_name=None):
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
        sparse_W : boolean
            Apply sparsity on W matrix.
        norm : str
            - 'L1' normalization after each iteration
            - 'L2' normalization after each iteration
        num_trials : int
            Number of different trials/initializations for the NMF algorithm.

        Optionals
        ---------
        max_iter : int
            Maximum number of iterations.
        verbose : boolean
            Verbose variable.
        save_init : boolean
            Save the model for each initialization.
        file_name : str
            Name of the model for each initialization.
            Only necessary if 'save_init' equals True.

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
            actual_model = self.run(V, r, alpha, sparse_W, norm,
                                    init=None,
                                    max_iter=max_iter,
                                    verbose=False)
            if trial == 0:
                best_model = actual_model
            else:
                if actual_model['total_error'][-1] < best_model['total_error'][-1]:
                    best_model = actual_model

            if save_init:
                name = file_name + '_init%i.pkl' % (trial+1)
                pickle.dump(actual_model, open(name, 'wb'))

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

    def iter_solver(self, V, W, H, alpha, sparse_W, norm):
        raise NotImplementedError

    def initializer(self, W, H, norm):
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

    def initializer(self, W, H, norm):
        W, H = scale_factor_matrices(W, H, by_norm=norm)

        return W, H

    def iter_solver(self, V, W, H, alpha, sparse_W, norm):
        # Update H
        # - This is the original multiplicative update rule, except for
        #   the alpha constant in the denominator
        numerator = H * (W.T.dot(V))
        denominator = W.T.dot(W.dot(H)) + alpha
        H = numerator / np.maximum(denominator, self.eps)

        # Update W
        if sparse_W:
            # - Well-Done NMF (With sparsity on W)
            # preallocate matrix of ones
            # which is a square matrix (n x n)
            ones = np.ones([V.shape[0], V.shape[0]])

            HHT = H.dot(H.T)
            numerator = W * V.dot(H.T) + W * ones.dot(W.dot(HHT) * W)
            denominator = W.dot(HHT) + W * ones.dot(V.dot(H.T) * W)
            W = numerator / np.maximum(denominator, self.eps)

            # normalize columns of W
            norm_vec = column_norm(W, by_norm=norm)
            W = W / norm_vec[None, :]
        else:
            # - Half-Baked NMF (Without sparsity on W)
            numerator = W * (V.dot(H.T))
            denominator = W.dot(H.dot(H.T)) + self.eps
            W = numerator / denominator

            # normalize and rescale W and H
            W, H = scale_factor_matrices(W, H, by_norm=norm)

        # Calculate errors
        V_hat = W.dot(H) # reconstruction
        original_error = frobenius_norm(V, V_hat)
        sparse_error = alpha * np.sum(H)
        total_error = original_error + sparse_error

        # create a dictonary with errors
        errors_dict = {'original_error': original_error,
                       'sparse_error': sparse_error,
                       'total_error': total_error}

        return W, H, errors_dict

class nmf_sparse_kl(NMF_SPARSE):
    """
    Implements NMF using the normalized Kullback-Leibler Divergence [1]
    with sparsity constraints proposed in [2, 3].

    NMF factorization goes as:
    V = WH; s. t. W>=0, H>=0

    The cost function is:
    D(V|W*H) + alpha * np.sum(H)

    REFS:
    [1]: "Algorithms for Non-Negative Matrix Factorization", D. Lee
    and S. Seung, NIPS (2001)
    [2]: "Sparse Coding and NMF", J. Eggert and E. Korner,
    Neural Networks (2004)
    [3]: "Speech Separation using Non-negative Features and Sparse
    Non-negative Matrix Factorization", M. Schmidt, Tech. Report (2007)
    """
    def __init__(self):
        self.eps = 1e-16

    def initializer(self, W, H, norm):
        W, H = scale_factor_matrices(W, H, by_norm=norm)

        return W, H

    def iter_solver(self, V, W, H, alpha, sparse_W, norm):
        # preallocate matrix of ones
        ones_n = np.ones([V.shape[0], V.shape[0]])
        ones_m = np.ones([V.shape[0], V.shape[1]])

        # Update H
        # - This is the original update rule, except for
        #   the alpha constant in the denominator
        WH = W.dot(H)
        numerator = H * W.T.dot(V/WH)
        denominator = W.T.dot(ones_m) + alpha
        H = numerator / np.maximum(denominator, self.eps)

        # Update W
        if sparse_W:
            # - Well-Done NMF (With sparsity on W)
            R = V / W.dot(H)
            aux_1 = W * ones_n.dot(ones_m.dot(H.T) * W)
            numerator = W * (R.dot(H.T) + aux_1)
            aux_2 = W * ones_n.dot(R.dot(H.T) * W)
            denominator = ones_m.dot(H.T) + aux_2
            W = numerator / np.maximum(denominator, self.eps)

            # normalize and rescale W and H
            W, H = scale_factor_matrices(W, H, by_norm=norm)

        else:
            # - Half-Baked NMF (Without sparsity on W)
            WH = W.dot(H)
            numerator = W * ((V/WH).dot(H.T))
            denominator = ones_m.dot(H.T)
            W = numerator / np.maximum(denominator, self.eps)

            # normalize and rescale W and H
            W, H = scale_factor_matrices(W, H, by_norm=norm)

        # Calculate errors
        V_hat = W.dot(H) # reconstruction
        original_error = kullback_leibler_divergence(V, V_hat)
        sparse_error = alpha * np.sum(H)
        total_error = original_error + sparse_error

        # create dictonary with errors
        errors_dict = {'original_error': original_error,
                       'sparse_error': sparse_error,
                       'total_error': total_error}

        return W, H, errors_dict

class nmf_sparse_is(NMF_SPARSE):
    """
    Implements NMF using the Itakura-Saito Divergence [1]
    with sparsity constraints proposed in [2].

    NMF factorization goes as:
    V = WH; s. t. W>=0, H>=0

    The cost function is:
    D(V|W*H) + alpha * np.sum(H)

    REFS:
    [1]: "Sparse NMF - half-baked or well done?", J. Le Roux,
    F. Weninger and J. R. Hershey, Techinical Report (2015)
    [2] "Sparse Coding and NMF", J. Eggert and E. Korner,
    Neural Networks (2004)
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

        # Update H
        WH = W.dot(H)
        numerator = H * W.T.dot(V * np.power(WH, -2))
        denominator = W.T.dot(np.power(WH, -1)) + alpha
        H = numerator / np.maximum(denominator, self.eps)

        # Update W
        WH = W.dot(H)
        aux_1 = W * ones.dot(np.power(WH, -1).dot(H.T))
        numerator = W * ( (V / np.power(WH, 2)).dot(H.T) + aux_1 )
        aux_2 = W * ones.dot(((V / np.power(WH, 2)).dot(H.T)) * W)
        denominator = np.power(WH, -1).dot(H.T) + aux_2
        W = numerator / np.maximum(denominator, self.eps)

        # normalize columns of W
        norm_vec = column_norm(W, by_norm='1')
        W = W / norm_vec[None, :]

        # Calculate errors
        V_hat = W.dot(H) # reconstruction
        error = itakura_saito_divergence(V, V_hat)
        sparse_error = np.sum(H)
        total_error = error + (alpha * sparse_error)

        # create dictonary with errors
        errors_dict = {'error': error,
                       'sparse_error': sparse_error,
                       'total_error': total_error}

        return W, H, errors_dict
