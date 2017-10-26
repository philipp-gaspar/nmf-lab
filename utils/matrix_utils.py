import numpy as np
import scipy.sparse as sps

__all__ = ['column_norm',
           'row_norm',
           'scale_factor_matrices']

def column_norm(X, by_norm='2'):
    """
    Compute the norms of each column of a given matrix.

    Parameters
    ----------
    X : numpy.array or scipy.sparce matrix
        Array to be normalized.

    by_norm : string
        - '1' for l1-norm
        - '2' for l2-norm

    Returns
    -------
    norm_vec: numpy.array
        Array with normalization factors for each column.
    """
    if sps.issparse(X):
        if by_norm == '2':
            norm_vec = np.sqrt(X.multiply(X).sum(axis=0))
        elif by_norm == '1':
            norm_vec = X.sum(axis=0)
        return np.asarray(norm_vec)[0]
    else:
        if by_norm == '2':
            norm_vec = np.sqrt(np.sum(X * X, axis=0))
        elif by_norm == '1':
            norm_vec = np.sum(X, axis=0)
        return norm_vec

def row_norm(X, by_norm='2'):
    """
    Compute the norms of each row of a given matrix.

    Parameters
    ----------
    X : numpy.array or scipy.sparce matrix
        Array of data to be normalized.
    by_norm : string
        - '1' for l1-norm
        - '2' for l2-norm

    Returns
    -------
    norm_vec : numpy.array
        Array with normalization factors for each row.
    """
    if sps.issparse(X):
        if by_norm == '2':
            norm_vec = np.sqrt(X.multiply(X).sum(axis=1))
        elif by_norm == '1':
            norm_vec = X.sum(axis=1)
        return np.asarray(norm_vec)[0]
    else:
        if by_norm == '2':
            norm_vec = np.sqrt(np.sum(X * X, axis=1))
        elif by_norm == '1':
            norm_vec = np.sum(X, axis=1)
        return norm_vec

def scale_factor_matrices(W, H, by_norm):
    """
    Column normalization of factor matrices.

    Scale the columns of W and H so that the columns of W
    have unit norms and the product W.dot(H) remains
    the same.

    Parameters
    ----------
    W : numpy.array, shape (n, r)
    H : numpy.array, shape (r, m)
    by_norm : string
        - '1' for l1-normalization
        - '2' for l2-normalization

    Returns
    -------
    W, H : normalized matrix pair
    """
    norm_vec = column_norm(W, by_norm=by_norm)
    W = W / norm_vec[None, :]
    H = H * norm_vec[:, None]

    return W, H
