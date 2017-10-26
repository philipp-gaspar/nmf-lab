import numpy as np
import scipy.sparse as sps
import math

__all__ = ['column_norm',
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

def scale_factor_matrices(A, B, by_norm='2'):
    """
    Column normalization of factor matrices.

    Scale the columns of A and B so that the columns of A
    have unit norms and the product A.dot(B.T) remains
    the same.

    Parameters
    ----------
    A : numpy.array, shape (i, j)
    B : numpy.array, shape (t, j)
    by_norm : string
        - '1' for l1-normalization
        - '2' for l2-normalization

    Returns
    -------
    A, B : normalized matrix pair
    """
    norm_vec = column_norm(A, by_norm=by_norm)
    A = A / norm_vec[None, :]
    B = B * norm_vec[None, :]

    return A, B
