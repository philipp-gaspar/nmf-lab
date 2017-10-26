import numpy as np

__all__ = ['frobenius_norm']

def frobenius_norm(Y, Y_hat):
    """
    Calculate the Frobenius Norm between two matrices.

    Parameters
    ----------
    Y : numpy.array
        Original matrix.
    Y_hat : numpy.array
        Estimated matrix.

    Returns
    -------
    error : float
        Calculated error using the Frobenius Norm.
    """
    diff = Y - Y_hat
    error = 0.5 * np.linalg.norm(diff, ord='fro')

    return error

def kullback_leibler_divergence(Y, Y_hat):
    """
    Calculate the Kullback-Leibler divergence between two matrices.

    Parameters
    ----------
    Y : numpy.array
        Original matrix.
    Y_hat : numpy.array
        Estimated matrix.

    Returns
    -------
    error : float
        Calculated error using the KL divergence.
    """
    error = ((Y * np.log((Y/Y_hat) + self.eps)) - Y + Y_hat).sum(axis=None)

    return error

def itakura_saito_divergence(Y, Y_hat):
    """
    Calculate the Itakura-Saito divergence between two matrices.

    Parameters
    ----------
    Y : numpy.array
        Original matrix.
    Y_hat : numpy.array
        Estimated matrix.

    Returns
    -------
    error : float
        Calculated error using the IS divergence.
    """
    error = ((Y/(Y_hat + self.eps)) - np.log((Y/Y_hat + self.eps)+self.eps) - 1).sum(axis=None)

    return error
