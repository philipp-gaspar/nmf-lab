import numpy as np

__all__ = ['frobenius_norm',
           'kullback_leibler_divergence',
           'itakura_saito_divergence']

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
        Calculated error based on the Frobenius Norm
        (Squared Euclidean Distance).
    """
    error = 0.5 * np.power((Y - Y_hat), 2).sum(axis=None)

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
    eps = 1e-16

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(Y, Y_hat)
        ratio[ratio == np.inf] = 0
        ratio = np.nan_to_num(ratio)

    error = ((Y * np.log(ratio + eps)) - Y + Y_hat).sum(axis=None)

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
    eps = 1e-16
    error = ((Y/(Y_hat + eps)) - np.log((Y/Y_hat + eps) + eps) - 1).sum(axis=None)

    return error
