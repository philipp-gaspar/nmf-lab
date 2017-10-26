from numpy import linalg.norm

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
    error = 0.5 * linalg.norm(diff, ord='fro')

    return error
