from sva.truth.xrd1dim import (
    truth_xrd1dim,
    residual_1d_phase_mse,
    residual_1d_phase_relative_mae,
)


def truth_xrd2dim(X):
    """Same as the truth in 1 dimension, except a two dimensional X array is
    provided, and the y-axis is ignored.

    Parameters
    ----------
    X : np.ndarray
        An N x 2 array of the input points.

    Returns
    -------
    np.ndarray
    """

    return truth_xrd1dim(X[:, 0])


def residual_2d_phase_mse(X):
    return residual_1d_phase_mse(X[:, 0])


def residual_2d_phase_relative_mae(X):
    return residual_1d_phase_relative_mae(X[:, 0])
