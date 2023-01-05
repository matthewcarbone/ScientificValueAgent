import numpy as np

from sva.truth.common import sigmoid, mu_Gaussians


def phase_1_linear_on_2d_raster(x, y, x0=0.5, a=30.0):
    """Takes the y-distance between a sigmoid function and the provided
    point."""

    distance = y + 1.5 * (x - 0.5)
    return sigmoid(distance, x0, a)


def truth_linear2phase(X):
    phase_1 = [phase_1_linear_on_2d_raster(*c) for c in X]
    return np.array([mu_Gaussians(p) for p in phase_1])
