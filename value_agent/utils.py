import numpy as np


def sigmoid(x, x0, a):
    return 1.0 / (1.0 + np.exp(-a * (x - x0)))
