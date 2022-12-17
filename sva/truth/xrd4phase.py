from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from sva.truth.common import sigmoid


def theta_phase(x, y, x0=1, a=10):
    # Radially symmetric about theta=0
    # x, y = scale_coords(x, y)
    angle = np.arctan2(y - 0.5, x - 0.5)
    return 1.0 - sigmoid(np.abs(angle), x0=x0, a=a)


def corner_phase(x, y, x0=0.25, a=20, loc_x=0.0, loc_y=1.0):
    # Distance from the top left corner
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x) ** 2 + (loc_y - y) ** 2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


def circle_phase(x, y, x0=0.5, a=20, loc_x=0.125, loc_y=0.125):
    # Distance from a point near the bottom right quadrant
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x) ** 2 + (loc_y - y) ** 2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


@cache
def _get_xrd_map():

    # Utterly horrendous practice here but for this it's fine...
    p = Path(__file__).absolute().parent / "xrd_map.xlsx"
    df = pd.read_excel(p, header=0, index_col=0)
    Y = df.to_numpy().T
    indexes = [0, 100, 150, 230]
    return Y[indexes, ::20]


def truth_xrd4phase(X):
    pure_phases = _get_xrd_map()
    # Gets the actual "value" of the observation
    prop2 = theta_phase(X[:, 0], X[:, 1], x0=0.5, a=5.0)
    prop1 = corner_phase(X[:, 0], X[:, 1], x0=0.5, a=30.0)
    prop3 = circle_phase(X[:, 0], X[:, 1], x0=0.05, a=50.0)
    total_prop = prop1 + prop2 + prop3
    total_prop[total_prop > 1.0] = 1.0
    prop4 = 1.0 - total_prop
    return np.array([prop1, prop2, prop3, prop4]).T @ pure_phases
