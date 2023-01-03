from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from sva.truth.common import sigmoid


# Don't change the keyword argument defaults, they actually should be
# treated as hardcoded...


def slow_linear_phase(x, y, x0=0.55, a=4.0):
    # y = 0.25 - 2x
    d = np.abs(y - (0.25 - 2.0 * x))
    # 2 * 0.4 =
    return 1.0 - sigmoid(d, x0=x0, a=a)


def slow_linear_phase_lines(yint=0.25, slope=-2.0):
    grid_main = np.linspace(0.0, 0.4, 100)
    phaseline_main = 0.8 - 2.0 * grid_main
    grid1 = np.linspace(0.0, 0.5, 100)
    phaseline1 = 1.0 - 2.0 * grid1
    grid2 = np.linspace(0.0, 0.3, 100)
    phaseline2 = 0.6 - 2.0 * grid2
    return grid1, phaseline1, grid_main, phaseline_main, grid2, phaseline2


def points_in_slow_linear_phase(X):
    total_points = X.shape[0]
    x = X[:, 0]
    y = X[:, 1]
    y_upper = 1.0 - 2.0 * x
    y_lower = 0.6 - 2.0 * x
    where = np.where((y < y_upper) & (y > y_lower))[0]
    L = len(where)
    return L / total_points


def corner_circle_phase(x, y, x0=0.5, a=30.0, loc_x=1.0, loc_y=1.0):
    # Distance from the top left corner
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x) ** 2 + (loc_y - y) ** 2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


def corner_circle_phase_lines(r1=0.47, r2=0.53, r_main=0.5):
    grid1 = np.linspace(1.0 - r1, 1.0, 100)
    phaseline1 = 1.0 - np.sqrt(r1**2 - (grid1 - 1.0) ** 2)
    grid_main = np.linspace(1.0 - r_main, 1.0, 100)
    phaseline_main = 1.0 - np.sqrt(r_main**2 - (grid_main - 1.0) ** 2)
    grid2 = np.linspace(1.0 - r2, 1.0, 100)
    phaseline2 = 1.0 - np.sqrt(r2**2 - (grid2 - 1.0) ** 2)
    return grid1, phaseline1, grid_main, phaseline_main, grid2, phaseline2


def points_in_corner_circle_phase(X):
    total_points = X.shape[0]
    x = X[:, 0]
    y = X[:, 1]
    y_upper = 1.0 - np.nan_to_num(np.sqrt(0.47**2 - (1.0 - x) ** 2))
    y_lower = 1.0 - np.nan_to_num(np.sqrt(0.53**2 - (1.0 - x) ** 2))
    where = np.where((y < y_upper) & (y > y_lower))[0]
    L = len(where)
    return L / total_points


def full_circle_phase(x, y, x0=0.15, a=50.0, loc_x=0.8, loc_y=0.25):
    # Distance from a point near the bottom right quadrant
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x) ** 2 + (loc_y - y) ** 2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


def full_circle_phase_lines(x0=0.8):
    # r^2 = 0.15^2 = (0.8 - x)^2 + (0.25 - y)^2
    # -sqrt(r^2 - (0.8 - x)^2) + 0.25 = y
    r = 0.15
    grid_main = np.linspace(x0 - r, x0 + r, 200)
    pl_main = 0.25 - np.nan_to_num(np.sqrt(r**2 - (0.8 - grid_main) ** 2))
    grid_main2 = np.linspace(x0 + r, x0 - r, 200)
    pl_main2 = 0.25 + np.nan_to_num(np.sqrt(r**2 - (0.8 - grid_main2) ** 2))
    grid_main = np.concatenate([grid_main, grid_main2])
    pl_main = np.concatenate([pl_main, pl_main2])

    r = 0.12
    g_1 = np.linspace(x0 - r, x0 + r, 200)
    pl_1_ = 0.25 - np.nan_to_num(np.sqrt(r**2 - (0.8 - g_1) ** 2))
    g_12 = np.linspace(x0 + r, x0 - r, 200)
    pl_1_2 = 0.25 + np.nan_to_num(np.sqrt(r**2 - (0.8 - g_12) ** 2))
    g_1 = np.concatenate([g_1, g_12])
    pl_1_2 = np.concatenate([pl_1_, pl_1_2])

    r = 0.18
    g_2 = np.linspace(x0 - r, x0 + r, 200)
    pl_2_ = 0.25 - np.nan_to_num(np.sqrt(r**2 - (0.8 - g_2) ** 2))
    g_22 = np.linspace(x0 + r, x0 - r, 200)
    pl_2_2 = 0.25 + np.nan_to_num(np.sqrt(r**2 - (0.8 - g_22) ** 2))
    g_2 = np.concatenate([g_2, g_22])
    pl_2_2 = np.concatenate([pl_2_, pl_2_2])

    return g_1, pl_1_2, grid_main, pl_main, g_2, pl_2_2


def points_in_full_circle_phase(X):
    total_points = X.shape[0]
    x = X[:, 0]
    y = X[:, 1]
    r1 = 0.18
    r2 = 0.12
    y_upper_1 = 0.25 + np.nan_to_num(np.sqrt(r1**2 - (x - 0.8) ** 2))
    y_lower_1 = 0.25 + np.nan_to_num(np.sqrt(r2**2 - (x - 0.8) ** 2))
    y_lower_2 = 0.25 - np.nan_to_num(np.sqrt(r1**2 - (x - 0.8) ** 2))
    y_upper_2 = 0.25 - np.nan_to_num(np.sqrt(r2**2 - (x - 0.8) ** 2))
    c1 = (y < y_upper_1) & (y > y_lower_1)
    c2 = (y < y_upper_2) & (y > y_lower_2)
    where = np.where(c1 | c2)[0]
    L = len(where)
    return L / total_points


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
    prop1 = corner_circle_phase(X[:, 0], X[:, 1])
    prop2 = slow_linear_phase(X[:, 0], X[:, 1])
    prop3 = full_circle_phase(X[:, 0], X[:, 1])
    total_prop = prop1 + prop2 + prop3
    total_prop[total_prop > 1.0] = 1.0
    prop4 = 1.0 - total_prop
    return np.array([prop1, prop2, prop3, prop4]).T @ pure_phases
