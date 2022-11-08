from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd
import yaml

from value_agent import experiments


def sigmoid(x, x0, a):
    return 1.0 / (1.0 + np.exp(-a * (x - x0)))


def _sine(x):
    return 0.25 * np.sin(2.0 * np.pi * x)


def mu_Gaussians(p, E=np.linspace(-1, 1, 100), x0=0.5, sd=0.05):
    """Returns a dummy "spectrum" which is just two Gaussian functions. The
    proportion of the two functions is goverened by ``p``.
    
    Parameters
    ----------
    p : float
        The proportion of the first phase.
    E : numpy.ndarray
        Energy grid.
    
    Returns
    -------
    numpy.ndarray
        The spectrum on the provided grid.
    """

    p2 = 1.0 - p
    return p * np.exp(-(x0 - E)**2 / sd) + p2 * np.exp(-(x0 + E)**2 / sd)


def phase_1_sine_on_2d_raster(x, y, x0=0.5, a=30.0):
    """Takes the y-distance between a sigmoid function and the provided point.
    """
    
    distance = y - _sine(x)
    return sigmoid(distance, x0, a)


def sine_on_2d_raster_observations(X):
    phase_1 = [phase_1_sine_on_2d_raster(*c) for c in X]
    return np.array([mu_Gaussians(p) for p in phase_1])


def theta_phase(x, y, x0=1, a=10):
    # Radially symmetric about theta=0
    # x, y = scale_coords(x, y)
    angle = np.arctan2(y - 0.5, x - 0.5)
    return 1.0 - sigmoid(np.abs(angle), x0=x0, a=a)


def corner_phase(x, y, x0=0.25, a=20, loc_x=0.0, loc_y=1.0):
    # Distance from the top left corner
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x)**2 + (loc_y - y)**2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


def circle_phase(x, y, x0=0.5, a=20, loc_x=0.125, loc_y=0.125):
    # Distance from a point near the bottom right quadrant
    # x, y = scale_coords(x, y)
    r = np.sqrt((loc_x - x)**2 + (loc_y - y)**2)
    return 1.0 - sigmoid(r, x0=x0, a=a)


# Utterly horrendous practice here but for this it's fine...
df = pd.read_excel('data/xrd_map.xlsx', header=0, index_col=0)
Y = df.to_numpy().T
indexes = [0, 100, 150, 230]
pure_phases = Y[indexes, ::20]


def truth_4phase(X):
    # Gets the actual "value" of the observation
    prop2 = theta_phase(X[:, 0], X[:, 1], x0=0.5, a=5.0)
    prop1 = corner_phase(X[:, 0], X[:, 1], x0=0.5, a=30.0)
    prop3 = circle_phase(X[:, 0], X[:, 1], x0=0.05, a=50.0)
    total_prop = prop1 + prop2 + prop3
    total_prop[total_prop > 1.0] = 1.0
    prop4 = 1.0 - total_prop
    return np.array([prop1, prop2, prop3, prop4]).T @ pure_phases


def get_random_experiments(params):

    jobs = params["jobs"].get("random_initial_seeds")
    if jobs is None:
        return

    experiment = params["experiment"]

    if experiment == "sine2phase":
        truth = sine_on_2d_raster_observations
    elif experiment == "xrd4phase":
        truth = truth_4phase
    else:
        raise ValueError(f"Invalid experiment: {experiment}")

    coordinates_fixed_kwargs = dict(
        truth=truth,
        xmin=0.0, xmax=1.0, n=3, ndim=2, n_raster=params["n_raster"],
        seed="set me"
    )

    # Still need to set `aqf`, `aqf_kwargs` and `name`
    experiment_fixed_kwargs = dict(
        points_per_dimension_full_grid=params["n_full_grid"],
        experiment_seed="set me",
        name=None,
        root=params["root"]
    )

    coordinate_seeds = list(range(*params["metadata"]["coords_seed_matrix"]))
    experiment_seeds = list(range(*params["metadata"]["exp_seed_matrix"]))

    list_of_experiments = []

    for job in jobs:
        aqf = job["aqf"]
        aqf_kwargs = job["aqf_kwargs"]
        if aqf_kwargs is None:
            aqf_kwargs = dict()
        beta = aqf_kwargs.get("beta")
        if beta is None:
            aqf_name = aqf
        else:
            beta = int(beta)
            aqf_name = f"{aqf}{beta}"

        for cseed in coordinate_seeds:
            for eseed in experiment_seeds:

                name = f"{experiment}-random-{aqf_name}-seed-{cseed}-{eseed}"

                experiment_kwargs = experiment_fixed_kwargs.copy()
                experiment_kwargs["aqf"] = aqf
                experiment_kwargs["aqf_kwargs"] = aqf_kwargs
                experiment_kwargs["experiment_seed"] = eseed
                experiment_kwargs["name"] = name

                coordinates_kwargs = coordinates_fixed_kwargs.copy()
                coordinates_kwargs["seed"] = cseed
                d0 = experiments.Data.from_random(**coordinates_kwargs)

                exp = experiments.Experiment(data=d0, **experiment_kwargs)
                list_of_experiments.append(exp)

    names = [xx.name for xx in list_of_experiments]
    assert len(names) == len(set(names))  # Assert all names unique
    return list_of_experiments


def get_grid_experiments(params):

    jobs = params["jobs"].get("initial_grid")
    if jobs is None:
        return

    experiment = params["experiment"]

    if experiment == "sine2phase":
        truth = sine_on_2d_raster_observations
    elif experiment == "xrd4phase":
        truth = truth_4phase
    else:
        raise ValueError(f"Invalid experiment: {experiment}")

    coordinates_fixed_kwargs = dict(
        truth=truth,
        xmin=0.0, xmax=1.0, points_per_dimension=3, ndim=2,
        n_raster=params["n_raster"],
    )

    experiment_fixed_kwargs = dict(    
        points_per_dimension_full_grid=params["n_full_grid"],
        experiment_seed="set me",
        name=None,
        root=params["root"]
    )

    experiment_seeds = list(range(*params["metadata"]["exp_seed_matrix"]))

    list_of_experiments = []

    for job in jobs:
        aqf = job["aqf"]
        aqf_kwargs = job["aqf_kwargs"]
        if aqf_kwargs is None:
            aqf_kwargs = dict()
        beta = aqf_kwargs.get("beta")
        if beta is None:
            aqf_name = aqf
        else:
            beta = int(beta)
            aqf_name = f"{aqf}{beta}"

        for eseed in experiment_seeds:

            name = f"{experiment}-grid-{aqf_name}-seed-x-{eseed}"

            experiment_kwargs = experiment_fixed_kwargs.copy()
            experiment_kwargs["aqf"] = aqf
            experiment_kwargs["aqf_kwargs"] = aqf_kwargs
            experiment_kwargs["experiment_seed"] = eseed
            experiment_kwargs["name"] = name

            coordinates_kwargs = coordinates_fixed_kwargs.copy()
            d0 = experiments.Data.from_grid(**coordinates_kwargs)

            exp = experiments.Experiment(data=d0, **experiment_kwargs)
            list_of_experiments.append(exp)

    names = [xx.name for xx in list_of_experiments]
    assert len(names) == len(set(names))  # Assert all names unique
    return list_of_experiments


if __name__ == '__main__':
    params = yaml.safe_load(open("jobs.yaml", "r"))
    copy2("jobs.yaml", Path(params["root"]) / "jobs.yaml")
    n_jobs = params["n_jobs"]
    exps = get_grid_experiments(params)
    exps.extend(get_random_experiments(params))
    print("Running", len(exps), "experiments")
    experiments.run_experiments(exps, n_jobs=n_jobs)
    
