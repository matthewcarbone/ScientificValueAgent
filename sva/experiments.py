from copy import deepcopy
from functools import lru_cache
from itertools import product
from joblib import delayed, Parallel
import json
from pathlib import Path
import sys
from warnings import warn

import numpy as np
from monty.json import MSONable
from scipy.spatial import distance_matrix
import torch
from tqdm import tqdm

from sva import __version__
from sva.utils import get_function_from_signature

# Used as a submodule
easybo_path = str(Path(__file__).absolute().parent / "EasyBO")
sys.path.append(easybo_path)
from easybo.gp import EasySingleTaskGPRegressor  # noqa
from easybo.bo import ask  # noqa
from easybo.logger import logging_mode  # noqa


def next_closest_raster_scan_point(
    proposed_points, observed_points, possible_coordinates, eps=1e-8
):
    """A helper function which determines the closest grid point for every
    proposed points, under the constraint that the proposed point is not
    present in the currently observed points, given possible coordinates.

    Parameters
    ----------
    proposed_points : array_like
        The proposed points. Should be of shape N x d, where d is the dimension
        of the space (e.g. 2-dimensional for a 2d raster). N is the number of
        proposed points (i.e. the batch size).
    observed_points : array_like
        Points that have been previously observed. N1 x d, where N1 is the
        number of previously observed points.
    possible_coordinates : array_like
        A grid of possible coordinates, options to choose from. N2 x d, where
        N2 is the number of coordinates on the grid.
    eps : float, optional
        The cutoff for determining that two points are the same, as computed
        by the L2 norm via scipy's ``distance_matrix``.

    Returns
    -------
    numpy.ndarray
        The new proposed points. REturns None if no new points were found.
    """

    assert proposed_points.shape[1] == observed_points.shape[1]
    assert proposed_points.shape[1] == possible_coordinates.shape[1]

    D2 = distance_matrix(observed_points, possible_coordinates) > eps
    D2 = np.all(D2, axis=0)

    actual_points = []
    for possible_point in proposed_points:
        p = possible_point.reshape(1, -1)
        D = distance_matrix(p, possible_coordinates).squeeze()
        argsorted = np.argsort(D)
        for index in argsorted:
            if D2[index]:
                actual_points.append(possible_coordinates[index])
                break

    if len(actual_points) == 0:
        return None

    return np.array(actual_points)


def oracle(X, truth, value, truth_kwargs=dict(), value_kwargs=dict()):
    """Converts observations to the value function.

    Parameters
    ----------
    X : array_like
        Input points.
    truth : callable
        The source of truth for the observation. Produces the observation from
        the input points.
    value : callable
        The value function, which maps the inputs and observations to the
        scalar value.
    truth_kwargs : dict, optional
        Keyword arguments for the truth function.
    value_kwargs : dict, optional
        Keyword arguments for the value function.

    Returns
    -------
    array_like
        The value of the points.
    """

    return value(X, truth(X, **truth_kwargs), **value_kwargs)


def get_valid_X(xmin, xmax, n_raster, ndim):
    valid_X = np.linspace(xmin, xmax, n_raster)
    gen = product(*[valid_X for _ in range(ndim)])
    return np.array([xx for xx in gen])


class Data(MSONable):
    """Data to be modeled by a Gaussian Process. Specifically this also
    includes the logic for computing the value function information given
    a source of truth. X are the input coordinates and Y is actually the
    scientific value function | X."""

    @classmethod
    def from_initial_conditions(
        cls,
        truth,
        value,
        seed,
        how,
        truth_kwargs=dict(),
        value_kwargs=dict(),
        xmin=0.0,
        xmax=1.0,
        points_per_dimension=3,
        ndim=2,
        n_raster=None,
    ):
        """Gets a ``Data`` object from a specified sampling of the space.

        Parameters
        ----------
        truth : callable
        value : callable
        seed : int
            The random seed to use for sampling the points.
        how : {"random", "grid"}
            The type of initial grid to use.
        truth_kwargs : dict, optional
        value_kwargs : dict, optional
        xmin : float, optional
            The minimum value to sample from on each dimension.
        xmax : float, optional
            The maximum value to sample from on each dimension.
        points_per_dimension : int, optional
            The number of points to sample per dimension.
        ndim : int, optional
            The number of dimensions.
        n_raster : int, optional
            If not None, this is the number of uniform points per dimension
            that are "allowed" to be sampled in any future experiment.
        """

        valid_X = None
        if n_raster is not None:
            valid_X = get_valid_X(xmin, xmax, n_raster, ndim)

        if how == "random":
            np.random.seed(seed)
            X = np.random.random(size=(points_per_dimension, ndim))
            X = (xmax - xmin) * X + xmin
        elif how == "grid":
            xgrid = np.linspace(xmin, xmax, points_per_dimension + 2)
            xgrid = xgrid[1:-1]
            X = np.array([xx for xx in product(*[xgrid for _ in range(ndim)])])
        else:
            raise ValueError(f"Unknown initial condition type {how}")

        if valid_X is not None:
            observed = np.ones(shape=(1, ndim)) * np.inf  # Dummy
            X = next_closest_raster_scan_point(X, observed, valid_X)

        Y = np.array(
            oracle(
                X,
                truth,
                value,
                truth_kwargs=truth_kwargs,
                value_kwargs=value_kwargs,
            )
        ).reshape(-1, 1)
        metadata = dict(xmin=xmin, xmax=xmax, seed=seed)
        return cls(
            truth,
            value,
            truth_kwargs,
            value_kwargs,
            X=X,
            Y=Y,
            valid_X=valid_X,
            metadata=metadata,
        )

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def X0(self):
        return self._X[: self._n_initial_points, :]

    @property
    def Y0(self):
        return self._Y[: self._n_initial_points, :]

    @property
    def N(self):
        assert self._X.shape[0] == self._Y.shape[0]
        return self._X.shape[0]

    @property
    def valid_X(self):
        return self._valid_X

    @property
    def metadata(self):
        return self._metadata

    @lru_cache()
    def get_full_grid(self, n=100):
        xmin = self._metadata["xmin"]
        xmax = self._metadata["xmax"]

        if isinstance(xmin, (float, int)) and isinstance(xmax, (float, int)):
            grid = np.linspace(xmin, xmax, n)
            gen = product(*[grid for _ in range(self._X.shape[1])])

        else:
            assert len(xmin) == len(xmax)
            grids = [np.linspace(xx, yy, n) for xx, yy in zip(xmin, xmax)]
            gen = product(*grids)

        return np.array([xx for xx in gen])

    def __init__(
        self,
        truth,
        value,
        truth_kwargs,
        value_kwargs,
        X,
        Y,
        n_initial_points=None,
        valid_X=None,
        metadata=dict(),
    ):
        self._truth = truth
        self._value = value
        self._truth_kwargs = truth_kwargs
        self._value_kwargs = value_kwargs
        self._X = X
        self._Y = Y
        if n_initial_points is None:
            self._n_initial_points = self.X.shape[0]
        else:
            self._n_initial_points = n_initial_points
        assert self.Y.shape[0] == self.X.shape[0]
        self._valid_X = valid_X
        self._metadata = metadata

    def append(self, X):
        X = X.reshape(-1, self._X.shape[1])
        self._X = np.concatenate([self._X, X], axis=0)
        self._Y = np.array(
            oracle(
                self._X,
                self._truth,
                self._value,
                self._truth_kwargs,
                self._value_kwargs,
            )
        ).reshape(-1, 1)


class UVData(Data):
    @classmethod
    def from_random(
        cls,
        truth,
        value,
        truth_kwargs,
        value_kwargs,
        xmin,
        xmax,
        seed=125,
        n=3,
    ):
        assert len(xmin) == len(xmax)
        ndim = len(xmin)

        np.random.seed(seed)
        X = np.random.random(size=(n, ndim))
        X = (xmax - xmin) * X + xmin

        Y = np.array(
            oracle(X, truth, value, truth_kwargs, value_kwargs)
        ).reshape(-1, 1)
        metadata = dict(xmin=xmin, xmax=xmax, seed=seed)
        return cls(
            truth,
            value,
            truth_kwargs,
            value_kwargs,
            X=X,
            Y=Y,
            valid_X=None,
            metadata=metadata,
        )


class Experiment(MSONable):
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, xx):
        assert isinstance(xx, str)
        self._name = xx

    @property
    def recorded_at(self):
        return self._recorded_at

    @property
    def recorded_X(self):
        return [self._data.X[:ii, :] for ii in self._recorded_at]

    @property
    def recorded_Yhat(self):
        return self._predictions

    @property
    def data(self):
        return self._data

    def _init_bounds(self):
        xmin = self._data._metadata["xmin"]
        xmax = self._data._metadata["xmax"]

        if isinstance(xmin, (int, float)) and isinstance(xmax, (int, float)):
            self._bounds = [[xmin, xmax] for _ in range(self._data.X.shape[1])]

        else:
            assert len(xmin) == len(xmax)
            self._bounds = [[x0, xf] for x0, xf in zip(xmin, xmax)]

    def __init__(
        self,
        data,
        aqf="MaxVar",
        aqf_kwargs=dict(),
        optimize_acqf_kwargs={"q": 1, "num_restarts": 5, "raw_samples": 20},
        experiment_seed=123,
        recorded_at=[],
        predictions=[],
        run_parameters=[],
        model_parameters=[],
        name=None,
    ):
        self._data = data
        self._aqf = aqf
        self._aqf_kwargs = aqf_kwargs
        self._optimize_acqf_kwargs = optimize_acqf_kwargs
        self._init_bounds()
        self._experiment_seed = experiment_seed
        self._recorded_at = recorded_at
        self._predictions = predictions
        self._run_parameters = run_parameters
        self._model_parameters = model_parameters
        self._name = name

    def run(
        self,
        save_at,
        pbar=False,
        return_self=False,
        production_mode=True,
        print_at_end=True,
        points_per_dimension_full_grid=100,
    ):
        """Runs the experiment.

        Parameters
        ----------
        save_at : array_like
            The points at which to record results.
        pbar : bool, optional
            If True, enables the progress bar when running.
        return_self : bool, optional
            Description
        n_experiments : int, optional
            Description
        production_mode : bool, optional
            Description
        print_at_end : bool, optional
            Description
        points_per_dimension_full_grid : int, optional
            If None, then we skip the entire step of predicting the value
            function on a dense grid. This can save a lot of space if this
            quantity is not of interest (and given the value function is just
            an auxiliary quantity, it usually is not).
        """

        path = None
        if self._name is not None:
            path = Path(self._name + ".json")
            if path.exists():
                print(str(path), "exists - continuing")
                return
            path = str(path)

        k = dict()
        if production_mode:
            k = dict(
                warning=False,
                error=False,
                success=False,
                info=False,
                debug=False,
            )

        run_parameters = {
            key: value
            for key, value in locals().items()
            if key not in ["pbar", "self", "return_self"]
        }
        self._run_parameters.append(run_parameters)

        if self._experiment_seed is not None:
            np.random.seed(self._experiment_seed)
            torch.manual_seed(self._experiment_seed)

        n_experiments = save_at[-1] + 1
        xmax = self.data.metadata["xmax"]
        xmin = self.data.metadata["xmin"]

        with logging_mode(**k):
            for ii in tqdm(range(n_experiments), disable=not pbar):

                n_dat = self._data.N

                if self._aqf == "Random":
                    if n_dat in save_at or ii == 0:
                        gp = EasySingleTaskGPRegressor(
                            train_x=self._data.X, train_y=self._data.Y
                        )
                        gp.train_()
                    next_point = np.random.random(
                        size=(1, self._data.X.shape[1])
                    )
                    next_point = (xmax - xmin) * next_point + xmin
                else:
                    gp = EasySingleTaskGPRegressor(
                        train_x=self._data.X, train_y=self._data.Y
                    )
                    gp.train_()
                    next_point = ask(
                        model=gp.model,
                        bounds=self._bounds,
                        acquisition_function=self._aqf,
                        acquisition_function_kwargs=self._aqf_kwargs
                        if self._aqf != "EI"
                        else dict(best_f=np.max(self._data.Y)),
                        optimize_acqf_kwargs=self._optimize_acqf_kwargs,
                    )

                if self._data.valid_X is not None:
                    next_point = next_closest_raster_scan_point(
                        next_point, self._data.X, self._data.valid_X
                    )

                if n_dat in save_at or ii == 0:
                    if points_per_dimension_full_grid is not None:
                        _N = points_per_dimension_full_grid
                        grid = self._data.get_full_grid(_N)
                        preds = gp.predict(grid=grid)
                        preds.pop("posterior")
                        preds.pop("mean+2std")
                        preds.pop("mean-2std")
                        self._predictions.append(preds)
                    self._recorded_at.append(n_dat)
                    p = str(gp._get_training_debug_information())
                    self._model_parameters.append(p)

                self._data.append(next_point)

        if path is not None:
            with open(path, "w") as f:
                json.dump(self.to_json(), f, indent=4)

        if print_at_end:
            print(f"Done: {self._name}", flush=True)
        if return_self:
            return self


def get_experiments(params):

    how = params["how"]

    value_signature = params["value_function_signature"]
    truth_signature = params["truth_function_signature"]
    value_name = value_signature.split(":")[1]
    truth_name = truth_signature.split(":")[1]

    value_function = get_function_from_signature(value_signature)
    truth_function = get_function_from_signature(truth_signature)

    from_initial_conditions_kwargs = dict(
        truth=truth_function,
        value=value_function,
        seed="set me",
        how=how,
        truth_kwargs=params.get("truth_function_kwargs", dict()),
        value_kwargs=params.get("value_function_kwargs", dict()),
        **params["from_initial_conditions_kwargs"],
    )

    # Still need to set `aqf`, `aqf_kwargs` and `name`
    experiment_fixed_kwargs = dict(
        experiment_seed="set me",
        name=None,
        **params["experiment_fixed_kwargs"],
    )

    # If how == "random", we sample over N x N different random combinations
    # of coordinate seeds and model/experiment seeds. If how == "grid", we just
    # sample over N x N total experiment seeds.
    np.random.seed(params["seed"])
    if how == "random":
        size = (params["total_jobs"], 2)
        seeds_arr = np.random.choice(range(int(1e6)), size=size, replace=False)
        coords_seeds = seeds_arr[:, 0].tolist()
        exp_seeds = seeds_arr[:, 1].tolist()
    else:
        size = params["total_jobs"]
        seeds_arr = np.random.choice(range(int(1e6)), size=size, replace=False)
        coords_seeds = [None for _ in range(size)]
        exp_seeds = seeds_arr.tolist()

    list_of_experiments = []

    for job in params["jobs"]:
        aqf = job["aqf"]
        aqf_kwargs = job.get("aqf_kwargs", dict())
        beta = aqf_kwargs.get("beta")
        if beta is None:
            aqf_name = aqf
        else:
            beta = int(beta)
            aqf_name = f"{aqf}{beta}"

        for (cseed, eseed) in zip(coords_seeds, exp_seeds):

            name = f"{value_name}-{truth_name}-{aqf_name}-{cseed}-{eseed}"

            experiment_kwargs = experiment_fixed_kwargs.copy()
            experiment_kwargs["aqf"] = aqf
            experiment_kwargs["aqf_kwargs"] = aqf_kwargs
            experiment_kwargs["experiment_seed"] = eseed
            experiment_kwargs["name"] = name

            ckwargs = from_initial_conditions_kwargs.copy()
            ckwargs["seed"] = cseed
            d0 = Data.from_initial_conditions(**ckwargs)

            exp = Experiment(data=d0, **experiment_kwargs)
            list_of_experiments.append(exp)

    names = [xx.name for xx in list_of_experiments]
    assert len(names) == len(set(names))  # Assert all names unique
    return list_of_experiments


def run_experiments(list_of_experiments, **kwargs):

    try:
        n_jobs = kwargs.pop("n_multiprocessing_jobs")
    except KeyError:
        warn("n_multiprocessing_jobs not set, setting to 1")
        n_jobs = 1

    try:
        N_total = kwargs.pop("N_total")
        N_save = kwargs.pop("N_save")
        log_scale_save = kwargs.pop("log_scale_save")
    except KeyError:
        warn(
            "Either of N_total, N_save or log_scale_save were not provided. "
            "Setting defaults of N_total=100, N_save=10, log_scale_save=False"
        )
        N_total = 100
        N_save = 10
        log_scale_save = False

    if log_scale_save:
        n0 = np.log10(N_total)
        save_at = np.unique(np.logspace(0, n0, N_save).astype(int))
    else:
        save_at = np.unique(np.linspace(1, N_total, N_save).astype(int))

    kwargs["save_at"] = save_at

    def _execute(exp):
        exp = deepcopy(exp)
        return exp.run(**kwargs)

    return Parallel(n_jobs=n_jobs)(
        delayed(_execute)(exp) for exp in list_of_experiments
    )


def execute(params):
    print(f"Value Agent version {__version__}")
    exps = get_experiments(params)
    print("Running", len(exps), "experiments")
    run_experiments(exps, **params["experiment_run_kwargs"])
