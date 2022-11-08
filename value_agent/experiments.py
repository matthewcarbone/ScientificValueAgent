from functools import lru_cache
from itertools import product
from joblib import delayed, Parallel
import json
from pathlib import Path
import sys

import numpy as np
from monty.json import MSONable
import torch
from tqdm import tqdm

from value_agent.dummy.grids import get_2d_grids
from value_agent.value import value_function, next_closest_raster_scan_point

# Used as a submodule
easybo_path = str(Path(__file__).absolute().parent / "EasyBO")
sys.path.append(easybo_path)
from easybo.gp import EasySingleTaskGPRegressor  # noqa
from easybo.bo import ask  # noqa
from easybo.logger import logging_mode  # noqa


def get_phase_plot_info(truth, **kwargs):
    grids = get_2d_grids()
    x1_grid = grids["x1"]
    x2_grid = grids["x2"]
    X, Y = np.meshgrid(x1_grid, x2_grid)
    Z = truth(X, Y, **kwargs)
    return x1_grid, x2_grid, Z


def oracle(X, truth, **kwargs):
    """Converts observations to the value function.

    Parameters
    ----------
    X : array_like
        Input points.
    truth : callable, optional
        The source of truth. Produces the observation from the input points.
    **kwargs
        Keyword arguments for the truth function.

    Returns
    -------
    array_like
        The value of the points.
    """

    return value_function(X, truth(X, **kwargs))


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
    def from_random(
        cls, truth, xmin=0.0, xmax=1.0, seed=125, n=3, ndim=2, n_raster=None
    ):
        """Gets a ``Data`` object from a random sampling of the space.

        Parameters
        ----------
        truth : TYPE
            Description
        xmin : float, optional
            Description
        xmax : float, optional
            Description
        seed : int, optional
            Description
        n : int, optional
            Description
        ndim : int, optional
            Description
        n_raster : None, optional
            Description
        """

        if n_raster is None:
            valid_X = None
        else:
            valid_X = get_valid_X(xmin, xmax, n_raster, ndim)

        np.random.seed(seed)
        X = np.random.random(size=(n, ndim))

        if valid_X is not None:
            observed = np.ones(shape=(1, ndim)) * np.inf  # Dummy
            X = next_closest_raster_scan_point(X, observed, valid_X)

        Y = np.array(oracle(X, truth)).reshape(-1, 1)
        metadata = dict(xmin=xmin, xmax=xmax, seed=seed)
        return cls(truth, X=X, Y=Y, valid_X=valid_X, metadata=metadata)

    @classmethod
    def from_grid(
        cls,
        truth,
        xmin=0.0,
        xmax=1.0,
        points_per_dimension=3,
        ndim=2,
        n_raster=None,
    ):
        """Summary

        Parameters
        ----------
        truth : TYPE
            Description
        xmin : float, optional
            Description
        xmax : float, optional
            Description
        n : int, optional
            Description
        ndim : int, optional
            Description
        n_raster : None, optional
            Description
        """

        if n_raster is None:
            valid_X = None
        else:
            valid_X = get_valid_X(xmin, xmax, n_raster, ndim)

        xgrid = np.linspace(xmin, xmax, points_per_dimension + 2)
        xgrid = xgrid[1:-1]
        X = np.array([xx for xx in product(*[xgrid for _ in range(ndim)])])

        if valid_X is not None:
            observed = np.ones(shape=(1, ndim)) * np.inf  # Dummy
            X = next_closest_raster_scan_point(X, observed, valid_X)

        Y = np.array(oracle(X, truth)).reshape(-1, 1)
        metadata = dict(xmin=xmin, xmax=xmax)
        return cls(truth, X=X, Y=Y, valid_X=valid_X, metadata=metadata)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def X0(self):
        return self._X[: self._nseed, :]

    @property
    def Y0(self):
        return self._Y[: self._nseed, :]

    @property
    def N(self):
        assert self._X.shape[0] == self._Y.shape[0]
        return self._X.shape[0]

    @property
    def valid_X(self):
        return self._valid_X

    @lru_cache()
    def get_full_grid(self, n=100):
        xmin = self._metadata["xmin"]
        xmax = self._metadata["xmax"]
        grid = np.linspace(xmin, xmax, n)
        gen = product(*[grid for _ in range(self._X.shape[1])])
        return np.array([xx for xx in gen])

    def __init__(self, truth, X, Y, nseed=None, valid_X=None, metadata=dict()):
        self._truth = truth
        self._X = X
        self._Y = Y
        if nseed is None:
            self._nseed = self.X.shape[0]
        else:
            self._nseed = nseed
        assert self.Y.shape[0] == self.X.shape[0]
        self._valid_X = valid_X
        self._metadata = metadata

    def append(self, X):
        X = X.reshape(-1, self._X.shape[1])
        self._X = np.concatenate([self._X, X], axis=0)
        self._Y = np.array(oracle(self._X, self._truth)).reshape(-1, 1)


class Experiment(MSONable):
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, xx):
        assert isinstance(xx, str)
        self._name = xx

    @property
    def recorded_X(self):
        return [self._data.X[:ii, :] for ii in self._recorded_at]

    @property
    def recorded_Yhat(self):
        return self._predictions

    @property
    def data(self):
        return self._data

    def __init__(
        self,
        data,
        aqf="MaxVar",
        aqf_kwargs=None,
        points_per_dimension_full_grid=100,
        experiment_seed=123,
        recorded_at=[],
        predictions=[],
        run_parameters=[],
        model_parameters=[],
        name=None,
        root=None,
    ):
        self._data = data
        self._aqf = aqf
        self._aqf_kwargs = aqf_kwargs if aqf_kwargs is not None else None
        self._bounds = [
            [self._data._metadata["xmin"], self._data._metadata["xmax"]]
            for _ in range(self._data.X.shape[1])
        ]
        self._points_per_dimension_full_grid = points_per_dimension_full_grid
        self._experiment_seed = experiment_seed
        self._recorded_at = recorded_at
        self._predictions = predictions
        self._run_parameters = run_parameters
        self._model_parameters = model_parameters
        self._name = name
        self._root = root

    def run(
        self,
        pbar=False,
        return_self=False,
        n_experiments=240,
        save_every=40,
        production_mode=True,
        print_at_end=True,
    ):
        """Runs the experiment.

        Parameters
        ----------
        pbar : bool, optional
            If True, enables the progress bar when running.
        """

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

        with logging_mode(**k):
            for ii in tqdm(range(n_experiments), disable=not pbar):

                n_dat = self._data.N

                if self._aqf == "Random":
                    if n_dat % save_every == 0 or ii == 0:
                        gp = EasySingleTaskGPRegressor(
                            train_x=self._data.X, train_y=self._data.Y
                        )
                        gp.train_()
                    next_point = np.random.random(
                        size=(1, self._data.X.shape[1])
                    )
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
                    )

                if self._data.valid_X is not None:
                    next_point = next_closest_raster_scan_point(
                        next_point, self._data.X, self._data.valid_X
                    )

                if n_dat % save_every == 0 or ii == 0:
                    _N = self._points_per_dimension_full_grid
                    grid = self._data.get_full_grid(_N)
                    preds = gp.predict(grid=grid)
                    preds.pop("posterior")
                    preds.pop("std")
                    preds.pop("mean+2std")
                    preds.pop("mean-2std")
                    self._recorded_at.append(n_dat)
                    self._predictions.append(preds)
                    p = str(gp._get_training_debug_information())
                    self._model_parameters.append(p)

                self._data.append(next_point)

        if self._root is not None and self._name is not None:
            path = Path(self._root)
            path.mkdir(exist_ok=True, parents=True)
            path = path / Path(self._name + ".json")
            with open(path, "w") as f:
                json.dump(self.to_json(), f, indent=4)

        if print_at_end:
            print(f"Done: {self._root}/{self._name}", flush=True)
        if return_self:
            return self


def run_experiments(list_of_experiments, n_jobs, **kwargs):
    return Parallel(n_jobs=n_jobs)(
        delayed(exp.run)(**kwargs)
        for exp in list_of_experiments
    )
