from copy import deepcopy
from functools import cache
from itertools import product
from joblib import delayed, Parallel
import json
from pathlib import Path
from time import perf_counter
from warnings import warn

from botorch.acquisition.penalized import PenalizedAcquisitionFunction
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
import gpytorch
import numpy as np
from monty.json import MSONable
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from tqdm import tqdm

from sva import __version__
from sva.utils import get_function_from_signature


torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Timer:
    def __enter__(self):
        self._time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._time = perf_counter() - self._time

    @property
    def dt(self):
        return self._time


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


def get_allowed_X(xmin, xmax, n_raster, ndim):
    allowed_X = np.linspace(xmin, xmax, n_raster)
    gen = product(*[allowed_X for _ in range(ndim)])
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

        allowed_X = None
        if n_raster is not None:
            allowed_X = get_allowed_X(xmin, xmax, n_raster, ndim)

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

        if allowed_X is not None:
            observed = np.ones(shape=(1, ndim)) * np.inf  # Dummy
            X = next_closest_raster_scan_point(X, observed, allowed_X)

        N_points = X.shape[0]
        observations = truth(X, **truth_kwargs).reshape(N_points, -1)
        value_value = value(X, observations, **value_kwargs)
        Y = np.array(value_value).reshape(-1, 1)

        return cls(
            truth,
            value,
            truth_kwargs,
            value_kwargs,
            X=X,
            observations=observations,
            Y=Y,
            xmin=xmin,
            xmax=xmax,
            seed=seed,
            allowed_X=allowed_X,
        )

    @property
    def X(self):
        return self._X

    @property
    def X_MinMax_scaled(self):
        scaler = MinMaxScaler()
        return scaler.fit_transform(self.X)

    @property
    def Y(self):
        return self._Y

    @property
    def Y_Standard_scaled(self):
        scaler = StandardScaler()
        return scaler.fit_transform(self.Y)

    @property
    def X0(self):
        return self._X[: self._n_initial_points, :]

    @property
    def Y0(self):
        return self._Y[: self._n_initial_points, :]

    @property
    def observations(self):
        return self._observations

    @property
    def N(self):
        assert self._X.shape[0] == self._Y.shape[0]
        return self._X.shape[0]

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def seed(self):
        return self._seed

    @property
    def allowed_X(self):
        return self._allowed_X

    @cache
    def get_full_grid(self, n=100):
        xmin = self.xmin
        xmax = self.xmax

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
        observations,
        Y,
        xmin,
        xmax,
        seed,
        n_initial_points=None,
        allowed_X=None,
    ):
        self._truth = truth
        self._value = value
        self._truth_kwargs = truth_kwargs
        self._value_kwargs = value_kwargs
        self._X = X
        self._observations = observations
        self._Y = Y
        self._xmin = xmin
        self._xmax = xmax
        self._seed = seed
        if n_initial_points is None:
            self._n_initial_points = self.X.shape[0]
        else:
            self._n_initial_points = n_initial_points
        assert self.Y.shape[0] == self.X.shape[0]
        self._allowed_X = allowed_X

    def append(self, X):
        # Need to convert to numpy array here because of the way that Pytorch
        # deals with arrays of size (1, 1)...
        X = np.array(X).reshape(-1, self._X.shape[1])

        # Get this particular observation
        N_points = X.shape[0]
        observations = self._truth(X, **self._truth_kwargs)
        observations = observations.reshape(N_points, -1)

        # Append all observations
        # Yes this is probably expensive but for now it's fine!
        self._observations = np.concatenate(
            [self._observations, observations], axis=0
        )

        # Append the input positions
        self._X = np.concatenate([self._X, X], axis=0)

        # Now calculate the new value
        value_value = self._value(
            self._X, self._observations, **self._value_kwargs
        )
        self._Y = np.array(value_value).reshape(-1, 1)


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
        points_per_dimension=3,
        ndim=3,
    ):
        if isinstance(xmin, np.ndarray):
            assert isinstance(xmax, np.ndarray)
            assert len(xmin) == len(xmax)
            ndim = len(xmin)

        np.random.seed(seed)
        X = np.random.random(size=(points_per_dimension, ndim))
        X = (xmax - xmin) * X + xmin

        Y = np.array(
            oracle(X, truth, value, truth_kwargs, value_kwargs)
        ).reshape(-1, 1)
        return cls(
            truth,
            value,
            truth_kwargs,
            value_kwargs,
            X=X,
            Y=Y,
            xmin=xmin,
            xmax=xmax,
            seed=seed,
            allowed_X=None,
        )


class Experiment(MSONable):
    """Summary"""

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

    def get_acquisition_function(self):
        return get_function_from_signature(self._acqf_signature)

    def _init_bounds(self):
        xmin = self.data.xmin
        xmax = self.data.xmax

        if isinstance(xmin, (int, float)) and isinstance(xmax, (int, float)):
            self._bounds = [[xmin, xmax] for _ in range(self._data.X.shape[1])]

        else:
            assert len(xmin) == len(xmax)
            self._bounds = [[x0, xf] for x0, xf in zip(xmin, xmax)]

    def _set_path(self):
        self._path = None
        if self._name is not None:
            self._path = str(Path(self._name + ".json"))

    def _check_path_exists(self):
        if self._path is not None:
            if Path(self._path).exists():
                return True
        return False

    def _seed_numpy_torch_rng(self):
        if self._experiment_seed is not None:
            np.random.seed(self._experiment_seed)
            torch.manual_seed(self._experiment_seed)

    def __init__(
        self,
        data,
        acqf_signature,
        acqf_kwargs=dict(),
        kernel_signature="gpytorch.kernels:MaternKernel",
        kernel_kwargs={"nu": 2.5},
        optimize_acqf_kwargs={"q": 1, "num_restarts": 5, "raw_samples": 20},
        experiment_seed=123,
        record=None,
        scale_inputs_MinMax=True,
        scale_outputs_Standard=True,
        name=None,
        path=None,
    ):
        self._data = data
        self._acqf_signature = acqf_signature
        self._acqf_kwargs = acqf_kwargs
        self._kernel_signature = kernel_signature
        self._kernel_kwargs = kernel_kwargs
        self._optimize_acqf_kwargs = optimize_acqf_kwargs
        self._init_bounds()
        self._experiment_seed = experiment_seed
        if record is None:
            self._record = []
        else:
            self._record = record
        self._scale_inputs_MinMax = scale_inputs_MinMax
        self._scale_outputs_Standard = scale_outputs_Standard
        self._name = name
        self._set_path()

    def _get_train_X(self):
        return torch.tensor(self.data.X.copy()).to(DEVICE)

    def _get_train_Y(self):
        return torch.tensor(self.data.Y.copy()).to(DEVICE)

    def _get_transforms(self, d, m):
        input_transform = (
            Normalize(d, transform_on_eval=True)
            if self._scale_inputs_MinMax
            else None
        )
        output_transform = (
            Standardize(m) if self._scale_outputs_Standard else None
        )
        return input_transform, output_transform

    @staticmethod
    def get_model_hyperparameters(gp):
        d = dict()
        for p in gp.named_parameters():
            p0 = str(p[0])
            p1 = p[1].detach().numpy()
            d[p0] = p1
        return d

    @staticmethod
    def set_train(gp):
        gp.train()
        gp.likelihood.train()

    @staticmethod
    def set_eval(gp):
        gp.eval()
        gp.likelihood.eval()

    def _adjust_kwargs_for_EI(self, train_Y):
        kwargs = self._acqf_kwargs.copy()
        if "ExpectedImprovement" in self._acqf_signature:
            kwargs["best_f"] = train_Y.max().item()
        return kwargs

    def _fit_with_fit_gpytorch_mll(self, gp, fit_kwargs):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood=gp.likelihood, model=gp
        )
        self.set_train(gp)
        fit_gpytorch_mll(mll, **fit_kwargs)

    def _fit_with_Adam(self, gp, train_X, fit_kwargs):
        losses = []
        gp.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood=gp.likelihood, model=gp
        ).to(train_X)
        self.set_train(gp)
        lr = fit_kwargs.get("lr", 0.05)
        optimizer = torch.optim.Adam(gp.parameters(), lr=lr)
        n_train = fit_kwargs.get("n_train", 200)
        for _ in range(n_train):
            optimizer.zero_grad()
            output = gp(train_X)
            loss = -mll(output, gp.train_targets)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        return losses

    def _ask(self, gp, acquisition_function_kwargs):
        bounds = torch.tensor(self._bounds).float().reshape(-1, 2).T
        if self._acqf_signature.lower() == "random":
            value = torch.tensor([0])
            dims = bounds.shape[1]
            sampled = torch.FloatTensor(np.random.random(size=(dims, 1)))
            # sampled is 2 x 1
            _max = bounds[1, :].reshape(dims, 1)
            _min = bounds[0, :].reshape(dims, 1)
            delta = _max - _min
            sampled = sampled * delta + _min
            return sampled, value

        _acqf = self.get_acquisition_function()
        acquisition_function = _acqf(gp, **acquisition_function_kwargs)
        return optimize_acqf(
            acquisition_function,
            bounds=bounds,
            **self._optimize_acqf_kwargs,
        )

    def _predict(self, gp, N_grid):
        grid = torch.tensor(self._data.get_full_grid(N_grid))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = gp.posterior(grid, observation_noise=True)
        mu = posterior.mean.detach().numpy().squeeze()
        sd = np.sqrt(posterior.variance.detach().numpy().squeeze())
        return mu, sd

    def run(
        self,
        max_n_dat,
        fit_with_Adam=False,
        pbar=False,
        return_self=False,
        print_at_end=True,
        model_kwargs=dict(),
        fit_via_BoTorch_kwargs=dict(),
        fit_via_Adam_kwargs=dict(),
        record_gp_every=0,
        points_per_dimension_full_grid=None,
    ):
        """Runs the experiment.

        Parameters
        ----------
        max_n_dat : int
            Description
        fit_with_Adam : bool, optional
            Description
        pbar : bool, optional
            If True, enables the progress bar when running.
        return_self : bool, optional
            Description
        print_at_end : bool, optional
            Description
        model_kwargs : dict, optional
            Description
        fit_kwargs : dict, optional
            Description
        record_gp_every : int, optional
            Description
        points_per_dimension_full_grid : int, optional
            If None, then we skip the entire step of predicting the value
            function on a dense grid. This can save a lot of space if this
            quantity is not of interest (and given the value function is just
            an auxiliary quantity, it usually is not).
        """

        if record_gp_every > 0 and points_per_dimension_full_grid is None:
            warn(
                "record_gp_every > 0 but no points_per_dimension_full_grid "
                "is set: no GP data will be saved"
            )

        self._seed_numpy_torch_rng()

        # Get some key objects for the GP and Bayesian Optimization
        # Underscores indicate that these are actually the objects, not the
        # initialized values
        _kernel = get_function_from_signature(self._kernel_signature)

        if self._check_path_exists():
            print(f"{self._path} exists: continuing")
            return

        n_experiments = max_n_dat - self.data.N

        for ii in tqdm(range(n_experiments), disable=not pbar):
            n_dat = self.data.N

            if n_dat >= max_n_dat:
                break

            log = dict(iteration=ii, N=n_dat)

            # Get the data, handles copying data.X and data.Y as well as
            # doing all of the scaling
            train_X = self._get_train_X()
            train_Y = self._get_train_Y()

            # Transform information
            d = train_X.shape[1]
            m = train_Y.shape[1]
            i_transform, o_transform = self._get_transforms(d, m)

            # Initialize the GP
            mean_prior = gpytorch.means.ConstantMean()
            kernel = gpytorch.kernels.ScaleKernel(
                _kernel(**self._kernel_kwargs)
            )
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                likelihood=likelihood,
                mean_module=mean_prior,
                covar_module=kernel,
                input_transform=i_transform,
                outcome_transform=o_transform,
                **model_kwargs,
            )

            # Fit the GP
            losses = None
            with Timer() as timer:
                if not fit_with_Adam:
                    try:
                        self._fit_with_fit_gpytorch_mll(
                            gp, fit_via_BoTorch_kwargs
                        )
                    except ModelFittingError:
                        losses = self._fit_with_Adam(
                            gp, train_X, fit_via_Adam_kwargs
                        )
                else:
                    losses = self._fit_with_Adam(
                        gp, train_X, fit_via_Adam_kwargs
                    )
            log["losses"] = losses
            log["hyperparameters"] = self.get_model_hyperparameters(gp)
            log["dt"] = timer.dt

            # Bayesian Optimization
            self.set_eval(gp)
            kwargs = self._adjust_kwargs_for_EI(train_Y)
            next_point, value = self._ask(gp, kwargs)
            log["next_point"] = next_point.detach().numpy()
            log["acquisition_function_value"] = value.detach().numpy()

            if self._data.allowed_X is not None:
                next_point = next_closest_raster_scan_point(
                    next_point, self._data.X, self._data.allowed_X
                )

            # Save GP predictions on a dense grid if specified
            if record_gp_every == 0:
                pass
            elif points_per_dimension_full_grid is not None and (
                n_dat % record_gp_every == 0
                or ii == 0
                or n_dat == max_n_dat - 1
            ):
                mu, sd = self._predict(gp, points_per_dimension_full_grid)
                log["mu"] = mu
                log["sd"] = sd

            # Ready the next experiment
            self._data.append(next_point)
            self._record.append(log)

        if self._path is not None:
            with open(self._path, "w") as f:
                json.dump(self.to_json(), f, indent=4)

        if print_at_end:
            print(f"Done: {self._name}", flush=True)

        if return_self:
            return self


class PenaltyModule(torch.nn.Module):
    def __init__(self, slope=1.0):
        super().__init__()
        self._slope = slope

    def forward(self, x):
        # Calculate the element-wise penalty
        x_copy = x.clone()
        x1 = torch.abs(x_copy[..., 1].clone())
        x_copy[..., 1] = x1
        sigma = x_copy.sum(axis=-1)  # Sum over the last dimension
        penalty = self._slope * sigma
        penalty[sigma <= 40] = 0

        # Handle the situation where any of the q's does not satisfy
        # the constraint. We just sum over that axis now for simplicity
        return penalty.sum(axis=-1)


class UVExperiment(Experiment):
    def _ask(self, gp, acquisition_function_kwargs):
        _acqf = self.get_acquisition_function()
        acquisition_function = _acqf(gp, **acquisition_function_kwargs)
        penalized_acquisition_function = PenalizedAcquisitionFunction(
            acquisition_function,
            penalty_func=PenaltyModule(),
            regularization_parameter=100.0,
        )
        bounds = torch.tensor(self._bounds).float().reshape(-1, 2).T
        return optimize_acqf(
            penalized_acquisition_function,
            bounds=bounds,
            **self._optimize_acqf_kwargs,
        )


def set_truth_info(truth_signature):
    # Set the type of class we're dealing with
    if "truth_uv" in truth_signature:
        data_klass = UVData
        experiment_klass = UVExperiment
    else:
        data_klass = Data
        experiment_klass = Experiment

    # Set the pieces that depend only on the type of truth
    data_kwargs = {}
    if (
        "truth_sine2phase" in truth_signature
        or "truth_xrd4phase" in truth_signature
        or "truth_linear2phase" in truth_signature
    ):
        # This experiment is two-dimensional between 0 and 1 on both axes
        data_kwargs["xmin"] = 0.0
        data_kwargs["xmax"] = 1.0
        data_kwargs["ndim"] = 2

    elif "truth_xrd1dim" in truth_signature:
        # This experiment is one-dimensional between 0 and 100 on the only axis
        data_kwargs["xmin"] = 0.0
        data_kwargs["xmax"] = 100.0
        data_kwargs["ndim"] = 1

    elif "truth_uv" in truth_signature:
        # This experiment is special and has special boundaries
        data_kwargs["xmin"] = np.array([1.0, -16.0, 2.0])
        data_kwargs["xmax"] = np.array([16.0, 16.0, 16.0])
        data_kwargs["ndim"] = 3

    elif "truth_bto" in truth_signature:
        data_kwargs["xmin"] = 150.0
        data_kwargs["xmax"] = 445.0
        data_kwargs["ndim"] = 1

    else:
        raise ValueError(f"Unknown truth signature: {truth_signature}")

    return data_klass, experiment_klass, data_kwargs


def get_experiments(params):
    how = params["how"]

    value_signature = params["value_function_signature"]
    truth_signature = params["truth_function_signature"]

    data_klass, experiment_klass, data_kwargs = set_truth_info(truth_signature)

    value_name = value_signature.split(":")[1]
    truth_name = truth_signature.split(":")[1]

    value_function = get_function_from_signature(value_signature)
    truth_function = get_function_from_signature(truth_signature)

    data_kwargs = dict(
        truth=truth_function,
        value=value_function,
        seed="set me",
        how=how,
        truth_kwargs=params.get("truth_function_kwargs", dict()),
        value_kwargs=params.get("value_function_kwargs", dict()),
        **params["data_kwargs"],
        **data_kwargs,
    )

    # Still need to set `acqf`, `acqf_kwargs` and `name`
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
        acqf_signature = job["acqf_signature"]
        acqf_kwargs = job.get("acqf_kwargs", dict())
        beta = acqf_kwargs.get("beta")

        acqf_name = acqf_signature.split(":")[1]
        if beta is not None:
            beta = int(beta)
            acqf_name = f"{acqf_name}{beta}"

        for cseed, eseed in zip(coords_seeds, exp_seeds):
            name = f"{value_name}-{truth_name}-{acqf_name}-{cseed}-{eseed}"

            experiment_kwargs = experiment_fixed_kwargs.copy()
            experiment_kwargs["acqf_signature"] = acqf_signature
            experiment_kwargs["acqf_kwargs"] = acqf_kwargs
            experiment_kwargs["experiment_seed"] = eseed
            experiment_kwargs["name"] = name

            ckwargs = data_kwargs.copy()
            ckwargs["seed"] = cseed
            d0 = data_klass.from_initial_conditions(**ckwargs)

            exp = experiment_klass(data=d0, **experiment_kwargs)
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
