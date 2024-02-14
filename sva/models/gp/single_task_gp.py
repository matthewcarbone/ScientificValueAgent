"""Module containing helpers for single task GPs written in gpytorch and
botorch."""

from copy import deepcopy

import gpytorch
import numpy as np
import torch
from attrs import define, field, validators
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from monty.json import MSONable

from sva.models.gp.common import SaveLoadMixin, set_eval_, set_train_
from sva.utils import Timer


def _get_model(
    X,
    Y,
    train_Yvar=None,
    likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    mean_module=gpytorch.means.ConstantMean(),
    covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),
    transform_input=True,
    transform_output=True,
    **model_kwargs,
):
    """A lightweight helper function that returns a botorch SingleTaskGP or
    FixedNoiseGP."""

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)

    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y)

    input_transform = (
        Normalize(X.shape[1], transform_on_eval=True)
        if transform_input
        else None
    )
    outcome_transform = Standardize(Y.shape[1]) if transform_output else None

    if train_Yvar is None:
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            **model_kwargs,
        )

    else:
        if not isinstance(train_Yvar, torch.Tensor):
            train_Yvar = torch.tensor(train_Yvar)

        # Likelihood argument is ignored here
        model = FixedNoiseGP(
            train_X=X,
            train_Y=Y,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            **model_kwargs,
        )

    return deepcopy(model)


def fit_gp_gpytorch_mll_(gp, **fit_kwargs):
    """Fits a provided GP model using the fit_gpytorch_mll method from
    botorch.

    Returns
    -------
    dict
        A dictionary containing training metadata.
    """

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood=gp.likelihood, model=gp
    )
    set_train_(gp)

    with Timer() as timer:
        fit_gpytorch_mll(mll, **fit_kwargs)

    return {"elapsed": timer.dt}


def fit_gp_Adam_(gp, X, lr=0.05, n_train=200):
    """Fits a provided GP model using the Adam optimizer.

    Returns
    -------
    dict
        A dictionary containing training metadata.
    """

    losses = []
    gp.likelihood.noise_covar.register_constraint(
        "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood=gp.likelihood, model=gp
    ).to(X)
    set_train_(gp)
    optimizer = torch.optim.Adam(gp.parameters(), lr=lr)

    with Timer() as timer:
        for _ in range(n_train):
            optimizer.zero_grad()
            output = gp(X)
            loss = -mll(output, gp.train_targets)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

    return {"elapsed": timer.dt, "losses": losses}


def predict(gp, X, observation_noise=True):
    """Runs a forward prediction on the model.

    Parameters
    ----------
    gp
    X : array_like

    Returns
    -------
    np.ndarray, np.ndarray
        Mean and variance of the distribution
    """

    set_eval_(gp)
    X = torch.tensor(X)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = gp.posterior(X, observation_noise=observation_noise)
    mu = posterior.mean.detach().numpy().squeeze()
    var = posterior.variance.detach().numpy().squeeze()
    return mu, var


def sample(gp, X, samples=20, observation_noise=False):
    """Samples from the GP posterior.

    Parameters
    ----------
    gp
    X : array_like
    samples : int, optional
    observation_noise : bool, optional
        Default set to False so as to get smoother samples.

    Returns
    -------
    np.ndarray
    """

    set_eval_(gp)
    X = torch.tensor(X)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = gp.posterior(X, observation_noise=observation_noise)
        sampled = posterior.sample(torch.Size([samples]))
    return sampled.detach().numpy().squeeze()


@define
class _SingleTaskGPMixin:
    X = field()

    @X.validator  # noqa
    def valid_X(self, attribute, value):
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise ValueError("X must be of type torch.Tensor or numpy.ndarray")
        if value.ndim != 2:
            raise ValueError("X must be of dimension (N, M) (ndims==2)")

    Y = field()

    @Y.validator  # noqa
    def valid_Y(self, attribute, value):
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise ValueError("Y must be of type torch.Tensor or numpy.ndarray")
        if value.ndim != 2:
            raise ValueError("Y must be of dimension (N, 1) (ndims==2)")
        if value.shape[1] != 1:
            raise ValueError("Y must have only one target output")

    Y_noise = field()

    @Y_noise.validator  # noqa
    def valid_Y_noise(self, attribute, value):
        if value is None:
            return
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise ValueError(
                "Y_noise must be of type torch.Tensor or numpy.ndarray"
            )
        if value.ndim != 2:
            raise ValueError("Y_noise must be of dimension (N, 1) (ndims==2)")
        if value.shape[1] != 1:
            raise ValueError("Y_noise must have only one target output")

    def fit_mll(self, **fit_kwargs):
        return fit_gp_gpytorch_mll_(self.model, **fit_kwargs)

    def fit_Adam(self, lr=0.05, n_train=200):
        return fit_gp_Adam_(self.model, self.X, lr=lr, n_train=n_train)

    def predict(self, X, observation_noise=True):
        return predict(self.model, X, observation_noise)

    def sample(self, X, samples=20, observation_noise=False):
        return sample(self.model, X, samples, observation_noise)


@define(kw_only=True)
class EasySingleTaskGP(SaveLoadMixin, _SingleTaskGPMixin, MSONable):
    model = field(validator=validators.instance_of(SingleTaskGP))

    @classmethod
    def from_default(
        cls,
        X,
        Y,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        mean_module=gpytorch.means.ConstantMean(),
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel()
        ),
        transform_input=True,
        transform_output=True,
        **model_kwargs,
    ):
        """Gets a SingleTaskGP from some sensible default parameters.

        Parameters
        ----------
        X : array_like
            Input training data.
        Y : array_like
            Output training data.
        likelihood : TYPE, optional
            Description
        mean_module : TYPE, optional
            Description
        covar_module : TYPE, optional
            Description
        transform_input : bool, optional
            Description
        transform_output : bool, optional
            Description
        **model_kwargs
            Description
        """

        model = _get_model(
            X,
            Y,
            None,  # train_Yvar
            likelihood,
            mean_module,
            covar_module,
            transform_input,
            transform_output,
            **model_kwargs,
        )

        return deepcopy(cls(X=X, Y=Y, Y_noise=None, model=model))


@define(kw_only=True)
class EasyFixedNoiseGP(SaveLoadMixin, _SingleTaskGPMixin, MSONable):
    model = field(validator=validators.instance_of(FixedNoiseGP))

    @classmethod
    def from_default(
        cls,
        X,
        Y,
        Y_noise,
        mean_module=gpytorch.means.ConstantMean(),
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel()
        ),
        transform_input=True,
        transform_output=True,
        **model_kwargs,
    ):
        """Gets a SingleTaskGP from some sensible default parameters.

        Parameters
        ----------
        X : array_like
            Input training data.
        Y : array_like
            Output training data.
        Y_noise : array_like
            Noise
        mean_module : TYPE, optional
            Description
        covar_module : TYPE, optional
            Description
        transform_input : bool, optional
            Description
        transform_output : bool, optional
            Description
        **model_kwargs
            Description
        """

        model = _get_model(
            X,
            Y,
            Y_noise**2,  # train_Yvar
            None,  # likelihood
            mean_module,
            covar_module,
            transform_input,
            transform_output,
            **model_kwargs,
        )

        return deepcopy(cls(X=X, Y=Y, Y_noise=Y_noise, model=model))