"""Module containinj helpers for single task GPs written in gpytorch and
botorch."""

import pickle
from copy import deepcopy
from pathlib import Path
from warnings import warn

import gpytorch
import numpy as np
import torch
from attrs import define, field, validators
from botorch.fit import fit_gpytorch_mll
from botorch.models import FixedNoiseGP, MultiTaskGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from sva import __version__
from sva.models.gp.bo import ask
from sva.utils import Timer, get_coordinates, read_json, save_json


def set_eval_(gp):
    """Sets a Gaussian Process model to evaluation mode.

    Parameters
    ----------
    gp : botorch.models.SingleTaskGP
    """

    gp.eval()
    gp.likelihood.eval()


def set_train_(gp):
    """Sets a Gaussian Process model to training mode.

    Parameters
    ----------
    gp : botorch.models.SingleTaskGP
    """

    gp.train()
    gp.likelihood.train()


def get_model_hyperparameters(model):
    """Iterates through a torch-like object which has a named_parameters()
    method and returns a dictionary of all of the parameters, where values
    are in numpy format.

    Parameters
    ----------
    model
        The model with named_parameters() defined on it.

    Returns
    -------
    dict
    """

    d = {}
    for p in model.named_parameters():
        p0 = str(p[0])
        p1 = p[1].detach().numpy()
        d[p0] = p1
    return d


def get_simple_model(
    model_type,
    X,
    Y,
    train_Yvar=None,
    likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    mean_module=gpytorch.means.ConstantMean(),
    covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),
    transform_input=True,
    transform_output=True,
    task_feature=-1,
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

    if train_Yvar is not None and not isinstance(train_Yvar, torch.Tensor):
        train_Yvar = torch.tensor(train_Yvar)

    if model_type == "SingleTaskGP":
        if train_Yvar is not None:
            raise ValueError(
                "You provided train_Yvar but selected SingleTaskGP, which "
                "does not take this input"
            )

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

    elif model_type == "FixedNoiseGP":
        # Likelihood argument is ignored here
        model = FixedNoiseGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            **model_kwargs,
        )

    elif model_type == "MultiTaskGP":
        model = MultiTaskGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            task_feature=task_feature,
            **model_kwargs,
        )

    else:
        raise ValueError(f"Invalid model type {model_type}")

    return deepcopy(model)


def fit_gp_gpytorch_mll_(gp, device="cpu", **fit_kwargs):
    """Fits a provided GP model using the fit_gpytorch_mll method from
    botorch.

    Returns
    -------
    dict
        A dictionary containing training metadata.
    """

    gp = gp.to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood=gp.likelihood, model=gp
    )
    set_train_(gp)

    with Timer() as timer:
        fit_gpytorch_mll(mll, **fit_kwargs)

    return {"elapsed": timer.dt}


def fit_gp_Adam_(gp, X, device="cpu", lr=0.05, n_train=200):
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
    ).to(device)
    set_train_(gp)
    optimizer = torch.optim.Adam(gp.parameters(), lr=lr)

    X = torch.tensor(X).to(device)

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


@classmethod
def load_model(path):
    """Loads a GPyTorch model from disk."""

    return torch.jit.load(path)


@define
class GPMixin:
    X = field()
    Y = field()
    Yvar = field()

    @X.validator
    def valid_X(self, _, value):
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise ValueError("X must be of type torch.Tensor or numpy.ndarray")
        if value.ndim != 2:
            raise ValueError("X must be of dimension (N, M) (ndims==2)")

    @Y.validator
    def valid_Y(self, _, value):
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise ValueError("Y must be of type torch.Tensor or numpy.ndarray")
        if value.ndim != 2:
            raise ValueError("Y must be of dimension (N, 1) (ndims==2)")
        if value.shape[1] != 1:
            raise ValueError("Y must have only one target output")

    @Yvar.validator  # noqa
    def valid_Y_noise(self, _, value):
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

    def fit_mll(self, device="cpu", **fit_kwargs):
        return fit_gp_gpytorch_mll_(self.model, device=device, **fit_kwargs)

    def fit_Adam(self, device="cpu", lr=0.05, n_train=200):
        return fit_gp_Adam_(
            self.model, self.X, device=device, lr=lr, n_train=n_train
        )

    def predict(self, X, observation_noise=True):
        return predict(self.model, X, observation_noise)

    def sample(self, X, samples=20, observation_noise=False):
        return sample(self.model, X, samples, observation_noise)

    def optimize(self, experiment=None, domain=None, **kwargs):
        """Finds the optima of the GP, either the minimum or the maximum.

        Properties
        ----------
        experiment
            The experiment object, mutually exclusive to bounds.
        domain
            The bounds of the acquisition, mutually exclusive to experiment.
            Note that the bounds should be provided in the same format as that
            of the experimental_domain; i.e., of shape (2, d), where d is
            the dimensionality of the input space.
        **kwargs
            Additional keyword arguments to pass to the optimizer.
        """

        if not ((experiment is not None) ^ (domain is not None)):
            raise ValueError("Provide either experiment or bounds, not both")

        if experiment is not None:
            domain = experiment.properties.experimental_domain

        if "q" in kwargs:
            warn(
                "q has been provided to the optimizer but will be overridden to 1"
            )
            kwargs["q"] = 1
        else:
            kwargs["q"] = 1

        if "num_restarts" not in kwargs:
            kwargs["num_restarts"] = 20

        if "raw_samples" not in kwargs:
            kwargs["raw_samples"] = 100

        result = ask(
            self.model,
            "UCB",
            bounds=domain,
            acquisition_function_kwargs={"beta": 0.0},
            optimize_acqf_kwargs={**kwargs},
        )
        return result

    def dream(self, ppd=20, experiment=None, domain=None):
        """Creates a new Gaussian Process model by sampling a single instance
        of the GP, and then retraining a _new_ GP on a densely sampled grid
        assuming that single sample is the ground truth.

        Parameters
        ----------
        experiment : ExperimentMixin
            The experiment object, mutually exclusive to bounds.
        domain : np.ndarray
            The bounds of the acquisition, mutually exclusive to experiment.
            Note that the bounds should be provided in the same format as that
            of the experimental_domain; i.e., of shape (2, d), where d is
            the dimensionality of the input space.

        Returns
        -------
        EasySingleTaskGP
        """

        if not ((experiment is not None) ^ (domain is not None)):
            raise ValueError("Provide either experiment or bounds, not both")

        if experiment is not None:
            X = experiment.get_dense_coordinates(ppd=ppd)
        else:
            X = get_coordinates(ppd, domain)

        Y = self.sample(X, samples=1)
        Y = Y.reshape(-1, 1)

        # Now fit a new GP to this data
        new_gp = EasySingleTaskGP.from_default(X, Y)
        new_gp.fit_mll()

        # And return this GP, which is now the new "truth" function
        # Very useful in campaigns
        return new_gp

    def save(self, path):
        """Serializes the EasyGP object to disk."""

        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        pickle.dump(
            self,
            open(path / "klass.pkl", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        # Finally we save the version information used to generate the
        # class
        d = {
            "__version__": __version__,
            "__name__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
        }
        save_json(d, path / "metadata.json")

    @classmethod
    def load(cls, path):
        path = Path(path)
        assert path.exists()

        d = read_json(path / "metadata.json")
        version = d["__version__"]
        if version != __version__:
            warn(
                f"Model version loaded ({version}) does not match"
                f"current sva version {__version__}"
            )

        return pickle.load(open(path / "klass.pkl", "rb"))


@define(kw_only=True)
class EasySingleTaskGP(GPMixin):
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
        """Gets a SingleTaskGP from some sensible default parameters."""

        model = get_simple_model(
            "SingleTaskGP",
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

        return deepcopy(cls(X=X, Y=Y, Yvar=None, model=model))


@define(kw_only=True)
class EasyFixedNoiseGP(GPMixin):
    model = field(validator=validators.instance_of(FixedNoiseGP))

    @classmethod
    def from_default(
        cls,
        X,
        Y,
        Yvar,
        mean_module=gpytorch.means.ConstantMean(),
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel()
        ),
        transform_input=True,
        transform_output=True,
        **model_kwargs,
    ):
        """Gets a SingleTaskGP from some sensible default parameters."""

        model = get_simple_model(
            "FixedNoiseGP",
            X,
            Y,
            Yvar,
            None,  # likelihood
            mean_module,
            covar_module,
            transform_input,
            transform_output,
            **model_kwargs,
        )

        return deepcopy(cls(X=X, Y=Y, Yvar=Yvar, model=model))


@define(kw_only=True)
class EasyMultiTaskGP(GPMixin):
    model = field(validator=validators.instance_of(MultiTaskGP))

    @classmethod
    def from_default(
        cls,
        X,
        Y,
        Yvar=None,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        mean_module=gpytorch.means.ConstantMean(),
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel()
        ),
        transform_input=True,
        transform_output=True,
        task_feature=-1,
        **model_kwargs,
    ):
        """Gets a MultiTaskGP from some sensible default parameters."""

        model = get_simple_model(
            "MultiTaskGP",
            X,
            Y,
            Yvar,
            likelihood,
            mean_module,
            covar_module,
            transform_input,
            transform_output,
            task_feature,
            **model_kwargs,
        )

        return deepcopy(cls(X=X, Y=Y, Yvar=Yvar, model=model))
