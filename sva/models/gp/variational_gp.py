"""TODO

See here

https://docs.gpytorch.ai/en/v1.6.0/examples/04_Variational_and_Approximate_GPs/Approximate_GP_Objective_Functions.html?highlight=heteroskedastic#Experimental-setup
"""

from copy import deepcopy

from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP, FixedNoiseGP
import gpytorch
import torch

from sva.utils import Timer
from sva.models.gp.common import set_train_, set_eval_


class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, mean_module, covar_module):
        variational_distribution = (
            gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.shape[0]
            )
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points.squeeze(),
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_approximate_gp_(model, X, Y, fit_kwargs):
    # model = ApproximateGPModel(torch.linspace(0, 1))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    objective_function = gpytorch.mlls.PredictiveLogLikelihood(
        likelihood, model, num_data=Y.shape[0]
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=0.1
    )

    # Train
    model.train()
    likelihood.train()
    n_iter = fit_kwargs.get("n_iter", 200)
    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -objective_function(output, Y.squeeze())
        loss.backward()
        optimizer.step()


def get_model(
    X,
    Y,
    train_Yvar=None,
    likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    mean_module=gpytorch.means.ConstantMean(),
    covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),
    transform_input=True,
    transform_output=True,
    heteroskedastic_noise=False,
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
        if not heteroskedastic_noise:
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
        return ApproximateGPModel(X, mean_module, covar_module)

    else:
        # Likelihood argument is ignored here
        model = FixedNoiseGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            **model_kwargs,
        )

    return deepcopy(model)


def fit_gp_gpytorch_mll_(gp, fit_kwargs=None):
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

    if fit_kwargs is None:
        fit_kwargs = {}

    with Timer() as timer:
        fit_gpytorch_mll(mll, **fit_kwargs)

    return {"elapsed": timer.dt}


def fit_gp_Adam_(gp, X, fit_kwargs=None):
    """Fits a provided GP model using the Adam optimizer.

    Returns
    -------
    dict
        A dictionary containing training metadata.
    """

    if fit_kwargs is None:
        fit_kwargs = {}

    lr = fit_kwargs.get("lr", 0.05)
    n_train = fit_kwargs.get("n_train", 200)
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
