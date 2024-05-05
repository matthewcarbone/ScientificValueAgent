from warnings import catch_warnings

import torch
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)


class MaxVariance(AnalyticAcquisitionFunction):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean
        view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        return posterior.variance.view(view_shape)


class qMaxVariance(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        sampler=None,
        objective=None,
        posterior_transform=None,
        X_pending=None,
    ):
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        ucb_samples = (obj - mean).abs()
        return ucb_samples.max(dim=-1)[0].mean(dim=0)


ACQF_ALIASES = {
    "EI": ExpectedImprovement,
    "UCB": UpperConfidenceBound,
    "qEI": qExpectedImprovement,
    "qUCB": qUpperConfidenceBound,
    "MaxVar": MaxVariance,
    "qMaxVar": qMaxVariance,
}


def is_EI(acquisition_function_factory):
    """Helper function for determining if an acquisition function factory is
    of type ExpectedImprovement."""

    if isinstance(acquisition_function_factory, str):
        if acquisition_function_factory in ["EI", "qEI"]:
            return True
    if "ExpectedImprovement" in acquisition_function_factory.__class__.__name__:
        return True
    return False


def ask(
    gp,
    acquisition_function,
    bounds,
    acquisition_function_kwargs=None,
    optimize_kwargs=None,
):
    """
    "Asks" the acquisition function to tell the user which point(s) to sample
    next.

    # bounds = torch.tensor(self.experiment.experimental_domain)

    Parameters
    ----------
    gp
        The Gaussian Process model under consideration.
    acquisition_function : str, callable
        The string representing the acquisition function to use. Can also
        be a callable object such as
        botorch.acquisition.analytic.ExpectedImprovement, such that it is
        initialized during the running of this function. Can also be a string
        representing the alias of the acquisition function, or a signature e.g.
        "botorch.acquisition.analytic:ExpectedImprovement".
    bounds : array_like
        The bounds on the procedure.
    acquisition_function_kwargs : dict
        Keyword arguments to pass to the acquisition function.
    optimize_kwargs : dict
        Keyword arguments to pass to the optimizer.

    Raises
    ------
    KeyError:
        If the provided acquisition function does not match available functions.

    Returns
    -------
    dict
        A dictionary containing the next points, value of the acquisition
        function as well as the acquisition function object itself.
    """

    bounds = torch.tensor(bounds)

    kwargs = acquisition_function_kwargs if acquisition_function_kwargs else {}

    if isinstance(acquisition_function, str):
        factory = ACQF_ALIASES[acquisition_function]
    else:
        factory = acquisition_function
    acqf = factory(gp, **kwargs)

    kwargs = optimize_kwargs if optimize_kwargs else {}

    with catch_warnings(record=True) as w:
        next_points, value = optimize_acqf(acqf, bounds=bounds, **kwargs)

    return {
        "next_points": next_points,
        "value": value,
        "acquisition_function": acqf,
        "warnings": w,
    }
