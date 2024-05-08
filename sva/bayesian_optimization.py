from functools import partial

from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qExpectedImprovement,
    qUpperConfidenceBound,
)
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


def parse_acquisition_function(acqf):
    """Parses the provided acquisition function information into a factory.
    Also returns a boolean for whether or not the acquisition function is
    EI and requires the best_f argument."""

    if isinstance(acqf, dict):
        acqf_kwargs = acqf["kwargs"]
        acqf = acqf["acquisition_function"]
        acqf_kwargs = acqf_kwargs if acqf_kwargs is not None else {}
        acqf = ACQF_ALIASES[acqf]
        requires_best_f = "ExpectedImprovement" in acqf.__class__.__name__
        return partial(acqf, **acqf_kwargs), requires_best_f
    elif isinstance(acqf, str):
        if "EI" in acqf:
            return partial(ACQF_ALIASES[acqf]), True
        acqf, beta = acqf.split("-")
        return partial(ACQF_ALIASES[acqf], beta=float(beta)), False
    if not isinstance(acqf, partial):
        raise ValueError("acqf is not dict or string, must be partial")
    return acqf, "ExpectedImprovement" in str(acqf.func)


def get_acquisition_function_name(acqf):
    """Gets a string representation of the acquisition function provided."""
    if isinstance(acqf, dict):
        return acqf["acquisition_function"]
    elif isinstance(acqf, str):
        return acqf
    # Otherwise, it's some sort of partial object
    return str(acqf.func.__name__)
