from functools import partial

import numpy as np
import torch
from attrs import define, field
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
from scipy.spatial import distance_matrix
from torch import nn


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


@define
class BasePenalty(nn.Module):
    """A base penalty module which defines the required experiment and data
    properties, so as to provide a consistent API for defining penalty
    functions."""

    experiment = field()
    data = field()

    def __attrs_pre_init__(self):
        super().__init__()


@define
class ProximityPenalty(BasePenalty):
    """A penalty function that strongly penalizes a new point if it is
    within the neighborhood of an existing point. The lengthscale of this
    penalty is set by some multiplier times the largest NN distance in the
    current dataset."""

    # 1/10th of the maximum nearest neighbor distance is used to fix the
    # cutoff
    divisor = field(default=10.0)

    def __attrs_post_init__(self):
        self.current_X = self.data.X.copy()
        if self.current_X is None or len(self.current_X) < 2:
            self.lengthscale = None
            return
        d = distance_matrix(self.current_X, self.current_X)
        d[d == 0] = np.inf
        self.lengthscale = d.min(axis=1).max() / self.divisor

    def forward(self, x):
        x = x.detach().numpy()
        # x is batch shape x (q=1) x dim tensor
        if x.shape[-2] != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} not implemented for q>1 yet"
            )

        if self.lengthscale is None:
            # No penalty if we don't have any data, or the number of data
            # points is <2
            return np.zeros((x.shape[0],))

        # Compute the distance between x in the current batch and the existing
        # points to find closest ones
        d = distance_matrix(x.squeeze(), self.current_X)

        # Compute the minimum distance between the candidates and the existing
        # points
        d_nn = d.min(axis=1)

        # Compute the penalty
        p = np.exp(-d_nn / self.lengthscale)

        return torch.tensor(p.squeeze())
