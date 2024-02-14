from abc import ABC, abstractmethod, abstractproperty
from functools import cached_property

import gpytorch
import numpy as np
import torch
from botorch.acquisition.penalized import PenalizedAcquisitionFunction
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from monty.json import MSONable
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from sva.utils import Timer, get_function_from_signature
from sva.value import svf as value_function

ACQF_ALIASES = {
    "EI": "botorch.acquisition.analytic:ExpectedImprovement",
    "UCB": "botorch.acquisition.analytic:UpperConfidenceBound",
}


def ask(
    gp,
    acquisition_function,
    bounds,
    acquisition_function_kwargs=None,
    optimize_acqf_kwargs=None,
):

    # bounds = torch.tensor(self.experiment.experimental_domain)
    bounds = torch.tensor(bounds)

    signature = ACQF_ALIASES.get(acquisition_function)
    if signature is None:
        raise KeyError(f"Incompatible signature {acquisition_function}")
    factory = get_function_from_signature(signature)

    kwargs = acquisition_function_kwargs if acquisition_function_kwargs else {}
    acqf = factory(gp, **kwargs)

    kwargs = optimize_acqf_kwargs if optimize_acqf_kwargs else {}
    next_points, value = optimize_acqf(acqf, bounds=bounds, **kwargs)

    return {
        "next_points": next_points,
        "value": value,
        "acquisition_function": acqf,
    }
