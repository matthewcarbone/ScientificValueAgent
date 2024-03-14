import torch
from botorch.optim import optimize_acqf

from sva.utils import get_function_from_signature

ACQF_ALIASES = {
    "EI": "botorch.acquisition.analytic:ExpectedImprovement",
    "UCB": "botorch.acquisition.analytic:UpperConfidenceBound",
    "qEI": "botorch.acquisition.monte_carlo:qExpectedImprovement",
    "qUCB": "botorch.acquisition.monte_carlo:qUpperConfidenceBound",
}


def ask(
    gp,
    acquisition_function,
    bounds,
    acquisition_function_kwargs=None,
    optimize_acqf_kwargs=None,
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
    optimize_acqf_kwargs : dict
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
        signature_from_alias = ACQF_ALIASES.get(acquisition_function)
        if signature_from_alias:
            factory = get_function_from_signature(signature_from_alias)
        else:
            factory = get_function_from_signature(acquisition_function)
        acqf = factory(gp, **kwargs)
    else:
        acqf = acquisition_function(gp, **kwargs)

    kwargs = optimize_acqf_kwargs if optimize_acqf_kwargs else {}
    next_points, value = optimize_acqf(acqf, bounds=bounds, **kwargs)

    return {
        "next_points": next_points,
        "value": value,
        "acquisition_function": acqf,
    }
