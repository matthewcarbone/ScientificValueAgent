from copy import deepcopy

from tqdm import tqdm

from sva.models.gp import EasySingleTaskGP
from sva.models.gp.bo import ask


def _is_EI(acquisition_function_factory):
    """Helper function for determining if an acquisition function factory is
    of type ExpectedImprovement."""

    if isinstance(acquisition_function_factory, str):
        if acquisition_function_factory in ["EI", "qEI"]:
            return True
    if "ExpectedImprovement" in acquisition_function_factory.__class__.__name__:
        return True
    return False


def run_simple_campaign(
    n_experiments,
    experiment,
    acquisition_function,
    acquisition_function_kwargs,
    optimize_acqf_kwargs,
    svf=None,
):
    """Executes a simple Bayesian Optimization campaign. Uses the following
    components:
        - The provided experiment as a source of truth
        - The provided acquisition function factory
        - A EasySingleTaskGP
    """

    for ii in tqdm(range(n_experiments)):
        # Get the data
        X = experiment.data.X
        Y = experiment.data.Y

        if X.shape[0] > n_experiments:
            break

        # Simple fitting of a Gaussian process
        # using some pretty simple default values for things, which we
        # can always change later
        if svf:
            Y = svf(X, Y)
            Y = Y.reshape(-1, 1)
        gp = EasySingleTaskGP.from_default(X, Y)

        # Use the built in botorch fitting procedure here
        gp.fit_mll()

        # Ask the model what to do next
        if _is_EI(acquisition_function):
            acquisition_function_kwargs["best_f"] = Y.max()
        state = ask(
            gp.model,
            acquisition_function,
            bounds=experiment.properties.experimental_domain,
            acquisition_function_kwargs=acquisition_function_kwargs,
            optimize_acqf_kwargs=optimize_acqf_kwargs,
        )

        # Update the internal data store with the next points
        X2 = state["next_points"]
        experiment.update_data(X2)

        # Append the history with everything we want to keep
        # Note that the complete state of the GP is saved in the
        # acquisition function model
        experiment.data.history.append(
            {
                "iteration": ii,
                "next_points": state["next_points"],
                "value": state["value"],
                "acquisition_function": deepcopy(state["acquisition_function"]),
                "easy_gp": deepcopy(gp),
            }
        )
