from copy import deepcopy
from warnings import warn

import numpy as np
from attrs import define, field
from attrs.validators import instance_of
from monty.json import MSONable
from tqdm import tqdm

from sva.models.gp import EasySingleTaskGP, get_train_protocol
from sva.models.gp.bo import ask, is_EI


@define
class CampaignParameters(MSONable):
    """Essentially a data class for containing necessary campaign parameters,
    and the logic for setting them to sensible defaults if they are unset
    at instantiation. This provides a common structure for all parts of the
    SVA code which use a campaign.

    Parameters
    ----------
    acquisition_function : dict
        Has two keys, "method" and "kwargs". The method should be either
        a string representation of a model signature, an alias for
        an acquisition function defined in sva.models.gp.bo or a factory
        for an acquisition function. The kwargs are the keyword arguments to
        be passed to the acquisition function.
    train_protocol : str
        The protocol for training. Has two keys, "method" and "kwargs". The
        method must be a method defined on the EasyGP. For example, "fit_mll".
        The kwargs are the optional keyword arguments to pass to the fitting
        method.
    optimize_acqf_kwargs : dict, optional
        Keyword arguments to pass to the optimizer of the acquisition
        function. Sensible defaults are set if this is not provided,
        including sequential (q=1) optimization.
    optimize_gp : dict, optional
        If set (not None), will perform an optimization step over the fitted
        GP at every step of the experiment, finding its maxima using the fully
        exploitative UCB(beta=0) acquisition function.
    model_factory : callable
        A factory that returns a GP model.
    modality_callback : callable
        Used in the multimodal experiments. Determines which modality to use
        as a function of experiment iteration.
    task_feature : int
        The index of the task feature in multimodal experiments. Defaults to
        -1.
    """

    acquisition_function = field(
        default=None, validator=instance_of((dict, type(None)))
    )

    @acquisition_function.validator
    def valid_acquisition_function(self, _, value):
        if value is None:
            return
        keys = list(value.keys())
        if "method" not in keys or "kwargs" not in keys:
            raise KeyError(
                "Either method or kwargs was not found in "
                "acquisition_function keys. Both are required."
            )

    train_protocol = field(
        default=None, validator=instance_of((dict, str, type(None)))
    )

    @train_protocol.validator
    def valid_train_protocol(self, _, value):
        if value is None:
            return
        if isinstance(value, str):
            if value not in ["fit_mll", "fit_Adam"]:
                raise ValueError(
                    "If train_protocol is a str, must be 'fit_mll' or "
                    "'fit_Adam'"
                )
        keys = list(value.keys())
        if "method" not in keys or "kwargs" not in keys:
            raise KeyError(
                "Either method or kwargs was not found in "
                "train_protocol keys. Both are required."
            )

    optimize_acqf_kwargs = field(
        default=None, validator=instance_of((dict, type(None)))
    )
    optimize_gp = field(default=None, validator=instance_of((dict, type(None))))
    model_factory = field(default=None)
    modality_callback = field(
        default=None, validator=instance_of((int, type(None)))
    )
    task_feature = field(default=-1, validator=instance_of((int, type(None))))

    def _set_acquisition_function(self):
        if self.acquisition_function is not None:
            return
        self.acquisition_function = {"method": "EI", "kwargs": None}
        warn(
            "acquisition_function was unset. Using default: "
            f"{self.acquisition_function}"
        )

    def _set_train_protocol(self):
        if self.train_protocol is None:
            self.train_protocol = {"method": "fit_mll", "kwargs": {}}
        if isinstance(self.train_protocol, str):
            train_method, train_kwargs = get_train_protocol(self.train_protocol)
            self.train_protocol = {
                "method": train_method,
                "kwargs": train_kwargs,
            }
            return
        warn(f"train_protocol was unset. Using default: {self.train_protocol}")

    def _set_optimize_acqf_kwargs(self):
        if self.optimize_acqf_kwargs is not None:
            return
        self.optimize_acqf_kwargs = {
            "q": 1,
            "num_restarts": 20,
            "raw_samples": 100,
        }
        warn(
            "optimize_acqf_kwargs was unset. Using default: "
            f"{self.optimize_acqf_kwargs}"
        )

    def _set_model_factory(self):
        if self.model_factory is not None:
            return
        self.model_factory = EasySingleTaskGP.from_default
        warn("model_factory was unset. Using default EasyTaskGP.from_default")

    def __attrs_post_init__(self):
        self._set_acquisition_function()
        self._set_train_protocol()
        self._set_optimize_acqf_kwargs()
        self._set_model_factory()

    def set_optimize_gp_default(self):
        self.optimize_gp = {
            "num_restarts": 150,
            "raw_samples": 150,
        }


class CampaignBaseMixin:
    """This class acts as a mixin for experiments in which there is a single
    output."""

    def _calculate_remaining_loops(self, n, q):
        # Calculate the number of remaining experiments, including the
        # number of experiment loops to perform
        if self.data.X is not None:
            initial_size = self.data.X.shape[0]
        else:
            initial_size = 0
        remaining = n - initial_size
        if remaining <= 0:
            warn("No experiments performed, set n higher for more experiments")
            return
        loops = np.ceil(remaining / q)
        return int(loops)

    def _ask(
        self,
        gp,
        Y,
        acquisition_function,
        acquisition_function_kwargs,
        optimize_acqf_kwargs,
    ):
        kwargs = deepcopy(acquisition_function_kwargs)
        if is_EI(acquisition_function):
            if kwargs is None:
                kwargs = {}
            kwargs["best_f"] = Y.max()
        return ask(
            gp.model,
            acquisition_function,
            bounds=self.properties.experimental_domain,
            acquisition_function_kwargs=kwargs,
            optimize_acqf_kwargs=optimize_acqf_kwargs,
        )

    def run(
        self,
        n,
        parameters=None,
        svf=None,
        pbar=True,
    ):
        """Executes the campaign by running many sequential experiments.

        Parameters
        ----------
        n : int
            The total number of experiments to run.
        parameters : CampaignParameters
            The set of parameters that defines the run.
        svf
            If None, does standard optimization. Otherwise, uses the
            Scientific Value Agent transformation.
        pbar : bool
            Whether or not to display the tqdm progress bar.
        """

        if parameters is None:
            parameters = CampaignParameters()

        q = parameters.optimize_acqf_kwargs["q"]
        loops = self._calculate_remaining_loops(n, q)

        # The for loop runs over the maximum possible number of experiments
        for ii in tqdm(range(loops), disable=not pbar):
            # Get the data
            X = self.data.X
            Y = self.data.Y
            Yvar = self.data.Yvar

            # Simple fitting of a Gaussian process
            # using some pretty simple default values for things, which we
            # can always change later
            # TODO: enable the SVA to use Yvar when present
            if svf:
                Y = svf(X, Y).reshape(-1, 1)

            # The factory instantiates a model. It must have the from_default
            # method defined on it. It must also be compatible with the
            # from_default method (so for example, if your model has noisy
            # observations, you must use a noise-compatible GP, such as
            # a fixed-noise GP).
            args = [X, Y]
            if Yvar is None:
                args.append(Yvar)
            gp = parameters.model_factory(*args)

            # Fit the model
            train_method = parameters.train_protocol["method"]
            train_kwargs = parameters.train_protocol["kwargs"]
            getattr(gp, train_method)(**train_kwargs)

            # Ask the model what to do next, we're also careful to check for
            # the best_f required keyword argument in the case of an EI
            # acquisition function
            # Update the internal data store with the next points
            state = self._ask(
                gp,
                Y,
                parameters.acquisition_function["method"],
                parameters.acquisition_function["kwargs"],
                parameters.optimize_acqf_kwargs,
            )

            # Append the history with everything we want to keep
            # Note that the complete state of the GP is saved in the
            # acquisition function model
            d = {
                "iteration": ii,
                "N": X.shape[0],
                "state": state,
                "easy_gp": deepcopy(gp),
            }

            # Optionally, we can run an additional optimization step on the GP
            # to get the maximum value of that GP in the experimental space.
            # This is useful for simulated campaigning and is disabled by
            # default.
            if parameters.optimize_gp is not None:
                r = gp.optimize(experiment=self, **parameters.optimize_gp)
                r.pop("acquisition_function")
                d["optimize_gp"] = r

            self.history.append(d)

            self.update_data(state["next_points"])
