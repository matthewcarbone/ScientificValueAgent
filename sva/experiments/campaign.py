from copy import deepcopy
from warnings import catch_warnings, warn

import numpy as np
import torch
from attrs import define, field
from attrs.validators import instance_of
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from tqdm import tqdm

from sva.models.gp import EasyMultiTaskGP, EasySingleTaskGP, get_train_protocol
from sva.models.gp.bo import ask, is_EI
from sva.monty.json import MSONable
from sva.utils import get_hash


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
    svf : SVF
        Scientific Value Function, can be optionally provided.
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
        if isinstance(value, dict):
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
    modality_callback = field(default=None)

    @modality_callback.validator
    def valid_modality_callback(self, _, value):
        if value is None:
            return
        if not callable(value):
            raise ValueError("modality_callback must be None or callable")

    task_feature = field(default=-1, validator=instance_of((int, type(None))))

    svf = field(default=None)

    save_models = field(default=False, validator=instance_of(bool))
    save_acquisition_functions = field(
        default=False, validator=instance_of(bool)
    )

    def _set_acquisition_function(self):
        if self.acquisition_function is not None:
            return
        self.acquisition_function = {"method": "EI", "kwargs": None}
        warn(
            "acquisition_function was unset. Using default: "
            f"{self.acquisition_function}"
        )

    @property
    def acqf_key(self):
        """Gets a simple string representation of the parameters. This actually
        abstracts away all other attributes except the acquisition function,
        which is of primary interest when running a campaign."""

        if self.acquisition_function["method"] == "EI":
            return "EI"
        if self.acquisition_function["method"] == "UCB":
            beta = self.acquisition_function["kwargs"]["beta"]
            return f"UCB-{beta:.01f}"
        raise ValueError("Unknown acquisition function method")

    @property
    def name(self):
        return self.acqf_key

    def _set_train_protocol(self):
        if self.train_protocol is None:
            self.train_protocol = {"method": "fit_mll", "kwargs": {}}
            warn(
                "train_protocol was unset. "
                f"Using default: {self.train_protocol}"
            )
        if isinstance(self.train_protocol, str):
            train_method, train_kwargs = get_train_protocol(self.train_protocol)
            self.train_protocol = {
                "method": train_method,
                "kwargs": train_kwargs,
            }
            return

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

    def validate_multimodal_experiment(self):
        if self.modality_callback is None:
            warn(
                "modality_callback is unset. Using modality 0 only by default."
                "Note that this is probably NOT what you want to do! Ensure "
                "modality_callback is set in the parameters."
            )
            self.modality_callback = lambda _: 0
        if self.model_factory == EasySingleTaskGP.from_default:
            warn(
                "Model factory was set to default EasySingleTaskGP. Changing "
                "to use EasyMultiTaskGP"
            )
            self.model_factory = EasyMultiTaskGP.from_default

    def get_hash(self):
        return get_hash(str(self.as_dict()))

    def __eq__(self, x):
        if not isinstance(x, CampaignParameters):
            return False
        return self.get_hash() == x.get_hash()

    @classmethod
    def from_standard_testing_array(cls, betas, use_EI, **kwargs):
        """Gets a standard testing array consisting of EI and a variety of
        choices for UCB."""

        parameters = []

        with catch_warnings(record=True) as _:
            if use_EI:
                acqf = {"method": "EI", "kwargs": None}
                klass = cls(acquisition_function=acqf, **kwargs)
                parameters.append(klass)
            for beta in betas:
                acqf = {"method": "UCB", "kwargs": {"beta": beta}}
                klass = cls(acquisition_function=acqf, **kwargs)
                parameters.append(klass)
        if len(parameters) == 0:
            raise RuntimeError("No parameters!")
        return parameters


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

    def _initialize_pre_loop(self, parameters, n):
        if parameters is None:
            parameters = CampaignParameters()

        q = parameters.optimize_acqf_kwargs["q"]
        loops = self._calculate_remaining_loops(n, q)
        return parameters, loops

    def _get_data(self):
        X = self.data.X
        Y = self.data.Y
        Yvar = self.data.Yvar
        return X, Y, Yvar

    def _svf_transform(self, svf, X, Y, Yvar):
        # TODO: enable the SVA to use Yvar when present
        if svf:
            return svf(X, Y).reshape(-1, 1)
        return Y

    def _get_gp(self, parameters, X, Y, Yvar):
        # The factory instantiates a model. It must have the from_default
        # method defined on it. It must also be compatible with the
        # from_default method (so for example, if your model has noisy
        # observations, you must use a noise-compatible GP, such as
        # a fixed-noise GP).
        args = [X, Y]
        if Yvar is None:
            args.append(Yvar)
        return parameters.model_factory(*args)

    def _fit(self, parameters, gp):
        # Fit the model
        train_method = parameters.train_protocol["method"]
        train_kwargs = parameters.train_protocol["kwargs"]
        getattr(gp, train_method)(**train_kwargs)

    def _ask(self, parameters, gp, Y):
        # Ask the model what to do next, we're also careful to check for
        # the best_f required keyword argument in the case of an EI
        # acquisition function
        # Update the internal data store with the next points
        acquisition_function = parameters.acquisition_function["method"]
        acquisition_function_kwargs = parameters.acquisition_function["kwargs"]
        optimize_acqf_kwargs = parameters.optimize_acqf_kwargs
        kwargs = deepcopy(acquisition_function_kwargs)
        if is_EI(acquisition_function):
            if kwargs is None:
                kwargs = {}
            kwargs["best_f"] = Y.max()
        state = ask(
            gp.model,
            acquisition_function,
            bounds=self.properties.experimental_domain,
            acquisition_function_kwargs=kwargs,
            optimize_acqf_kwargs=optimize_acqf_kwargs,
        )

        if not parameters.save_acquisition_functions:
            state.pop("acquisition_function")

        return state

    def _optimize_gp(self, parameters, gp, d):
        # Optionally, we can run an additional optimization step on the GP
        # to get the maximum value of that GP in the experimental space.
        # This is useful for simulated campaigning and is disabled by
        # default.
        if parameters.optimize_gp is not None:
            r = gp.optimize(experiment=self, **parameters.optimize_gp)
            r.pop("acquisition_function")
            d["optimize_gp"] = r
        return d

    def run(
        self,
        n,
        parameters=None,
        pbar=True,
        additional_experiments=True,
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

        if not additional_experiments:
            parameters, loops = self._initialize_pre_loop(parameters, n)
        else:
            loops = n

        self.metadata["runtime_properties"].append(parameters)

        # The for loop runs over the maximum possible number of experiments
        for ii in tqdm(range(loops), disable=not pbar):
            X, Y, Yvar = self._get_data()
            Y = self._svf_transform(parameters.svf, X, Y, Yvar)
            gp = self._get_gp(parameters, X, Y, Yvar)
            self._fit(parameters, gp)
            state = self._ask(parameters, gp, Y)

            # Append the history with everything we want to keep
            # Note that the complete state of the GP is saved in the
            # acquisition function model
            d = {
                "iteration": ii,
                "N": X.shape[0],
                "state": state,
                "easy_gp": deepcopy(gp) if parameters.save_models else None,
            }

            d = self._optimize_gp(parameters, gp, d)

            self.history.append(d)

            self.update_data(state["next_points"])


class MultimodalCampaignMixin(CampaignBaseMixin):
    def _svf_transform(self, svf, X, Y, Yvar, task_feature):
        if svf is not None:
            new_target = np.empty(shape=(Y.shape[0], 1))
            # Assign each of the values based on the individual modal
            # experiments. The GP should take care of the rest
            for modality_index in range(self.n_modalities):
                where = np.where(X[:, task_feature] == modality_index)[0]
                if len(where) > 0:
                    target = svf(X[where, :], Y[where, :])
                    new_target[where, :] = target.reshape(-1, 1)
            target = new_target
        else:
            target = Y
        return target

    def _ask(self, parameters, gp, Y, ii):
        acquisition_function = parameters.acquisition_function["method"]
        acquisition_function_kwargs = parameters.acquisition_function["kwargs"]
        optimize_acqf_kwargs = parameters.optimize_acqf_kwargs
        acquisition_function_kwargs = deepcopy(acquisition_function_kwargs)

        # get the current modality of the experiment we're currently
        # running
        modality_index = parameters.modality_callback(ii)

        # Need to use a posterior transform here to tell the acquisition
        # function how to weight the multiple outputs
        weights = np.zeros(shape=(self.n_modalities,))
        weights[modality_index] = 1.0
        weights = torch.tensor(weights)
        transform = ScalarizedPosteriorTransform(weights=weights)
        acquisition_function_kwargs["posterior_transform"] = transform

        # Ask the model what to do next
        if is_EI(acquisition_function):
            acquisition_function_kwargs["best_f"] = Y[:, modality_index].max()

        state = ask(
            gp.model,
            acquisition_function,
            bounds=self.properties.experimental_domain,
            acquisition_function_kwargs=acquisition_function_kwargs,
            optimize_acqf_kwargs=optimize_acqf_kwargs,
        )

        if not parameters.save_acquisition_functions:
            state.pop("acquisition_function")

        return state

    def run(
        self,
        n,
        parameters=None,
        pbar=True,
        additional_experiments=True,
    ):
        """A special experiment executor which works on multimodal data. The
        user must specify a special callback function which provides the
        modality index as a function of iteration. By default, this is just
        0 for any iteration index (returns the low-fidelity experiment index).
        """

        if not additional_experiments:
            parameters, loops = self._initialize_pre_loop(parameters, n)
        else:
            loops = n
        parameters.validate_multimodal_experiment()
        task_feature = parameters.task_feature

        self.metadata["runtime_properties"].append(parameters)

        # The for loop runs over the maximum possible number of experiments
        for ii in tqdm(range(loops), disable=not pbar):
            X, Y, Yvar = self._get_data()
            Y = self._svf_transform(parameters.svf, X, Y, Yvar, task_feature)

            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            if Yvar is not None and Yvar.ndim == 1:
                Yvar = Yvar.reshape(-1, 1)

            gp = self._get_gp(parameters, X, Y, Yvar)
            self._fit(parameters, gp)

            state = self._ask(parameters, gp, Y, ii)

            # Append the current modality to the next points
            x = state["next_points"]
            n_pts = x.shape[0]
            modality = parameters.modality_callback(ii)
            modality_array = np.zeros((n_pts, 1)) + modality
            state["next_points"] = np.concatenate([x, modality_array], axis=1)

            # Append the history with everything we want to keep
            # Note that the complete state of the GP is saved in the
            # acquisition function model
            d = {
                "iteration": ii,
                "N": X.shape[0],
                "state": state,
                "easy_gp": deepcopy(gp) if parameters.save_models else None,
            }

            d = self._optimize_gp(parameters, gp, d)

            self.history.append(d)

            self.update_data(state["next_points"])
