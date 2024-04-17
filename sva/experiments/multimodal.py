from abc import abstractproperty
from copy import deepcopy

import numpy as np
import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from tqdm import tqdm

from sva.models.gp import EasyMultiTaskGP
from sva.models.gp.bo import ask

from .base import ExperimentMixin


# WARNING: This is not really functional yet
class MultimodalExperimentMixin(ExperimentMixin):
    @abstractproperty
    def n_modalities(self):
        raise NotImplementedError

    def _validate_input(self, x):
        # Ensure x has the right shape
        if not x.ndim == 2:
            raise ValueError(
                "x must have shape (N, d + 1), where d is the dimension of "
                "the feature, and the added extra dimension is the task index"
            )

        # Ensure that the second dimension of x is the right size for the
        # chosen experiment
        if not x.shape[1] == self.properties.n_input_dim + 1:
            raise ValueError(
                f"x second dimension must be {self.properties.n_input_dim+1}"
            )

        # Ensure that the input is in the bounds
        # Domain being None implies the domain is the entirety of the reals
        # This is a fast way to avoid the check
        if self.properties.valid_domain is not None:
            check1 = self.properties.valid_domain[0, :] <= x[:, :-1]
            check2 = x[:, :-1] <= self.properties.valid_domain[1, :]
            if not np.all(check1 & check2):
                raise ValueError("Some inputs x were not in the domain")

    def get_random_coordinates(self, n, modality=0):
        x = super().get_random_coordinates(n)
        n = x.shape[0]
        modality_array = np.zeros((n, 1)) + modality
        return np.concatenate([x, modality_array], axis=1)

    def get_dense_coordinates(self, ppd, modality=0):
        """Gets a set of dense coordinates, augmented with the modality
        index, which defaults to 0.

        Parameters
        ----------
        ppd : int or list
            Points per dimension.
        modality : int
            Indexes the modality to use in multi-modal experiments.
        domain : np.ndarray, torch.tensor, optional
            The experimental domain of interest. If not provided defaults to
            that of the self experiment.

        Returns
        -------
        np.ndarray
        """

        x = super().get_dense_coordinates(ppd)
        n = x.shape[0]
        modality_array = np.zeros((n, 1)) + modality
        return np.concatenate([x, modality_array], axis=1)

    def initialize_data(self, n, modality=0, protocol="random"):
        """Initializes the X data via some provided protocol. Takes care to
        initialize the multi-modal.

        Parameters
        ----------
        n : int
            The number of points to use initially.
        modality : int
            The modality to use during initialization. Defaults to 0, which
            can be assumed to be the low-fidelity experiment inforomation.
        protocol : str, optional
        """

        if protocol == "random":
            X = self.get_random_coordinates(n=n, modality=modality)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")
        self.update_data(X)

    def run_gp_experiment(
        self,
        max_experiments,
        modality_callback=lambda x: 0,
        task_feature=-1,
        svf=None,
        acquisition_function="UCB",
        acquisition_function_kwargs={"beta": 10.0},
        optimize_acqf_kwargs={"q": 1, "num_restarts": 20, "raw_samples": 100},
        optimize_results=False,
        pbar=True,
    ):
        """A special experiment runner which works on multimodal data. The
        user must specify a special callback function which provides the
        modality index as a function of iteration. By default, this is just
        0 for any iteration index (returns the low-fidelity experiment index).
        """

        if len(self.data.history) > 0:
            start = self.data.history[-1]["iteration"] + 1
        else:
            start = 0

        # First, check to see if the data is initialized
        if self.data.X is None:
            raise ValueError("You must initialize starting data first")

        # Run the experiment
        for ii in tqdm(range(start, start + max_experiments), disable=not pbar):
            # Get the data
            X = self.data.X
            Y = self.data.Y

            if X.shape[0] > max_experiments:
                break

            # Simple fitting of a Gaussian process
            # using some pretty simple default values for things, which we
            # can always change later
            if svf:
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

            if target.ndim > 1 and target.shape[1] > 1:
                raise ValueError("Can only predict on a scalar target")
            if target.ndim == 1:
                target = target.reshape(-1, 1)
            gp = EasyMultiTaskGP.from_default(
                X, target, task_feature=task_feature
            )

            # Should be able to change how the gp is fit here
            gp.fit_mll()

            # Get the current modality of the experiment we're currently
            # running
            modality_index = modality_callback(ii)

            # Need to use a posterior transform here to tell the acquisition
            # function how to weight the multiple outputs
            weights = np.zeros(shape=(self.n_modalities,))
            weights[modality_index] = 1.0
            weights = torch.tensor(weights)
            transform = ScalarizedPosteriorTransform(weights=weights)
            acquisition_function_kwargs["posterior_transform"] = transform

            # Ask the model what to do next
            if acquisition_function in ["EI", "qEI"]:
                acquisition_function_kwargs["best_f"] = Y[
                    :, modality_index
                ].max()

            state = ask(
                gp.model,
                acquisition_function,
                bounds=self.properties.experimental_domain,
                acquisition_function_kwargs=acquisition_function_kwargs,
                optimize_acqf_kwargs=optimize_acqf_kwargs,
            )

            # Update the internal data store with the next points
            X2 = state["next_points"]
            self.update_data(X2)

            # Append the history with everything we want to keep
            # Note that the complete state of the GP is saved in the
            # acquisition function model

            d = {
                "iteration": ii,
                "next_points": state["next_points"],
                "value": state["value"],
                "acquisition_function": deepcopy(state["acquisition_function"]),
                "easy_gp": deepcopy(gp),
            }

            self.data.history.append(d)
