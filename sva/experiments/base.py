from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from attrs import define, field, frozen, validators

from sva.monty.json import MSONable
from sva.utils import get_coordinates, get_random_points


@define
class ExperimentData(MSONable):
    """Container for sampled data during experiments. This is a serializable
    abstraction over the data used during the experiments. This includes
    updating it and saving it to disk. Note that data contained here must
    always be two-dimensional (x.ndim == 2)."""

    X: np.ndarray = field(default=None)
    Y: np.ndarray = field(default=None)
    Yvar: np.ndarray = field(default=None)

    @property
    def N(self):
        return self.X.shape[0]

    @property
    def is_initialized(self):
        if self.X is None and self.Y is None:
            return False
        return True

    def update_X(self, X):
        """Updates the current input data with new inputs.

        Parameters
        ----------
        X : np.ndarray
            Two-dimensional data to update the X values with.
        """

        assert X.ndim == 2
        if self.X is not None:
            self.X = np.concatenate([self.X, X], axis=0)
        else:
            self.X = X

    def update_Y(self, experiment):
        """Updates the current output data with new outputs.

        Parameters
        ----------
        callable
            Must be callable and take a 2d input array as input.
        """

        assert self.X is not None

        if self.Y is not None:
            diff = self.X.shape[0] - self.Y.shape[0]
            new_Y, new_Yvar = experiment(self.X[-diff:])
            self.Y = np.concatenate([self.Y, new_Y], axis=0)
        else:
            self.Y, new_Yvar = experiment(self.X)

        # If we compute variances
        if new_Yvar is not None:
            # And we have existing variances
            if self.Yvar is not None:
                self.Yvar = np.concatenate([self.Yvar, new_Yvar], axis=0)

            # Otherwise, we don't have existing variances
            else:
                self.Yvar = new_Yvar

    def __eq__(self, exp):
        # XOR for when things aren't initialized
        if (exp.X is None) ^ (self.X is None):
            return False
        if (exp.Y is None) ^ (self.Y is None):
            return False
        if (exp.Yvar is None) ^ (self.Yvar is None):
            return False

        if exp.X is not None and self.X is not None:
            if exp.X.shape != self.X.shape:
                return False
            if not np.all(exp.X == self.X):
                return False
        if exp.Y is not None and self.Y is not None:
            if exp.Y.shape != self.Y.shape:
                return False
            if not np.all(exp.Y == self.Y):
                return False
        if exp.Yvar is not None and self.Yvar is not None:
            if exp.Yvar.shape != self.Yvar.shape:
                return False
            if not np.all(exp.Yvar == self.Yvar):
                return False

        return True


@define
class ExperimentHistory(MSONable):
    """Container for the history of the experiment. Note that this is not
    MSONable as it will contain a variety of objects that can only be
    pickled, so that is the protocol we'll use for this."""

    history: list = field(factory=list)

    def append(self, x):
        assert isinstance(x, dict)
        self.history.append(x)

    def extend(self, x):
        assert all([isinstance(xx, dict) for xx in x])
        self.history.extend(x)

    def __len__(self):
        return len(self.history)

    def __getitem__(self, ii):
        return self.history[ii]

    def __eq__(self, x):
        if len(self.history) != len(x.history):
            return False
        for x1, x2 in zip(self.history, x.history):
            if len(x1) != len(x2):
                return False
            if x1.keys() != x2.keys():
                return False
            for key in x1.keys():
                # Not really sure how to do this comparison yet
                if key == "state":
                    continue
                v1 = x1[key]
                v2 = x2[key]
                if key == "easy_gp":
                    if v1 != v2:
                        return False
                if isinstance(v1, (np.ndarray, torch.Tensor)) and np.all(
                    v1 != v2
                ):
                    return False
        return True


@frozen
class ExperimentProperties(MSONable):
    """Defines the core set of experiment properties, which are frozen after
    setting. These are also serializable and cannot be changed after they
    are set."""

    n_input_dim = field(validator=validators.instance_of(int))
    n_output_dim = field(validator=validators.instance_of(int))
    valid_domain = field(
        validator=validators.instance_of((type(None), np.ndarray))
    )
    experimental_domain = field(validator=validators.instance_of(np.ndarray))

    def __eq__(self, x):
        if self.n_input_dim != x.n_input_dim:
            return False
        if self.n_output_dim != x.n_output_dim:
            return False
        if (self.valid_domain is None) ^ (x.valid_domain is None):
            return False
        if self.valid_domain is not None and x.valid_domain is not None:
            if self.valid_domain.shape != x.valid_domain:
                return False
            if not np.all(self.valid_domain == x.valid_domain):
                return False
        if self.experimental_domain.shape != x.experimental_domain.shape:
            return False
        if not np.all(self.experimental_domain == x.experimental_domain):
            return False
        return True


NOISE_TYPES = (Callable, float, np.ndarray, list, type(None))


@define
class ExperimentMixin(ABC, MSONable):
    """Abstract base class for a source of truth. These sources of truth are
    for a single modality."""

    metadata = field(factory=lambda: defaultdict(list))
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    data = field(factory=lambda: ExperimentData())
    history = field(factory=lambda: ExperimentHistory())

    @classmethod
    def from_random(cls, n=3):
        klass = cls()
        klass.initialize_data(n=n, protocol="random")
        return klass

    @classmethod
    def from_dense(cls, ppd=10):
        klass = cls()
        klass.initialize_data(ppd=ppd, protocol="dense")
        return klass

    @abstractproperty
    def properties(self): ...

    @abstractmethod
    def _truth(self, _):
        """Vectorized truth function. Should return the value of the truth
        function for all rows of the provided x input."""

        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    def _variance(self, _):
        """Vectorized truth of the variance of the experiment. Distinct from
        the 'noise' property, this method returns the variance (one standard
        deviation^2) on the observed results. By default, this returns None,
        which is distinct from 0 (there's a difference between having a
        completely precise observation, returning 0, and having no knowledge
        of the noise, returning None)."""

        return None

    def _dtruth(self):
        """The derivative of the truth value with respect to x. May not be
        implemented, depending on the function."""

        raise NotImplementedError(
            "The derivative of the truth function is not implemented for "
            "this experiment."
        )

    def _validate_input(self, x):
        # Ensure x has the right shape
        if not x.ndim == 2:
            raise ValueError("x must have shape (N, d)")

        # Ensure that the second dimension of x is the right size for the
        # chosen experiment
        if not x.shape[1] == self.properties.n_input_dim:
            raise ValueError(
                f"x second dimension must be {self.properties.n_input_dim}"
            )

        # Ensure that the input is in the bounds
        # Domain being None implies the domain is the entirety of the reals
        # This is a fast way to avoid the check
        if self.properties.valid_domain is not None:
            check1 = self.properties.valid_domain[0, :] <= x
            check2 = x <= self.properties.valid_domain[1, :]
            if not np.all(check1 & check2):
                raise ValueError("Some inputs x were not in the domain")

    def _validate_output(self, y):
        # Ensure y has the right shape
        if not y.ndim == 2:
            raise ValueError("y must have shape (N, d')")

        # Assert that the second dimension of the output has the right size for
        # the chosen experiment
        if not y.shape[1] == self.properties.n_output_dim:
            raise ValueError(
                f"y second dimension must be {self.properties.n_output_dim}"
            )

    def variance(self, x: np.ndarray, validate=False):
        if validate:
            self._validate_input(x)
        return self._variance(x)

    def truth(self, x: np.ndarray) -> np.ndarray:
        """Access the noiseless results of an "experiment"."""

        self._validate_input(x)
        y = self._truth(x)
        self._validate_output(y)
        # No need to validate x again since it's validated above
        yvar = self._variance(x)
        return y, yvar

    def dtruth(self, x: np.ndarray) -> np.ndarray:
        """Access the derivative of the noiseless results of the experiment."""

        self._validate_input(x)
        y = self._dtruth(x)
        self._validate_output(y)
        return y

    def get_random_coordinates(self, n=1):
        """Runs n random input points."""

        domain = self.properties.experimental_domain
        return get_random_points(domain, n=n)

    def get_dense_coordinates(self, ppd):
        """Gets a set of dense coordinates.

        Parameters
        ----------
        ppd : int or list
            Points per dimension.

        Returns
        -------
        np.ndarray
        """

        domain = self.properties.experimental_domain
        return get_coordinates(ppd, domain)

    def get_experimental_domain_mpl_extent(self):
        """This is a helper for getting the "extent" for matplotlib's
        imshow.
        """

        if self.properties.n_input_dim != 2:
            raise NotImplementedError(
                "get_mpl_extent only implemented for 2d inputs."
            )

        x0 = self.properties.experimental_domain[0, 0]
        x1 = self.properties.experimental_domain[1, 0]

        y0 = self.properties.experimental_domain[0, 1]
        y1 = self.properties.experimental_domain[1, 1]

        return [x0, x1, y0, y1]

    def update_data(self, x):
        """Helper method for updating the data attribute with new data.

        Parameters
        ----------
        x : numpy.ndarray
        """

        self.data.update_X(x)
        self.data.update_Y(self)

    def initialize_data(self, n, protocol="random"):
        """Initializes the X data via some provided protocol.

        Parameters
        ----------
        n : int
            The number of points to use initially.
        protocol : str, optional
            The method for using to initialize the data.
        """

        if protocol == "random":
            X = self.get_random_coordinates(n=n)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")

        self.update_data(X)

    def __call__(self, x):
        """The (possibly noisy) result of the experiment. Also returns the
        variance of the observations."""

        y, yvar = self.truth(x)

        if self.noise is None:
            return y, yvar

        if isinstance(self.noise, Callable):
            noise = self.noise(x)
            return y + np.random.normal(scale=noise, size=y.shape), yvar

        if isinstance(self.noise, float):
            pass
        elif isinstance(self.noise, np.ndarray):
            assert self.noise.ndim == 1
            assert len(self.noise) == self.properties.n_output_dim
        elif isinstance(self.noise, list):
            assert len(self.noise) == self.properties.n_output_dim
        else:
            raise ValueError("Incompatible noise type")

        return y + np.random.normal(scale=self.noise, size=y.shape), yvar


class MultimodalExperimentMixin(ExperimentMixin):
    @abstractmethod
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

    def initialize_data(self, modality=0, protocol="random", **kwargs):
        """Initializes the X data via some provided protocol. Takes care to
        initialize the multi-modal data with the provided modality.

        Parameters
        ----------
        modality : int
            The modality to use during initialization. Defaults to 0, which
            can be assumed to be the low-fidelity experiment inforomation.
        protocol : str, optional
            Can be one of either "random" or "dense".
        kwargs
            Keyword arguments to pass to the particular "getter". For example
            if protocol == "random", one needs to pass `n=...` (the number
            of points to sample). If protocol == "dense", one needs to pass
            `ppd=...` (the number of points per dimension on a dense grid).

        Note
        ----
        You can call initialize_data multiple times using different modalities.
        """

        if protocol == "random":
            X = self.get_random_coordinates(**kwargs, modality=modality)
        elif protocol == "dense":
            X = self.get_dense_coordinates(**kwargs, modality=modality)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")
        self.update_data(X)
