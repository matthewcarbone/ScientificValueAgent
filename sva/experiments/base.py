from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from attrs import define, field, frozen, validators

from sva.monty.json import MSONable
from sva.utils import (
    get_coordinates,
    get_latin_hypercube_points,
    get_random_points,
)


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


@define
class ExperimentMixin(ABC, MSONable):
    """Abstract base class for a source of truth. These sources of truth are
    for a single modality."""

    metadata = field(factory=dict)

    @classmethod
    def from_random(cls, n=3):
        klass = cls()
        klass.initialize_data(n=n, protocol="random")
        return klass

    @classmethod
    def from_LatinHypercube(cls, n=5):
        klass = cls()
        klass.initialize_data(n=n, protocol="LatinHypercube")
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

    def truth(self, x: np.ndarray) -> np.ndarray:
        """Access the noiseless results of an "experiment"."""

        self._validate_input(x)
        y = self._truth(x)
        self._validate_output(y)
        return y

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

    def get_latin_hypercube_coordinates(self, n=5):
        """Gets n Latin Hypercube-random points."""

        domain = self.properties.experimental_domain
        return get_latin_hypercube_points(domain, n)

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

    def initialize_data(self, protocol="random", **kwargs):
        """Initializes the X data via some provided protocol.

        Parameters
        ----------
        protocol : str, optional
            The method for using to initialize the data.
        kwargs
            To pass to the particular method.
        """

        if protocol == "random":
            X = self.get_random_coordinates(**kwargs)
        elif protocol == "LatinHypercube":
            X = self.get_latin_hypercube_coordinates(**kwargs)
        elif protocol == "dense":
            X = self.get_dense_coordinates(**kwargs)
        else:
            raise NotImplementedError(f"Unknown provided protocol {protocol}")

        self.update_data(X)

    def __call__(self, x):
        """The (possibly noisy) result of the experiment. Also returns the
        variance of the observations."""

        return self.truth(x)
