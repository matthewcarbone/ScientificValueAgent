from abc import ABC, abstractmethod, abstractproperty

import numpy as np
from attrs import define, field, frozen
from attrs.validators import instance_of, optional

from sva.monty.json import MSONable
from sva.utils import (
    get_coordinates,
    get_latin_hypercube_points,
    get_random_points,
)


@frozen
class ExperimentProperties(MSONable):
    """Defines the core set of experiment properties, which are frozen after
    setting. These are also serializable and cannot be changed after they
    are set."""

    n_input_dim = field(validator=instance_of(int))
    n_output_dim = field(validator=instance_of(int))
    domain = field(validator=optional(instance_of(np.ndarray)))

    def __eq__(self, x):
        if self.n_input_dim != x.n_input_dim:
            return False
        if self.n_output_dim != x.n_output_dim:
            return False
        if (self.domain is None) ^ (x.domain is None):
            return False
        if self.domain is not None and x.domain is not None:
            if self.domain.shape != x.domain:
                return False
            if not np.all(self.domain == x.domain):
                return False
        return True


@define
class Experiment(ABC, MSONable):
    """Abstract base class for a source of truth. These sources of truth are
    for a single modality. The experiment contains 2 attributes. First, a
    catch-all metadata attribute for anything else that should be stored in
    the class. Second and most importantly, a frozen class called
    ExperimentProperties. This contains all of the required parameters for
    the experiment: n_input_dim, n_output_dim and the valid domain (which
    can be None)."""

    metadata = field(factory=dict)

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

    # Passthroughs to the data
    @property
    def X(self):
        return self.data.X

    @property
    def Y(self):
        return self.data.Y

    @property
    def N(self):
        return self.data.N

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
        if self.properties.domain is not None:
            check1 = self.properties.domain[0, :] <= x
            check2 = x <= self.properties.domain[1, :]
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

        return get_random_points(self.properties.domain, n=n)

    def get_latin_hypercube_coordinates(self, n=5):
        """Gets n Latin Hypercube-random points."""

        return get_latin_hypercube_points(self.properties.domain, n)

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

        return get_coordinates(ppd, self.properties.domain)

    def get_domain_mpl_extent(self):
        """This is a helper for getting the "extent" for matplotlib's
        imshow.
        """

        if self.properties.n_input_dim != 2:
            raise NotImplementedError("Only implemented for 2d inputs.")

        x0 = self.properties.domain[0, 0]
        x1 = self.properties.domain[1, 0]

        y0 = self.properties.domain[0, 1]
        y1 = self.properties.domain[1, 1]

        return [x0, x1, y0, y1]

    def __call__(self, x):
        """The result of the experiment."""

        return self.truth(x)
