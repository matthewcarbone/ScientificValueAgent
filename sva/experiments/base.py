from abc import ABC, abstractmethod, abstractproperty
from typing import Optional

from monty.json import MSONable
import numpy as np

from sva.utils import get_random_points


class ExperimentData(MSONable):
    """Container for sampled data during experiments. Does essentially nothing
    except hold the data and provide methods for dealing with it, including
    updating it and saving it to disk. Note that data contained here must
    always be two-dimensional (x.ndim == 2).

    Attributes
    ----------
    data : np.ndarray, optional
    """

    @property
    def N(self):
        return self._X.shape[0]

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def __init__(self, X=None, Y=None):
        self._X = X
        self._Y = Y

    def update_X_(self, X):
        """Updates the current input data with new inputs.

        Parameters
        ----------
        X : np.ndarray
            Two-dimensional data to update the X values with.
        """

        assert X.ndim == 2
        if self._X is not None:
            self._X = np.concatenate([self._X, X], axis=0)
        else:
            self._X = X

    def update_Y_(self, experiment):
        """Updates the current output data with new outputs.

        Parameters
        ----------
        callable
            Must be callable and take a 2d input array as input.
        """

        assert self._X is not None

        if self._Y is not None:
            diff = self._X.shape[0] - self._Y.shape[0]
            new_Y = experiment(self._X[-diff:])
            self._Y = np.concatenate([self._Y, new_Y], axis=0)
        else:
            self._Y = experiment(self._X)


class Experiment(ABC):
    """Abstract base class for a source of truth. These sources of truth are
    for a single modality."""

    @abstractproperty
    def valid_domain(self) -> Optional[np.ndarray]:
        """The domain of valid points in the input space. The returned array
        should be of shape (2, d), where d is the number of dimensions in the
        input space."""

        raise NotImplementedError

    @abstractproperty
    def experimental_domain(self) -> Optional[np.ndarray]:
        """The domain of valid points for the experimental campaigns. This is
        always a subset or equal to the valid_domain."""

        raise NotImplementedError

    @abstractproperty
    def n_input_dim(self) -> int:
        raise NotImplementedError

    @abstractproperty
    def n_output_dim(self) -> int:
        raise NotImplementedError

    @property
    def expense(self) -> float:
        """The user-defined expense of the experiment. Defaults to 1 "unit" of
        time, money, or whatever."""

        return 1.0

    @abstractmethod
    def _truth(self, x: np.ndarray) -> np.ndarray:
        """Vectorized truth function. Should return the value of the truth
        function for all rows of the provided x input."""

        raise NotImplementedError

    def _dtruth(self, x: np.ndarray) -> np.ndarray:
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
        if not x.shape[1] == self.n_input_dim:
            raise ValueError(f"x second dimension must be {self.n_input_dim}")

        # Ensure that the input is in the bounds
        # Domain being None implies the domain is the entirety of the reals
        # This is a fast way to avoid the check
        if self.valid_domain is not None:
            check1 = self.valid_domain[0, :] <= x
            check2 = x <= self.valid_domain[1, :]
            if not np.all(check1 & check2):
                raise ValueError("Some inputs x were not in the domain")

    def _validate_output(self, y):

        # Ensure y has the right shape
        if not y.ndim == 2:
            raise ValueError("y must have shape (N, d')")

        # Assert that the second dimension of the output has the right size for
        # the chosen experiment
        if not y.shape[1] == self.n_output_dim:
            raise ValueError(f"y second dimension must be {self.n_output_dim}")

    def truth(self, x: np.ndarray) -> np.ndarray:
        """Access the results of an "experiment"."""

        self._validate_input(x)
        y = self._truth(x)
        self._validate_output(y)
        return y

    def dtruth(self, x: np.ndarray) -> np.ndarray:
        """Access the derivative of the results of the experiment."""

        self._validate_input(x)
        y = self._dtruth(x)
        self._validate_output(y)
        return y

    def random_inputs(self, n=1, seed=None):
        """Runs n random input points."""

        return get_random_points(self.experimental_domain, n=n, seed=seed)

    def update_data_(self, x):
        self._data.update_X_(x)
        self._data.update_Y_(self)

    def __call__(self, x):
        """Alias for truth."""

        return self.truth(x)
