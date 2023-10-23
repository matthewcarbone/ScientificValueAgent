from abc import ABC, abstractmethod, abstractproperty
from typing import Optional

import numpy as np


class Truth(ABC):
    """Abstract base class for a source of truth. These sources of truth are
    for a single modality."""

    @abstractproperty
    def domain(self) -> Optional[np.ndarray]:
        """The domain of valid points in the input space. The returned array
        should be of shape (2, d), where d is the number of dimensions in the
        input space."""

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
        if self.domain is not None:
            check1 = self.domain[0, :] <= x
            check2 = x <= self.domain[1, :]
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


class SimpleSigmoid(Truth):
    """A simple 1d experimental response to a 1d input. This is a sigmoid
    function centered at 0, with a range (-0.5, 0.5). The sharpness of the
    sigmoid function is adjustable by setting the parameter a."""

    n_input_dim = 1
    n_output_dim = 1
    # domain = np.array([-np.inf, np.inf]).reshape(2, 1)
    domain = None  # Equivalent to the above

    def __init__(self, a: float = 10.0) -> None:
        self._a = a

    def _truth(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self._a * x)) - 0.5

    def _dtruth(self, x: np.ndarray) -> np.ndarray:
        d = 1 + np.exp(-self._a * x)
        return self._a * np.exp(-self._a * x) / d**2
