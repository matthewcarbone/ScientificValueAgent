import numpy as np
from attrs import define, field, validators
from monty.json import MSONable

from sva.experiments.base import (
    NOISE_TYPES,
    ExperimentData,
    ExperimentMixin,
    ExperimentProperties,
)

from .gpax import (
    get_gpax_sinusoidal_dataset,
    high_fidelity_sinusoidal,
    low_fidelity_sinusoidal,
)


@define
class SimpleSigmoid(ExperimentMixin, MSONable):
    """A simple 1d experimental response to a 1d input. This is a sigmoid
    function centered at 0, with a range (-0.5, 0.5). The sharpness of the
    sigmoid function is adjustable by setting the parameter a."""

    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=1,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([-2.0, 2.0]).reshape(2, 1),
        )
    )
    a = field(default=10.0, validator=validators.instance_of(float))
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    data = field(factory=lambda: ExperimentData())

    def _truth(self, x):
        return 2.0 / (1.0 + np.exp(-self.a * x)) - 1.0

    def _dtruth(self, x):
        e = np.exp(-self.a * x)
        d = 1.0 + e
        return 2.0 * self.a * e / d**2


# TODO: this should inherit from base.MultiModalExperiment or whatever
@define
class WavySinusoidalGPax:
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=1,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([0.0, 100.0]).reshape(2, 1),
        )
    )
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    data = field(factory=lambda: ExperimentData())

    @staticmethod
    def get_default_dataset():
        """Returns the dataset used in Maxim Ziatdinov's notebook which can
        be found in his code here: https://github.com/ziatdinovmax/gpax.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            x_train, y_train, x_test_full_range, y_test_full_range
        """
        return get_gpax_sinusoidal_dataset()


@define
class WavySinusoidalGPaxLowFidelity(
    WavySinusoidalGPax, ExperimentMixin, MSONable
):
    def _truth(self, x):
        return low_fidelity_sinusoidal(x, noise=0.0)


@define
class WavySinusoidalGPaxHighFidelity(
    WavySinusoidalGPax, ExperimentMixin, MSONable
):
    def _truth(self, x):
        return high_fidelity_sinusoidal(x, noise=0.0)


@define
class NegatedGramacyLeeFunction(ExperimentMixin, MSONable):
    """Maximum is approximately 0.548563."""

    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=1,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([0.5, 2.5]).reshape(2, 1),
        )
    )
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    data = field(factory=lambda: ExperimentData())

    def _truth(self, x):
        t1 = -np.sin(10.0 * np.pi * x) / 2.0 / (x + 1e-8 * np.sign(x))
        t2 = -((x - 1.0) ** 4)
        return t1 + t2

    def _dtruth(self, x):
        t1 = -4.0 * (x - 1.0) ** 3
        t2 = -5.0 * np.pi * np.cos(10.0 * np.pi * x) / x
        t3 = np.sin(10.0 * np.pi * x) / 2.0 / x**2
        return t1 + t2 + t3


# def _growing_noisy_function_gpytorch(x):
#     return np.cos(x * 2.0 * np.pi) + np.random.normal(size=x.shape) * x**3


# class GrowingNoisyFunctionGPyTorch(ExperimentMixin, MSONable):
#     """

#     train_x = torch.linspace(0, 1, 100)
#     train_y = torch.cos(train_x * 2 * math.pi)
#     + torch.randn(100).mul(train_x.pow(3) * 1.)

#     fig, ax = plt.subplots(1, 1, figsize=(5, 3))
#     ax.scatter(train_x, train_y, c='k', marker='.', label="Data")
#     ax.set(xlabel="x", ylabel="y")
#     """

#     @staticmethod
#     def get_default_dataset():
#         np.random.seed(123)
#         train_x = np.linspace(0, 1, 100)
#         train_y = _growing_noisy_function_gpytorch(train_x)
#         return train_x.reshape(-1, 1), train_y.reshape(-1, 1)

#     def _truth(self, x):
#         y = _growing_noisy_function_gpytorch(x)
#         return y.reshape(-1, 1)


@define
class Simple2d(ExperimentMixin, MSONable):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([[-4.0, 5.0], [-5.0, 4.0]]).T,
        )
    )
    noise = field(default=None, validator=validators.instance_of(NOISE_TYPES))
    data = field(factory=lambda: ExperimentData())

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        res = (1 - x / 3.0 + x**5 + y**5) * np.exp(
            -(x**2) - y**2
        ) + 2.0 * np.exp(-((x - 2) ** 2) - (y + 4) ** 2)
        return res.reshape(-1, 1)
