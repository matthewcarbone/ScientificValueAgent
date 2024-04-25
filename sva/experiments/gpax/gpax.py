"""
MIT License

Copyright (c) 2021 Maxim Ziatdinov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from attrs import define, field

from sva.monty.json import MSONable

from ..base import ExperimentProperties, MultimodalExperimentMixin
from ..campaign import MultimodalCampaignMixin


def _f(x):
    return 2.0 * np.sin(x / 10.0) + 0.5 * np.sin(x / 2.0) + 0.1 * x


def low_fidelity_sinusoidal(x, noise=0.0):
    return _f(x) + np.random.normal(0, noise, x.shape)


def high_fidelity_sinusoidal(x, noise=0.0):
    return (
        1.5 * _f(x)
        + np.sin(x / 15.0)
        - 5.0
        + np.random.normal(0.0, noise, x.shape)
    )


def get_gpax_sinusoidal_dataset(
    n_low=101, n_high=5, init_right=True, low_max=100.0
):
    np.random.seed(1)  # for reproducibility

    # Fidelity 1 - "theoretical model"
    assert 1.0 < low_max <= 100.0
    X1 = np.linspace(0, low_max, n_low)
    y1 = low_fidelity_sinusoidal(X1)

    # Fidelity 2 - "experimental measurements"
    if init_right:
        X2 = np.concatenate(
            [np.linspace(0, 25, n_high), np.linspace(75, 100, n_high)]
        )  # only have data for some frequencies
    else:
        X2 = np.linspace(0, 25, n_high)
    y2 = high_fidelity_sinusoidal(X2, noise=0.3)

    # Ground truth for Fidelity 2
    X_full_range = np.linspace(0, 100, 200)
    y2_true = high_fidelity_sinusoidal(X_full_range)

    # Add fidelity indices
    X = np.vstack(
        (
            np.column_stack(
                (X1, np.zeros_like(X1))
            ),  # add indices associated with the fidelity
            np.column_stack((X2, np.ones_like(X2))),
        )  # add indices associated with the fidelity
    )

    # We will pass target values to GP as a single array
    y = np.concatenate([y1, y2]).squeeze()

    return X, y, X_full_range, y2_true


@define
class GPaxTwoModalityTest(
    MultimodalExperimentMixin, MultimodalCampaignMixin, MSONable
):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=1,
            n_output_dim=1,
            valid_domain=None,
            experimental_domain=np.array([[0.0, 100.0]]).T,
        )
    )
    noise = None
    n_modalities = 2

    def initialize_data_from_default(self, *args, **kwargs):
        X, _, _, _ = get_gpax_sinusoidal_dataset(*args, **kwargs)
        self.update_data(X)

    def _truth(self, x):
        if self.noise is not None:
            raise ValueError(
                "Noise is not None, but is not allowed in this experiment"
            )

        low_y = None
        low_ii = np.where(x[:, -1] == 0)[0]
        if len(low_ii) > 0:
            low_y = low_fidelity_sinusoidal(x[low_ii, :-1])

        high_y = None
        high_ii = np.where(x[:, -1] == 1)[0]
        if len(high_ii) > 0:
            high_y = high_fidelity_sinusoidal(x[high_ii, :-1])

        to_return = np.empty((x.shape[0], self.properties.n_output_dim))
        if low_y is not None:
            to_return[low_ii, :] = low_y
        if high_y is not None:
            to_return[high_ii, :] = high_y
        return to_return
