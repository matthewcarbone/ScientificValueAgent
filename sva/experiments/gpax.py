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


def _f(x):
    return 2.0 * np.sin(x / 10.0) + 0.5 * np.sin(x / 2.0) + 0.1 * x


def _low_fidelity_sinusoidal(x, noise=0.0):
    return _f(x) + np.random.normal(0, noise, x.shape)


def _high_fidelity_sinusoidal(x, noise=0.0):
    return (
        1.5 * _f(x)
        + np.sin(x / 15.0)
        - 5.0
        + np.random.normal(0.0, noise, x.shape)
    )


def get_gpax_sinusoidal_dataset():
    np.random.seed(1)  # for reproducibility

    # Fidelity 1 - "theoretical model"
    X1 = np.linspace(0, 100, 100)
    y1 = _low_fidelity_sinusoidal(X1)

    # Fidelity 2 - "experimental measurements"
    X2 = np.concatenate(
        [np.linspace(0, 25, 20), np.linspace(75, 100, 20)]
    )  # only have data for some frequencies
    y2 = _high_fidelity_sinusoidal(X2, noise=0.3)

    # Ground truth for Fidelity 2
    X_full_range = np.linspace(0, 100, 200)
    y2_true = _high_fidelity_sinusoidal(X_full_range)

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
