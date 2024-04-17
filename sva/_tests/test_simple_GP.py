import numpy as np
import pytest

# Import the various experiments we need for the notebook
from sva.experiments import SimpleSigmoid

# Import the helper functions for Gaussian Processes
from sva.models.gp import EasyFixedNoiseGP, EasySingleTaskGP

# Other utilities
from sva.utils import random_indexes, seed_everything


def test_EasySingleTaskGP():
    seed_everything(124)
    sigmoid = SimpleSigmoid(a=10.0, noise=lambda x: 0.2 * np.abs(x))
    x = np.linspace(-1, 1, 500).reshape(-1, 1)
    y, _ = sigmoid(x.reshape(-1, 1))
    train_indexes = random_indexes(x.shape[0], samples=10)
    x_train = x[train_indexes, :]
    y_train = y[train_indexes, :]

    _, _ = sigmoid.truth(x.reshape(-1, 1))

    gp1 = EasySingleTaskGP.from_default(x_train, y_train)
    gp1.fit_mll()
    gp1.sample(x, samples=200)


def test_EasyFixedNoiseGP():
    seed_everything(124)
    sigmoid = SimpleSigmoid(a=10.0, noise=lambda x: 0.2 * np.abs(x))
    x = np.linspace(-1, 1, 500).reshape(-1, 1)
    y, _ = sigmoid(x.reshape(-1, 1))
    train_indexes = random_indexes(x.shape[0], samples=10)
    x_train = x[train_indexes, :]
    y_train = y[train_indexes, :]

    gp2 = EasyFixedNoiseGP.from_default(
        x_train, y_train, Yvar=(0.2 * np.abs(x_train)) ** 2
    )
    gp2.fit_mll()
    gp2.sample(x, samples=200)


@pytest.mark.filterwarnings("ignore")
def test_EasyFixedNoiseGP2():
    seed_everything(124)
    sigmoid = SimpleSigmoid(a=10.0, noise=lambda x: 0.2 * np.abs(x))
    x = np.linspace(-1, 1, 500).reshape(-1, 1)
    y, _ = sigmoid(x.reshape(-1, 1))
    train_indexes = random_indexes(x.shape[0], samples=10)
    x_train = x[train_indexes, :]
    y_train = y[train_indexes, :]
    gp3 = EasyFixedNoiseGP.from_default(
        x_train, y_train, Yvar=(1e-4 * np.abs(x_train)) ** 2
    )
    gp3.fit_mll()
    gp3.sample(x, samples=200)
