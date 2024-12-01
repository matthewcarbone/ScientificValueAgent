import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties


def blackboxfunc(X):
    x0, x1 = X[:, 0], X[:, 1]
    eq = (
        np.sin(x0 * x0) * np.cos(x1 * x1)
        + 0.7 * x1 * x0
        + 0.2 * np.sin(x0 * x1)
    )
    results = np.zeros((X.shape[0], 5))
    condition1 = (eq < 0.1) & (x0 < 0.1)
    results[np.where(condition1)[0]] = np.array([1, 0, 0, 0, 0])
    condition2 = (eq < 0.3) & (x1 > 0.3) & ~condition1
    results[np.where(condition2)[0]] = np.array([0, 1, 0, 0, 0])
    condition3 = (
        (eq < 0.5) & (x1 < 0.5) & (x0 > 0.5) & ~(condition1 | condition2)
    )
    results[np.where(condition3)[0]] = np.array([0, 0, 1, 0, 0])
    condition4 = (
        (eq < 0.8) & (x1 < 0.7) & ~(condition1 | condition2 | condition3)
    )
    results[np.where(condition4)[0]] = np.array([0, 0, 0, 1, 0])
    condition5 = ~(condition1 | condition2 | condition3 | condition4)
    results[np.where(condition5)[0]] = np.array([0, 0, 0, 0, 1])
    return results


@define
class Simple5Phase(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            domain=np.array([[0.0, 1.0], [0.0, 1.0]]).T,
        )
    )

    def _truth(self, X):
        return blackboxfunc(X)
