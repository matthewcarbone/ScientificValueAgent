import numpy as np
from attrs import define, field

from ..base import Experiment, ExperimentProperties


@define
class Simple2d(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            n_output_dim=1,
            domain=np.array([[-4.0, 5.0], [-5.0, 4.0]]).T,
        )
    )

    true_optima = np.array([2.0, -4.0])

    def _truth(self, X):
        x = X[:, 0]
        y = X[:, 1]
        res = (1 - x / 3.0 + x**5 + y**5) * np.exp(
            -(x**2) - y**2
        ) + 2.0 * np.exp(-((x - 2) ** 2) - (y + 4) ** 2)
        return res.reshape(-1, 1)

    @staticmethod
    def compute_distance_metric(data):
        """A simplistic metric that takes a data dictionary such that its keys
        are names of acquisition functions, and its values are list of
        Campaign objects. The minimum L2 distance between the true_optima and the
        sampled point is calculated at every step."""

        optima = Simple2d.true_optima

        metrics = {}
        for acqf, exp_list in data.items():
            tmp = []
            for campaign in exp_list:
                X = campaign.data.X
                distance = np.sqrt(np.sum((X - optima) ** 2, axis=1))
                for ii in range(1, len(distance)):
                    distance[ii] = min(distance[ii], distance[ii - 1])
                tmp.append(distance)
            metrics[acqf] = np.array(tmp)

        return metrics
