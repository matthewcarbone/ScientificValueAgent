import numpy as np


def x_distance_metric(data, optimum):
    """A simplistic metric that takes a data dictionary such that its keys
    are names of acquisition functions, and its values are list of
    Campaign objects. The minimum L2 distance between the true_optima and
    sampled point is calculated at every step.

    Parameters
    ----------
    data : dict
    optimum : np.ndarray
        The optimum x-value location.

    Returns
    -------
    dict
    """

    metrics = {}
    for acqf, exp_list in data.items():
        tmp = []
        for campaign in exp_list:
            X = campaign.data.X
            distance = np.sqrt(np.sum((X - optimum) ** 2, axis=1))
            for ii in range(1, len(distance)):
                distance[ii] = min(distance[ii], distance[ii - 1])
            tmp.append(distance)
        metrics[acqf] = np.array(tmp)

    return metrics
