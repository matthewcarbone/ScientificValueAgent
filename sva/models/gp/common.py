"""Common helper functions for all Gaussian Process models."""


def set_eval_(gp):
    """Sets a Gaussian Process model to evaluation mode.

    Parameters
    ----------
    gp : botorch.models.SingleTaskGP
    """

    gp.eval()
    gp.likelihood.eval()


def set_train_(gp):
    """Sets a Gaussian Process model to training mode.

    Parameters
    ----------
    gp : botorch.models.SingleTaskGP
    """

    gp.train()
    gp.likelihood.train()


def get_model_hyperparameters(model):
    """Iterates through a torch-like object which has a named_parameters()
    method and returns a dictionary of all of the parameters, where values
    are in numpy format.

    Parameters
    ----------
    model
        The model with named_parameters() defined on it.

    Returns
    -------
    dict
    """

    d = {}
    for p in model.named_parameters():
        p0 = str(p[0])
        p1 = p[1].detach().numpy()
        d[p0] = p1
    return d
