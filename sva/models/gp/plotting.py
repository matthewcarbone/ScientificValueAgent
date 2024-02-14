import numpy as np


def plot_1d_gp(
    ax,
    train_X,
    train_Y,
    train_Yvar=None,
    test_X=None,
    gp_samples=None,
    errorbar_plot_kwargs={
        "linewidth": 0.0,
        "marker": "s",
        "ms": 1.0,
        "capthick": 0.3,
        "capsize": 2.0,
        "elinewidth": 0.3,
    },
):
    """Summary
    TODO: do the docs here

    Parameters
    ----------
    ax : TYPE
        Description
    train_X : TYPE
        Description
    train_Y : TYPE
        Description
    train_Yvar : None, optional
        Description
    test_X : None, optional
        Description
    gp_samples : None, optional
        Description
    errorbar_plot_kwargs : dict, optional
        Description
    """

    if train_Yvar is not None:
        if not isinstance(train_Yvar, float):
            train_Yvar = train_Yvar.squeeze()

        ax.errorbar(
            train_X.squeeze(),
            train_Y.squeeze(),
            yerr=2 * np.sqrt(train_Yvar),
            color="black",
            zorder=3,
            **errorbar_plot_kwargs,
        )
    else:
        ax.scatter(
            train_X.squeeze(),
            train_Y.squeeze(),
            color="black",
            zorder=3,
            s=errorbar_plot_kwargs.get("ms", 1.0),
        )

    for sample in gp_samples:
        ax.plot(test_X.squeeze(), sample, "r-", linewidth=0.1)
