import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import axes_grid1


def get_function_from_signature(signature):
    """Parases a function of the form module.submodule:function to import
    and get the actual function as defined.

    Parameters
    ----------
    signature : str

    Returns
    -------
    callable
    """

    module, function = signature.split(":")
    exec(f"from {module} import {function}")
    return eval(function)


def set_defaults(labelsize=12, dpi=250):
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["text.usetex"] = True
    plt.rc("xtick", labelsize=labelsize)
    plt.rc("ytick", labelsize=labelsize)
    plt.rc("axes", labelsize=labelsize)
    mpl.rcParams["figure.dpi"] = dpi
    plt.rcParams["figure.figsize"] = (3, 2)


def set_grids(
    ax,
    minorticks=True,
    grid=False,
    bottom=True,
    left=True,
    right=True,
    top=True,
):
    if minorticks:
        ax.minorticks_on()

    ax.tick_params(
        which="both",
        direction="in",
        bottom=bottom,
        left=left,
        top=top,
        right=right,
    )

    if grid:
        ax.grid(which="minor", alpha=0.2, linestyle=":")
        ax.grid(which="major", alpha=0.5)


def legend_without_duplicate_labels(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l)
        for i, (h, l) in enumerate(zip(handles, labels))
        if l not in labels[:i]
    ]
    ax.legend(*zip(*unique), **kwargs)


def add_colorbar(
    im, aspect=10, pad_fraction=0.5, integral_ticks=None, **kwargs
):
    """Add a vertical color bar to an image plot."""

    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    cbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)
    if integral_ticks is not None:
        L = len(integral_ticks)
        cbar.set_ticks(
            [
                cbar.vmin
                + (cbar.vmax - cbar.vmin) / L * ii
                - (cbar.vmax - cbar.vmin) / L / 2.0
                for ii in range(1, L + 1)
            ]
        )
        cbar.set_ticklabels(integral_ticks)
    return cbar
