from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from yaml import safe_load

from sva.monty.json import load_anything


def load_hydra_result(path, load_configs=False, load_logs=False):
    """Loads the result of the SVA hydra run. The provided path should be
    to that of the experiment json file. Everything else can be inferred
    from that."""

    d = path.parent

    result = {"campaign": load_anything(path)}

    if load_configs:
        result["config"] = safe_load(open(d / ".hydra" / "config.yaml", "r"))
        result["hydra"] = safe_load(open(d / ".hydra" / "hydra.yaml", "r"))
        result["overrides"] = safe_load(
            open(d / ".hydra" / "overrides.yaml", "r")
        )

    if load_logs:
        with open(d / "log.out", "r") as f:
            result["stdout"] = f.readines()

        with open(d / "log.err", "r") as f:
            result["stderr"] = f.readlines()

        debug_path = d / "log.debug"
        if debug_path.exists():
            with open(debug_path, "r") as f:
                result["debug"] = f.readlines()

    return result


def read_data(path, load_configs=False, load_logs=False):
    """Reads all results into memory recursively from the provided path.
    Searches for all files matching the .json pattern and loads a re-hydrated
    experiment, as well as all config and log information if specified."""

    # Nested default dicts
    # Structure should be like
    # results[experiment_name][acquisition_function_name] is of type list
    results = defaultdict(lambda: defaultdict(list))

    # Find all of the files that match *.json pattern recursively
    # We know that in that directory, there is a hydra config file as well
    paths = list(Path(path).rglob("*.json"))

    for p in tqdm(paths):
        r = load_hydra_result(p, load_configs=load_configs, load_logs=load_logs)
        experiment_name = r["campaign"].experiment.name
        policy_name = r["campaign"].policy.name
        if not load_configs and not load_logs:
            r = r["campaign"]
        results[experiment_name][policy_name].append(r)

    return results


def sort_keys(keys):
    """Sorts the acquisition function keys returned by read_data in order
    of increasing exploitation (mostly)."""

    new_keys = []
    if "RandomPolicy" in keys:
        new_keys.append("RandomPolicy")
    if "GridPolicy" in keys:
        new_keys.append("GridPolicy")
    if "MaxVar" in keys:
        new_keys.append("MaxVar")
    # everything else should be UCB
    remaining_keys = [k for k in keys if "UCB" in k]
    remaining_keys.sort(key=lambda x: -float(x.split("-")[1]))
    new_keys.extend(remaining_keys)
    if "EI" in keys:
        new_keys.append("EI")
    return new_keys


# def interpolant_2d(
#     X,
#     grid_x,
#     grid_y,
#     phase_truth,
#     interpolation_method="linear",
# ):
#     """Returns a 2-dimensional interpolant of the data.
#
#     Parameters
#     ----------
#     X : np.ndarray
#         The points on the grid, of shape (n x d).
#     grid_points : int
#         The number of grid points to use for the linear interpolant.
#     phase_truth : callable
#         A function that takes as input meshgrids x and y and returns an array
#         containing the phase proportions of phase 1.
#     interpolation_method : str, optional
#         The interpolation method to pass to ``griddata``. Recommendation is to
#         use "linear", "nearest" and "cubic".
#
#     No Longer Returned
#     ------------------
#     np.ndarray, np.ndarray
#         The "true" (dense grid) and interpolated (sampled points) results.
#     """
#
#     g = np.linspace(0, 1, grid_points)
#     dense_x, dense_y = np.meshgrid(g, g)
#     space = np.vstack([dense_x.ravel(), dense_y.ravel()]).T
#     true = phase_truth(space[:, 0], space[:, 1])
#     known = phase_truth(X[:, 0], X[:, 1])
#     interpolated = griddata(
#         X, known, (dense_x, dense_y), method=interpolation_method
#     )
#     return true.reshape(grid_points, grid_points), interpolated
#
