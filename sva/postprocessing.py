from collections import defaultdict
from functools import cache
from pathlib import Path

from monty.json import load
from tqdm import tqdm
from yaml import safe_load


def load_hydra_result(path, load_configs=False, load_logs=False):
    """Loads the result of the SVA hydra run. The provided path should be
    to that of the experiment json file. Everything else can be inferred
    from that."""

    d = path.parent

    result = {"campaign": load(path)}

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


@cache
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
    if "PI" in keys:
        new_keys.append("PI")
    return new_keys


# def _compute_all_metrics(campaign):
#     """A helper function for computing the metrics given a campaign."""
#
#     # We want N_start to be the actual experiment not the priming step
#     metadata = campaign.data.metadata
#     STARTS = ["random", "LatinHypercube"]
#     N_start = sum([xx["policy"] in STARTS for xx in metadata])
#     N_max = len(metadata)
#
#     # Get some information about the experiment
#     experiment = campaign.experiment
#     experiment_maxima = experiment.metadata["optimum"][1]
#
#     # Relative opportunity cost
#     model_maxima_x = np.array(
#         [
#             md["model_optimum"][0].numpy()
#             for md in campaign.data.metadata[N_start:]
#         ]
#     ).squeeze()
#     with warnings.catch_warnings(record=True) as w:
#         experiment_at_model_maxima_x = experiment(model_maxima_x)
#     relative_opportunity_cost = np.abs(
#         experiment_maxima - experiment_at_model_maxima_x
#     ) / np.abs(experiment_maxima)
#
#     # Values of the points themselves
#     sampled_y_values = campaign.data.Y.squeeze()
#     relative_sampled_y_values_cost = np.abs(
#         experiment_maxima - sampled_y_values
#     ) / np.abs(experiment_maxima)
#     relative_sampled_y_values_cost = relative_sampled_y_values_cost.numpy()
#     relative_sampled_y_values_cost = [
#         np.min(relative_sampled_y_values_cost[:ii])
#         for ii in range(N_start, N_max)
#     ]
#
#     metrics = {
#         "relative_opportunity_cost": relative_opportunity_cost.numpy().tolist(),
#         "relative_sampled_y_values_cost": relative_sampled_y_values_cost,
#     }
#
#     return metrics
#
#
# def compute_metrics(data):
#     """Computes the metrics on provided data. The data must be a dict with
#     keys corresponding to acquisition functions and values as lists of
#     campaigns (essentially, what is read by read_data above)."""
#
#     results = defaultdict(dict)
#
#     for acqf, list_of_campaigns in tqdm(data.items()):
#         m = [_compute_all_metrics(c) for c in list_of_campaigns]
#
#         for metric in m[0].keys():
#             results[acqf][metric] = np.array([xx[metric] for xx in m]).squeeze()
#
#     return results
