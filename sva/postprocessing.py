from collections import defaultdict
from pathlib import Path

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

    for p in paths:
        r = load_hydra_result(p, load_configs=load_configs, load_logs=load_logs)
        experiment_name = r["campaign"].experiment.name
        policy_name = r["campaign"].policy.name
        if not load_configs and not load_logs:
            r = r["campaign"]
        results[experiment_name][policy_name].append(r)

    return results
