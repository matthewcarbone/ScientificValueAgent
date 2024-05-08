from collections import defaultdict
from functools import cache
from pathlib import Path

from yaml import safe_load

from sva.monty.json import load_anything


def load_hydra_result(path, load_configs=False, load_logs=False):
    """Loads the result of the SVA hydra run. The provided path should be
    to that of the experiment json file. Everything else can be inferred
    from that."""

    d = path.parent
    _, _, seed = path.stem.split("_")

    result = {
        "experiment": load_anything(path),
        "seed": seed,
    }

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

    # Ensure everything is unique
    if len(set([xx.stem for xx in paths])) != len(paths):
        raise RuntimeError("Non-unique file names are ambiguous")

    for p in paths:
        experiment, policy, _ = p.stem.split("_")
        r = load_hydra_result(p, load_configs=load_configs, load_logs=load_logs)
        results[experiment][policy].append(r)

    return results
