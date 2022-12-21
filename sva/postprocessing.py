from functools import cache
import json
from pathlib import Path

from sva.experiments import Experiment


@cache
def read_data(path):
    results = dict()
    for file in Path(path).rglob("*.json"):
        with open(file, "r") as f:
            d = json.loads(json.load(f))
        c = Experiment.from_dict(d)
        results[str(c.name)] = c
    return results


def parse_results_by_acquisition_function(results):
    """Takes the results dictionary returned by ``read_data`` and organizes
    it by acquisition function. Keys are assumed to be of the form:

    ``"value_sig-truth_sig-acqf_name-seed-seed.json"``

    Parameters
    ----------
    results : dict

    Returns
    -------
    dict
        A dictionary with keys as the acquisition function names.
    """

    keys = list(results.keys())
    keys_split = [key.split("-") for key in keys]
    acquisition_function_names = list(set([kk[2] for kk in keys_split]))

    for_return = {acqf_name: [] for acqf_name in acquisition_function_names}
    for key, value in results.items():
        acqf_name = key.split("-")[2]
        for_return[acqf_name].append(value)

    return for_return
