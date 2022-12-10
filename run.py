import numpy as np
import yaml

from value_agent import experiments
from value_agent.utils import get_function_from_signature


def get_experiments(params):

    how = params["how"]

    value_signature = params["value_function_signature"]
    truth_signature = params["truth_function_signature"]

    value_function = get_function_from_signature(value_signature)
    truth_function = get_function_from_signature(truth_signature)

    coordinates_fixed_kwargs = dict(
        truth=truth_function,
        value=value_function,
        seed="set me",
        how=how,
        truth_kwargs=params.get("truth_function_kwargs", dict()),
        value_kwargs=params.get("value_function_kwargs", dict()),
        xmin=0.0,
        xmax=1.0,
        points_per_dimension=3,
        ndim=2,
        n_raster=params.get("n_raster", None),
    )

    # Still need to set `aqf`, `aqf_kwargs` and `name`
    experiment_fixed_kwargs = dict(
        points_per_dimension_full_grid=params["n_full_grid"],
        experiment_seed="set me",
        name=None,
    )

    # If how == "random", we sample over N x N different random combinations
    # of coordinate seeds and model/experiment seeds. If how == "grid", we just
    # sample over N x N total experiment seeds.
    np.random.seed(params["seed"])
    if how == "random":
        size = (params["total_jobs"], 2)
        seeds_arr = np.random.choice(range(int(1e6)), size=size, replace=False)
        coords_seeds = seeds_arr[:, 0].tolist()
        exp_seeds = seeds_arr[:, 1].tolist()
    else:
        size = params["total_jobs"]
        seeds_arr = np.random.choice(range(int(1e6)), size=size, replace=False)
        coords_seeds = [None for _ in range(size)]
        exp_seeds = seeds_arr.tolist()

    list_of_experiments = []

    for job in params["jobs"]:
        aqf = job["aqf"]
        aqf_kwargs = job.get("aqf_kwargs", dict())
        beta = aqf_kwargs.get("beta")
        if beta is None:
            aqf_name = aqf
        else:
            beta = int(beta)
            aqf_name = f"{aqf}{beta}"

        for (cseed, eseed) in zip(coords_seeds, exp_seeds):

            name = f"{aqf_name}-seed-{cseed}-{eseed}"

            experiment_kwargs = experiment_fixed_kwargs.copy()
            experiment_kwargs["aqf"] = aqf
            experiment_kwargs["aqf_kwargs"] = aqf_kwargs
            experiment_kwargs["experiment_seed"] = eseed
            experiment_kwargs["name"] = name

            coordinates_kwargs = coordinates_fixed_kwargs.copy()
            coordinates_kwargs["seed"] = cseed
            d0 = experiments.Data.from_initial_conditions(
                **coordinates_kwargs
            )

            exp = experiments.Experiment(data=d0, **experiment_kwargs)
            list_of_experiments.append(exp)

    names = [xx.name for xx in list_of_experiments]
    assert len(names) == len(set(names))  # Assert all names unique
    return list_of_experiments


if __name__ == '__main__':
    params = yaml.safe_load(open("jobs.yaml", "r"))
    exps = get_experiments(params)
    print("Running", len(exps), "experiments")
    experiments.run_experiments(exps, n_jobs=params["n_multiprocessing_jobs"])
    
