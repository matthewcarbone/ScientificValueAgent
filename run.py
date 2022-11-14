from pathlib import Path
from shutil import copy2

import yaml

from value_agent import experiments
from value_agent.phases import sine_on_2d_raster_observations, truth_4phase


def get_random_experiments(params):

    jobs = params["jobs"].get("random_initial_seeds")
    if jobs is None:
        return

    experiment = params["experiment"]

    if experiment == "sine2phase":
        truth = sine_on_2d_raster_observations
    elif experiment == "xrd4phase":
        truth = truth_4phase
    else:
        raise ValueError(f"Invalid experiment: {experiment}")

    coordinates_fixed_kwargs = dict(
        truth=truth,
        xmin=0.0, xmax=1.0, n=3, ndim=2, n_raster=params["n_raster"],
        seed="set me"
    )

    # Still need to set `aqf`, `aqf_kwargs` and `name`
    experiment_fixed_kwargs = dict(
        points_per_dimension_full_grid=params["n_full_grid"],
        experiment_seed="set me",
        name=None,
        root=params["root"]
    )

    coordinate_seeds = list(range(*params["metadata"]["coords_seed_matrix"]))
    experiment_seeds = list(range(*params["metadata"]["exp_seed_matrix"]))

    list_of_experiments = []

    for job in jobs:
        aqf = job["aqf"]
        aqf_kwargs = job["aqf_kwargs"]
        if aqf_kwargs is None:
            aqf_kwargs = dict()
        beta = aqf_kwargs.get("beta")
        if beta is None:
            aqf_name = aqf
        else:
            beta = int(beta)
            aqf_name = f"{aqf}{beta}"

        for cseed in coordinate_seeds:
            for eseed in experiment_seeds:

                name = f"{experiment}-random-{aqf_name}-seed-{cseed}-{eseed}"

                experiment_kwargs = experiment_fixed_kwargs.copy()
                experiment_kwargs["aqf"] = aqf
                experiment_kwargs["aqf_kwargs"] = aqf_kwargs
                experiment_kwargs["experiment_seed"] = eseed
                experiment_kwargs["name"] = name

                coordinates_kwargs = coordinates_fixed_kwargs.copy()
                coordinates_kwargs["seed"] = cseed
                coordinates_kwargs["sd"] = params["sd"]
                d0 = experiments.Data.from_random(**coordinates_kwargs)

                exp = experiments.Experiment(data=d0, **experiment_kwargs)
                list_of_experiments.append(exp)

    names = [xx.name for xx in list_of_experiments]
    assert len(names) == len(set(names))  # Assert all names unique
    return list_of_experiments


def get_grid_experiments(params):

    jobs = params["jobs"].get("initial_grid")
    if jobs is None:
        return

    experiment = params["experiment"]

    if experiment == "sine2phase":
        truth = sine_on_2d_raster_observations
    elif experiment == "xrd4phase":
        truth = truth_4phase
    else:
        raise ValueError(f"Invalid experiment: {experiment}")

    coordinates_fixed_kwargs = dict(
        truth=truth,
        xmin=0.0, xmax=1.0, points_per_dimension=3, ndim=2,
        n_raster=params["n_raster"],
    )

    experiment_fixed_kwargs = dict(    
        points_per_dimension_full_grid=params["n_full_grid"],
        experiment_seed="set me",
        name=None,
        root=params["root"]
    )

    experiment_seeds = list(range(*params["metadata"]["exp_seed_matrix"]))

    list_of_experiments = []

    for job in jobs:
        aqf = job["aqf"]
        aqf_kwargs = job["aqf_kwargs"]
        if aqf_kwargs is None:
            aqf_kwargs = dict()
        beta = aqf_kwargs.get("beta")
        if beta is None:
            aqf_name = aqf
        else:
            beta = int(beta)
            aqf_name = f"{aqf}{beta}"

        for eseed in experiment_seeds:

            name = f"{experiment}-grid-{aqf_name}-seed-x-{eseed}"

            experiment_kwargs = experiment_fixed_kwargs.copy()
            experiment_kwargs["aqf"] = aqf
            experiment_kwargs["aqf_kwargs"] = aqf_kwargs
            experiment_kwargs["experiment_seed"] = eseed
            experiment_kwargs["name"] = name

            coordinates_kwargs = coordinates_fixed_kwargs.copy()
            coordinates_kwargs["sd"] = params["sd"]
            d0 = experiments.Data.from_grid(**coordinates_kwargs)

            exp = experiments.Experiment(data=d0, **experiment_kwargs)
            list_of_experiments.append(exp)

    names = [xx.name for xx in list_of_experiments]
    assert len(names) == len(set(names))  # Assert all names unique
    return list_of_experiments


if __name__ == '__main__':
    params = yaml.safe_load(open("jobs.yaml", "r"))
    copy2("jobs.yaml", Path(params["root"]) / "jobs.yaml")
    n_jobs = params["n_jobs"]
    exps = get_grid_experiments(params)
    exps.extend(get_random_experiments(params))
    print("Running", len(exps), "experiments")
    experiments.run_experiments(exps, n_jobs=n_jobs)
    
