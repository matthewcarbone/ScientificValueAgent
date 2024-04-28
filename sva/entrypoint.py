from pathlib import Path

import hydra
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from sva import __version__
from sva.hydra_utils.instantiators import (
    instantiate_campaign_parameters,
    instantiate_experiments,
)
from sva.utils import Timer, seed_everything


def _run_job(job):
    with Timer() as timer:
        experiment = job["experiment"]
        config = job["config"]
        params = job["parameters"]
        seed = job["seed"]

        seed_everything(seed)

        experiment = experiment()

        rtp = config["procedure"]["experiment_run_parameters"]
        experiment.run(parameters=params, **rtp)

        experiment_name = experiment.name
        if config["use_full_hashes"]:
            parameters_name = f"{params.name}-{params.get_hash()}"
        else:
            parameters_name = params.name

        root = (
            Path(config["paths"]["output_dir"])
            / experiment_name
            / parameters_name
        )
        path = root / f"{seed}.json"
        experiment.save(path, json_kwargs={"indent": 4, "sort_keys": True})

    print(f"done with exp: {path} in {timer.dt:.02f} s")


def run_single_policy(experiments, parameters, config):
    """Runs a set of experiments and parameters under a single policy. In
    other words, the acquisition function used at the start of an experiment
    will not change throughout."""

    jobs = []
    config_resolved = OmegaConf.to_container(config, resolve=True)
    for experiment in experiments:
        for params in parameters:
            for unique_seed in range(config.replicas):
                jobs.append(
                    {
                        "experiment": experiment,
                        "parameters": params,
                        "config": config_resolved,
                        "seed": config.seed + unique_seed,
                    }
                )
    Parallel(n_jobs=config.n_jobs)(delayed(_run_job)(job) for job in jobs)


@hydra.main(version_base="1.3", config_path="configs", config_name="core.yaml")
def hydra_main(config):
    """Executes training powered by Hydra, given the configuration file. Note
    that Hydra handles setting up the config.

    Parameters
    ----------
    config : omegaconf.DictConfig
    """

    print(f"Running sva version {__version__}")
    if config.name is None:
        raise ValueError("name must be set")
    print(f"name set to: {config.name}")
    print(f"seed set to: {config.seed}")
    print()

    # print(config.paths.output_dir)
    # print(type(config.paths.output_dir))
    # return

    experiments = instantiate_experiments(config)

    campaign_parameters = instantiate_campaign_parameters(config)

    if config.procedure.name == "single_policy":
        run_single_policy(experiments, campaign_parameters, config)
    else:
        raise ValueError("Unknown procedure provided")


def entrypoint():
    hydra_main()
