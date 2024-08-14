from pathlib import Path

import hydra
from hydra.utils import instantiate

from sva.campaign import Campaign, CampaignData
from sva.logger import log_warnings, logger_setup


@log_warnings
def run(config):
    """Runs the campaign provided the experiment and policy."""

    logger_setup(config.logging, config.paths.output_dir)

    # Setup the experiment
    experiment = instantiate(config.experiment, _convert_="partial")

    # Initialize the data object
    data = CampaignData()
    data_p = config.data.prime_kwargs
    data.prime(
        experiment, protocol=data_p.protocol, seed=config.seed, **data_p.kwargs
    )

    # Initialize the policy
    policy = instantiate(config.policy, _convert_="partial")

    # Initialize the campaign
    campaign = Campaign(
        data=data,
        experiment=experiment,
        policy=policy,
        seed=config.seed,
    )

    # Run the campaign
    campaign.run()

    # Save the campaign
    save_path = Path(config.paths.output_dir) / f"{campaign.name}.json"
    campaign.save(
        save_path,
        json_kwargs={"indent": 4, "sort_keys": True},
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="core.yaml")
def hydra_main(config):
    """Executes training powered by Hydra, given the configuration file. Note
    that Hydra handles setting up the config.

    Parameters
    ----------
    config : omegaconf.DictConfig
    """

    run(config)


def entrypoint():
    hydra_main()
