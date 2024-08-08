import hydra
from hydra.utils import instantiate

from sva.campaign import Campaign, CampaignData
from sva.logger import log_warnings, logger_setup


@log_warnings
def run(config):
    """Runs the campaign provided the experiment and policy."""

    logger_setup(config.logging, config.paths.output_dir)
    experiment = instantiate(config.experiment, _convert_="partial")
    data = CampaignData()
    data.prime(
        experiment,
    )
    policy = instantiate(config.policy, _convert_="partial")
    campaign = Campaign(
        experiment=experiment,
        policy=policy,
        seed=config.seed,
    )
    campaign.run()
    campaign.save(file_path=config.path.output_dir)


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
