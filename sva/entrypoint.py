import hydra
from hydra.utils import instantiate

from sva import __version__
from sva.campaign import Campaign
from sva.logger import configure_loggers2, log_warnings


@log_warnings
def run(config):
    """Runs the campaign provided the experiment and policy."""

    experiment = instantiate(config.experiment, _convert_="partial")
    policy = instantiate(config.policy, _convert_="partial")
    campaign = Campaign(
        seed=config.seed,
        experiment=experiment,
        policy=policy,
        save_dir=config.paths.output_dir,
    )
    campaign.run()


@hydra.main(version_base="1.3", config_path="configs", config_name="core.yaml")
def hydra_main(config):
    """Executes training powered by Hydra, given the configuration file. Note
    that Hydra handles setting up the config.

    Parameters
    ----------
    config : omegaconf.DictConfig
    """

    configure_loggers2(config.logging, log_directory=config.paths.output_dir)
    run(config)


def entrypoint():
    hydra_main()
