import hydra


@hydra.main(version_base="1.3", config_path="configs", config_name="core.yaml")
def hydra_main(config):
    """Executes training powered by Hydra, given the configuration file. Note
    that Hydra handles setting up the config.

    Parameters
    ----------
    config : omegaconf.DictConfig

    Returns
    -------
    float
        Validation metrics on the best checkpoint.
    """

    print(config)


def entrypoint():
    hydra_main()
