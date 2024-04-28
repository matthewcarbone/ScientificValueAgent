from hydra.utils import instantiate


def convert_to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def instantiate_experiments(config):
    return convert_to_list(instantiate(config.experiment))


def instantiate_campaign_parameters(config):
    return convert_to_list(
        instantiate(config.campaign_parameters, _convert_="partial")
    )
