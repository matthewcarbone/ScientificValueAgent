from hydra import compose, initialize
from hydra.utils import instantiate


def get_default_model_factory():
    """Simple method to instantiate the default model factory."""

    f = "models"
    with initialize(version_base="1.3", config_path=f):
        cfg = compose(config_name="single_task_gp")
    return instantiate(cfg)
