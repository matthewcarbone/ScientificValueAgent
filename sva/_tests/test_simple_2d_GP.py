# Get the campaign data
from sva.campaign import CampaignData

# Import the various experiments we need for the notebook
from sva.experiments import Simple2d

# Import the helper functions for Gaussian Processes
from sva.models.gp import EasySingleTaskGP

# Import the seeding function for reproducibility
from sva.utils import seed_everything


def test_simple_2d_GP():
    seed_everything(123)

    experiment = Simple2d()
    data = CampaignData()
    data.prime(experiment, protocol="random", n=5)
    extent = experiment.get_domain_mpl_extent()
    X = data.X
    assert extent == [-4.0, 5.0, -5.0, 4.0]
    x = experiment.get_dense_coordinates(ppd=100)

    # test experiment forward
    experiment(x)
    y = experiment(X)

    gp = EasySingleTaskGP.from_default(X, y)
    gp.fit_mll()
    gp.predict(x)
