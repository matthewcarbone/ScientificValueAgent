# Import the various experiments we need for the notebook
from sva.experiments import Simple2d

# Import the helper functions for Gaussian Processes
from sva.models.gp import EasySingleTaskGP

# Import the seeding function for reproducibility
from sva.utils import seed_everything


def test_simple_2d_GP():
    seed_everything(123)

    experiment = Simple2d()
    experiment.initialize_data(n=35, protocol="random")
    extent = experiment.get_experimental_domain_mpl_extent()
    X = experiment.data.X
    assert extent == [-4.0, 5.0, -5.0, 4.0]
    x = experiment.get_dense_coordinates(ppd=100)
    _, _ = experiment(x)

    _y, _ = experiment(X)
    gp = EasySingleTaskGP.from_default(X, _y)
    gp.fit_mll()
    _, _ = gp.predict(x)
