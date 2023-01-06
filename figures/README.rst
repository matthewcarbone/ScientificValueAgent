Figures
=======

Unpack ``results.tar.bz2`` into this directory in order to use the plotting notebooks.

Note, the notebooks here generally output individual pieces of the figures in the manuscript in ``svg`` format. Significant post-processing was performed in an ``svg`` graphics editor (in the case of this work, Inkscape).


Results
-------

The ``results`` directory, when unpacked, should contain the following directories:

- ``results_22-12-21_sine2phase*``: results of the sine 2-phase, 2-dimensional experiment. These use a sigmoid sharpness parameter of ``a=30``, which was later updated to ``a=100`` in future calculations.
- ``results_22-12-21_xrd1dim``: results of the multi-phase, 1-dimensional experiment using the LGBFS optimizer
- ``results_22-12-21_xrd1dim_Adam``: results of the multi-phase, 1-dimensional experiment using the Adam optimizer with an initial learning rate of 0.05 and 200 training iterations
- ``results_23-01-06_sine2phase``: results of the sine 2-phase, 2-dimensional experiment using ``a=100``. This is a much sharper transition than ``a=30``.
