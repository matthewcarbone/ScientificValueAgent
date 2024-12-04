#!/bin/bash

export CUDA_VISIBLE_DEVICES=

# Run the control experiments while squaring the exponent
tsp uv run --python 3.12 \
sva_run -m hydra/launcher=joblib \
"seed=range(100, 300)" \
"hydra.launcher.verbose=10" \
"hydra.launcher.n_jobs=8" \
"experiment=Sine2Phase" \
"name=Sine2Phase_sqTrue" \
"policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
"policy.svf.params.square_exponent=true" \
"policy.n_max=150" \
"paths.root_dir=mc_work/SVA2"

# Run the control experiments while squaring the exponent, and applying the density
tsp uv run --python 3.12 \
sva_run -m hydra/launcher=joblib \
"seed=range(100, 300)" \
"hydra.launcher.verbose=10" \
"hydra.launcher.n_jobs=8" \
"experiment=Sine2Phase" \
"name=Sine2Phase_sqTrue_densityTrue" \
"policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
"policy.svf.params.density=true" \
"policy.svf.params.square_exponent=true" \
"policy.n_max=150" \
"paths.root_dir=mc_work/SVA2"

# Run the control experiments
tsp uv run --python 3.12 \
sva_run -m hydra/launcher=joblib \
"seed=range(100, 300)" \
"hydra.launcher.verbose=10" \
"hydra.launcher.n_jobs=8" \
"experiment=Sine2Phase" \
"name=Sine2Phase" \
"policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
"policy.n_max=150" \
"paths.root_dir=mc_work/SVA2"

# Run the density=True experiments
tsp uv run --python 3.12 \
sva_run -m hydra/launcher=joblib \
"seed=range(100, 300)" \
"hydra.launcher.verbose=10" \
"hydra.launcher.n_jobs=8" \
"experiment=Sine2Phase" \
"name=Sine2Phase_densityTrue" \
"policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
"policy.svf.params.density=true" \
"policy.n_max=150" \
"paths.root_dir=mc_work/SVA2"

