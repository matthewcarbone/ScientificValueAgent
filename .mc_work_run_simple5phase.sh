#!/bin/bash

export CUDA_VISIBLE_DEVICES=

tsp uv run --python 3.12 \
sva_run -m hydra/launcher=joblib \
"seed=range(100, 120)" \
"hydra.launcher.verbose=10" \
"hydra.launcher.n_jobs=8" \
"experiment=Simple5Phase" \
"name=Simple5Phase_densityTrue" \
"policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
"policy.svf.params.square_exponent=false" \
"policy.svf.params.density=true" \
"policy.n_max=350" \
"paths.root_dir=mc_work/SVA2"

tsp uv run --python 3.12 \
sva_run -m hydra/launcher=joblib \
"seed=range(100, 120)" \
"hydra.launcher.verbose=10" \
"hydra.launcher.n_jobs=8" \
"experiment=Simple5Phase" \
"name=Simple5Phase_control" \
"policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
"policy.svf.params.square_exponent=false" \
"policy.svf.params.density=false" \
"policy.n_max=350" \
"paths.root_dir=mc_work/SVA2"
