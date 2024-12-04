#!/bin/bash

export CUDA_VISIBLE_DEVICES=

EXPERIMENT="Sine2Phase"
PROJECT_ROOT_DIR="mc_work/SVA2"
N_MAX="250"

mkdir -p "$PROJECT_ROOT_DIR"

# Run the control experiments
ts uv run --python 3.12 \
    sva_run -m hydra/launcher=joblib \
    "seed=range(100, 200)" \
    "hydra.launcher.verbose=10" \
    "hydra.launcher.n_jobs=8" \
    "experiment=$EXPERIMENT" \
    "name=Sine2Phase" \
    "policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
    "policy.svf.density=false" \
    "policy.svf.denominator_pbc=false" \
    "policy.n_max=$N_MAX" \
    "paths.root_dir=$PROJECT_ROOT_DIR"

# Run the density=True experiments
ts uv run --python 3.12 \
    sva_run -m hydra/launcher=joblib \
    "seed=range(100, 200)" \
    "hydra.launcher.verbose=10" \
    "hydra.launcher.n_jobs=8" \
    "experiment=$EXPERIMENT" \
    "name=Sine2Phase_densityTrue" \
    "policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
    "policy.svf.density=true" \
    "policy.svf.denominator_pbc=false" \
    "policy.n_max=$N_MAX" \
    "paths.root_dir=$PROJECT_ROOT_DIR"

# Run the density=True experiments, plus pbc
ts uv run --python 3.12 \
    sva_run -m hydra/launcher=joblib \
    "seed=range(100, 200)" \
    "hydra.launcher.verbose=10" \
    "hydra.launcher.n_jobs=8" \
    "experiment=$EXPERIMENT" \
    "name=Sine2Phase_densityTrue_pbc" \
    "policy.acquisition_function=MaxVar,EI,PI,UCB-20,UCB-10,UCB-1" \
    "policy.svf.density=true" \
    "policy.svf.denominator_pbc=true" \
    "policy.n_max=$N_MAX" \
    "paths.root_dir=$PROJECT_ROOT_DIR"
