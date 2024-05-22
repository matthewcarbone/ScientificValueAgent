#!/bin/bash

N_MAX=225
NAME=GGCE
SEED=133

sva_run experiment=Peierls seed="$SEED" name="$NAME" policy=random policy.n_max="$N_MAX"
sva_run experiment=Peierls seed="$SEED" name="$NAME" policy=grid policy.n_max="$N_MAX"

sva_run -m hydra/launcher=joblib experiment=Peierls seed="$SEED" policy=sva_proximity_penalized name="$NAME" "policy.acquisition_function=EI,UCB-10.0,UCB-20.0,UCB-100.0,MaxVar" policy.n_max="$N_MAX" hydra.launcher.n_jobs=5 hydra.launcher.verbose=10

# sva_run -m hydra/launcher=joblib experiment=Peierls seed=133 policy=sva name=ggce_test "policy.acquisition_function=EI,UCB-10.0,UCB-20.0,UCB-100.0" policy.n_max="$N_MAX" hydra.launcher.n_jobs=4 hydra.launcher.verbose=10

# sva_run experiment=Peierls seed="$SEED" policy=sva_proximity_penalized name="$NAME" policy.acquisition_function=EI policy.n_max="$N_MAX"
