#!/bin/bash

# This is a simple utility for running many GP Dream experiments

# GPU usage actually slows down this code because the experiments are so
# small
export CUDA_VISIBLE_DEVICES=""

# Set the parameters for the job
# One can obviously change things below too, but these are the most common
# things to change
# -----
# First, the parameters explicitly looped over here
EXPERIMENT_LENGTH_SCALES=("0.1" "0.2" "0.3" "0.4" "0.5")

# -----
# Then the parameters that are looped over within SVA via hydra/launcher
SEED_RANGE="range(1,6)"
EXPERIMENT_KERNEL="rbf"
DIMENSION="2"
EXPERIMENT_SEED_RANGE="range(1,21)"
MODEL_KERNEL="rbf"
ACQUISITION_FUNCTIONS="MaxVar,EI,UCB-100,UCB-20,UCB-10,UCB-1"
N_MAX="100"

# Jobs are run using task spooler, but can in principle also be submitted to
# a comput cluster
for exp_length_scale in "${EXPERIMENT_LENGTH_SCALES[@]}"; do
	tsp sva_run -m hydra/launcher=joblib \
		seed="$SEED_RANGE" \
		experiment=dream_gp \
		experiment.gp_model_params.kernel="$EXPERIMENT_KERNEL" \
		experiment.d="$DIMENSION" \
		experiment.gp_model_params.lengthscale="$exp_length_scale" \
		experiment.seed="$EXPERIMENT_SEED_RANGE" \
		policy.model_factory.covar_module.kernel="$MODEL_KERNEL" \
		policy.acquisition_function="$ACQUISITION_FUNCTIONS" \
		name="dream_${DIMENSION}d_${EXPERIMENT_KERNEL}_${exp_length_scale}" \
		policy.n_max="$N_MAX" \
		hydra.launcher.verbose=10 \
		hydra.launcher.n_jobs=16
done
