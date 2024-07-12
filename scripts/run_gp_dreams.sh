#!/bin/bash

SEED="range(100,120)"
ACQUISITION_FUNCTIONS="UCB-4.0,UCB-36.0,UCB-100.0,MaxVar,EI"
MAX_EXPERIMENT_SEED=10
LENGTH_SCALES=('0.1' '0.2' '0.3' '0.4' '0.5')
DIMENSIONS=('1' '2')
KERNEL='matern'

export CUDA_VISIBLE_DEVICES=""

for experiment_length_scale in "${LENGTH_SCALES[@]}"; do
	for experiment_dimension in "${DIMENSIONS[@]}"; do
		for experiment_seed in $(seq 1 $MAX_EXPERIMENT_SEED); do

			name="dream_${KERNEL}_${experiment_length_scale}_${experiment_dimension}_${experiment_seed}"

			tsp sva_run \
				-m hydra/launcher=joblib \
				seed="${SEED}" \
				experiment=dream_gp \
				experiment.kernel="${KERNEL}" \
				experiment.d="${experiment_dimension}" \
				experiment.lengthscale="${experiment_length_scale}" \
				experiment.seed="${experiment_seed}" \
				policy.model_factory.covar_module.kernel="${KERNEL}" \
				policy.acquisition_function="${ACQUISITION_FUNCTIONS}" \
				name="${name}" \
				policy.n_max=100 \
				hydra.launcher.n_jobs=12 \
				hydra.launcher.verbose=10 \
				policy.save_model=false \
				policy.save_acquisition_function=false

		done
	done
done
