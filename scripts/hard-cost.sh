#! /usr/bin/env bash
# Run the basic configurations for hard half-cheetah
# Should be run from the repo root
#
# ./scripts/hard-cost.sh

python mpc_bootstrap/main.py mpc --onpol_iters 15 --exp_name mpc  --seed 1 --verbose --env_name hc-hard
python mpc_bootstrap/main.py zero_bootstrap --onpol_iters 15 --exp_name zero --seed 1 --verbose --env_name hc-hard
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name delta --seed 1 --verbose --explore_std 0 --env_name hc-hard
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name gaussian --seed 1 --verbose --env_name hc-hard
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name no-explore-gaussian --seed 1 --verbose --no_extra_explore --env_name hc-hard

python mpc_bootstrap/plot.py \
    "data/mpc_hc-hard:AverageReturn:MPC" \
    "data/zero_hc-hard:AverageReturn:Zero BMPC" \
    'data/delta_hc-hard:AverageReturn:$\delta$-BMPC' \
    "data/gaussian_hc-hard:AverageReturn:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-hard:AverageReturn:No-explore BMPC" \
    --outfile report/hard-AverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc_hc-hard:AverageReturn:MPC" \
    "data/zero_hc-hard:LearnerAverageReturn:Zero BMPC" \
    'data/delta_hc-hard:LearnerAverageReturn:$\delta$-BMPC' \
    "data/gaussian_hc-hard:LearnerAverageReturn:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-hard:LearnerAverageReturn:No-explore BMPC" \
    --outfile report/hard-LearnerAverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc_hc-hard:DynamicsMSE:MPC" \
    "data/zero_hc-hard:DynamicsMSE:Zero BMPC" \
    'data/delta_hc-hard:DynamicsMSE:$\delta$-BMPC' \
    "data/gaussian_hc-hard:DynamicsMSE:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-hard:DynamicsMSE:No-explore BMPC" \
    --outfile report/hard-DynamicsMSE.pdf --yaxis "dynamics MSE" \
    --drop_iterations 5
python mpc_bootstrap/plot.py \
    "data/mpc_hc-hard:StandardizedRewardMSE:MPC" \
    "data/zero_hc-hard:StandardizedRewardMSE:Zero BMPC" \
    'data/delta_hc-hard:StandardizedRewardMSE:$\delta$-BMPC' \
    "data/gaussian_hc-hard:StandardizedRewardMSE:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-hard:StandardizedRewardMSE:No-explore BMPC" \
    --outfile report/hard-StandardizedRewardMSE.pdf --yaxis "reward MSE" \
    --drop_iterations 5
python mpc_bootstrap/plot.py \
    "data/mpc_hc-hard:StandardizedRewardBias:MPC" \
    "data/zero_hc-hard:StandardizedRewardBias:Zero BMPC" \
    'data/delta_hc-hard:StandardizedRewardBias:$\delta$-BMPC' \
    "data/gaussian_hc-hard:StandardizedRewardBias:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-hard:StandardizedRewardBias:No-explore BMPC" \
    --outfile report/hard-StandardizedRewardBias.pdf --yaxis "reward bias" \
    --drop_iterations 5
