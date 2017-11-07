#! /usr/bin/env bash
# Run the basic configurations for easy half-cheetah
# Should be run from the repo root
# Saves runs in ./data and the following images in ./report:
#
# ./scripts/hard-cost.sh

python mpc_bootstrap/main.py mpc --onpol_iters 20 --exp_name mpc-hard  --seed 1 5 10 --verbose --env_name hc-hard
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 20 --exp_name deterministic-hard --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --env_name hc-hard
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 20 --exp_name stochastic-hard --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-hard
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 20 --exp_name pure-hard --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-hard

python mpc_bootstrap/plot.py \
    "data/mpc-hard_hc-hard:AverageReturn:MPC" \
    'data/deterministic-hard_hc-hard:AverageReturn:$\delta$-BMPC' \
    "data/stochastic-hard_hc-hard:AverageReturn:Gaussian BMPC" \
    "data/pure-hard_hc-hard:AverageReturn:No-explore BMPC" \
    --outfile report/hard-AverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc-hard_hc-hard:AverageReturn:MPC" \
    'data/deterministic-hard_hc-hard:LearnerAverageReturn:$\delta$-BMPC' \
    "data/stochastic-hard_hc-hard:LearnerAverageReturn:Gaussian BMPC" \
    "data/pure-hard_hc-hard:LearnerAverageReturn:No-explore BMPC" \
    --outfile report/hard-LearnerAverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc-hard_hc-hard:DynamicsMSE:MPC" \
    'data/deterministic-hard_hc-hard:DynamicsMSE:$\delta$-BMPC' \
    "data/stochastic-hard_hc-hard:DynamicsMSE:Gaussian BMPC" \
    "data/pure-hard_hc-hard:DynamicsMSE:No-explore BMPC" \
    --outfile report/hard-DynamicsMSE.pdf --yaxis "dynamics MSE" \
    --drop_iterations 5
python mpc_bootstrap/plot.py \
    "data/mpc-hard_hc-hard:RewardMSE:MPC" \
    'data/deterministic-hard_hc-hard:RewardMSE:$\delta$-BMPC' \
    "data/stochastic-hard_hc-hard:RewardMSE:Gaussian BMPC" \
    "data/pure-hard_hc-hard:RewardMSE:No-explore BMPC" \
    --outfile report/hard-RewardMSE.pdf --yaxis "reward MSE" \
    --drop_iterations 5
python mpc_bootstrap/plot.py \
    "data/mpc-hard_hc-hard:AverageRewardBias:MPC" \
    'data/deterministic-hard_hc-hard:AverageRewardBias:$\delta$-BMPC' \
    "data/stochastic-hard_hc-hard:AverageRewardBias:Gaussian BMPC" \
    "data/pure-hard_hc-hard:AverageRewardBias:No-explore BMPC" \
    --outfile report/hard-AverageRewardBias.pdf --yaxis "reward MSE" \
    --drop_iterations 5
