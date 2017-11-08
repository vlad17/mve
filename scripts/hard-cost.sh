#! /usr/bin/env bash
# Run the basic configurations for hard half-cheetah
# Should be run from the repo root
# Saves runs in ./data and the following images in ./report:
#
# ./scripts/hard-cost.sh

python mpc_bootstrap/main.py mpc --onpol_iters 15 --exp_name mpc-hard  --seed 1 5 10 --verbose --env_name hc-hard
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name deterministic-hard --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --env_name hc-hard
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name stochastic-hard --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-hard
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name pure-hard --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-hard
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name deterministic-hard-stale1 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --env_name hc-hard --con_stale_data 1
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name stochastic-hard-stale1 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-hard --con_stale_data 1
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name pure-hard-stale1 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-hard --con_stale_data 1
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name deterministic-hard-stale3 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --env_name hc-hard --con_stale_data 3
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name stochastic-hard-stale3 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-hard --con_stale_data 3
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name pure-hard-stale3 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-hard --con_stale_data 3

for i in "" -stale1 -stale3 ; do
    python mpc_bootstrap/plot.py \
        "data/mpc-hard_hc-hard:AverageReturn:MPC" \
        'data/deterministic-hard${i}_hc-hard:AverageReturn:$\delta$-BMPC' \
        "data/stochastic-hard${i}_hc-hard:AverageReturn:Gaussian BMPC" \
        "data/pure-hard${i}_hc-hard:AverageReturn:No-explore BMPC" \
        --outfile report/hard${i}-AverageReturn.pdf --yaxis "average rollout return"
    python mpc_bootstrap/plot.py \
        "data/mpc-hard_hc-hard:AverageReturn:MPC" \
        'data/deterministic-hard${i}_hc-hard:LearnerAverageReturn:$\delta$-BMPC' \
        "data/stochastic-hard${i}_hc-hard:LearnerAverageReturn:Gaussian BMPC" \
        "data/pure-hard${i}_hc-hard:LearnerAverageReturn:No-explore BMPC" \
        --outfile report/hard${i}-LearnerAverageReturn.pdf --yaxis "average rollout return"
    python mpc_bootstrap/plot.py \
        "data/mpc-hard_hc-hard:DynamicsMSE:MPC" \
        'data/deterministic-hard${i}_hc-hard:DynamicsMSE:$\delta$-BMPC' \
        "data/stochastic-hard${i}_hc-hard:DynamicsMSE:Gaussian BMPC" \
        "data/pure-hard${i}_hc-hard:DynamicsMSE:No-explore BMPC" \
        --outfile report/hard${i}-DynamicsMSE.pdf --yaxis "dynamics MSE" \
        --drop_iterations 5
    python mpc_bootstrap/plot.py \
        "data/mpc-hard_hc-hard:RewardMSE:MPC" \
        "data/stochastic-hard${i}_hc-hard:RewardMSE:Gaussian BMPC" \
        "data/pure-hard${i}_hc-hard:RewardMSE:No-explore BMPC" \
        --outfile report/hard${i}-RewardMSE.pdf --yaxis "reward MSE"
    #     'data/deterministic-hard${i}_hc-hard:RewardMSE:$\delta$-BMPC' \
    python mpc_bootstrap/plot.py \
        "data/mpc-hard_hc-hard:AverageRewardBias:MPC" \
        "data/stochastic-hard${i}_hc-hard:AverageRewardBias:Gaussian BMPC" \
        "data/pure-hard${i}_hc-hard:AverageRewardBias:No-explore BMPC" \
        --outfile report/hard${i}-AverageRewardBias.pdf --yaxis "reward MSE"
    #    'data/deterministic-hard_hc-hard:AverageRewardBias:$\delta$-BMPC' \
done
