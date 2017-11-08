#! /usr/bin/env bash
# Run the basic configurations for easy half-cheetah
# Should be run from the repo root
# Saves runs in ./data and the following images in ./report:
#
# ./scripts/easy-cost

python mpc_bootstrap/main.py mpc --onpol_iters 15 --exp_name mpc-easy  --seed 1 5 10 --verbose --env_name hc-easy
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name deterministic-easy --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --env_name hc-easy
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name stochastic-easy --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-easy
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name pure-easy --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-easy
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name deterministic-easy-stale1 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --env_name hc-easy --con_stale_data 1
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name stochastic-easy-stale1 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-easy --con_stale_data 1
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name pure-easy-stale1 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-easy --con_stale_data 1
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name deterministic-easy-stale3 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --env_name hc-easy --con_stale_data 3
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name stochastic-easy-stale3 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-easy --con_stale_data 3
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name pure-easy-stale3 --seed 1 5 10 --verbose --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-easy --con_stale_data 3

for i in "" -stale1 -stale3 ; do
    python mpc_bootstrap/plot.py \
        "data/mpc-easy_hc-easy:AverageReturn:MPC" \
        'data/deterministic-easy${i}_hc-easy:AverageReturn:$\delta$-BMPC' \
        "data/stochastic-easy${i}_hc-easy:AverageReturn:Gaussian BMPC" \
        "data/pure-easy${i}_hc-easy:AverageReturn:No-explore BMPC" \
        --outfile report/easy${i}-AverageReturn.pdf --yaxis "average rollout return"
    python mpc_bootstrap/plot.py \
        "data/mpc-easy_hc-easy:AverageReturn:MPC" \
        'data/deterministic-easy${i}_hc-easy:LearnerAverageReturn:$\delta$-BMPC' \
        "data/stochastic-easy${i}_hc-easy:LearnerAverageReturn:Gaussian BMPC" \
        "data/pure-easy${i}_hc-easy:LearnerAverageReturn:No-explore BMPC" \
        --outfile report/easy${i}-LearnerAverageReturn.pdf --yaxis "average rollout return"
    python mpc_bootstrap/plot.py \
        "data/mpc-easy_hc-easy:DynamicsMSE:MPC" \
        'data/deterministic-easy${i}_hc-easy:DynamicsMSE:$\delta$-BMPC' \
        "data/stochastic-easy${i}_hc-easy:DynamicsMSE:Gaussian BMPC" \
        "data/pure-easy${i}_hc-easy:DynamicsMSE:No-explore BMPC" \
        --outfile report/easy${i}-DynamicsMSE.pdf --yaxis "dynamics MSE" \
        --drop_iterations 5
    python mpc_bootstrap/plot.py \
        "data/mpc-easy_hc-easy:RewardMSE:MPC" \
        "data/stochastic-easy${i}_hc-easy:RewardMSE:Gaussian BMPC" \
        "data/pure-easy${i}_hc-easy:RewardMSE:No-explore BMPC" \
        --outfile report/easy${i}-RewardMSE.pdf --yaxis "reward MSE"
    #     'data/deterministic-easy${i}_hc-easy:RewardMSE:$\delta$-BMPC' \
    python mpc_bootstrap/plot.py \
        "data/mpc-easy_hc-easy:AverageRewardBias:MPC" \
        "data/stochastic-easy${i}_hc-easy:AverageRewardBias:Gaussian BMPC" \
        "data/pure-easy${i}_hc-easy:AverageRewardBias:No-explore BMPC" \
        --outfile report/easy${i}-AverageRewardBias.pdf --yaxis "reward MSE"
    #    'data/deterministic-easy_hc-easy:AverageRewardBias:$\delta$-BMPC' \
done
