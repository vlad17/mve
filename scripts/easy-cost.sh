#! /usr/bin/env bash
# Run the basic configurations for easy half-cheetah
# Should be run from the repo root
# Saves runs in ./data and the following images in ./report:
#
# ./scripts/easy-cost

python mpc_bootstrap/main.py mpc --onpol_iters 15 --exp_name mpc-easy  --seed 1 5 10 --time --env_name hc-easy
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name deterministic-easy --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --deterministic_learner --env_name hc-easy
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name stochastic-easy --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-easy
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name pure-easy --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-easy

python mpc_bootstrap/plot.py \
    "data/mpc-easy_hc-easy:AverageReturn:MPC" \
    'data/deterministic-easy_hc-easy:AverageReturn:$\delta$-BMPC' \
    "data/stochastic-easy_hc-easy:AverageReturn:Gaussian BMPC" \
    "data/pure-easy_hc-easy:AverageReturn:No-explore BMPC" \
    --outfile report/easy-AverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc-easy_hc-easy:AverageReturn:MPC" \
    'data/deterministic-easy_hc-easy:LearnerAverageReturn:$\delta$-BMPC' \
    "data/stochastic-easy_hc-easy:LearnerAverageReturn:Gaussian BMPC" \
    "data/pure-easy_hc-easy:LearnerAverageReturn:No-explore BMPC" \
    --outfile report/easy-LearnerAverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc-easy_hc-easy:DynamicsMSE:MPC" \
    'data/deterministic-easy_hc-easy:DynamicsMSE:$\delta$-BMPC' \
    "data/stochastic-easy_hc-easy:DynamicsMSE:Gaussian BMPC" \
    "data/pure-easy_hc-easy:DynamicsMSE:No-explore BMPC" \
    --outfile report/easy-DynamicsMSE.pdf --yaxis "dynamics MSE" \
    --drop_iterations 5
