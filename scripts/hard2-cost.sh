#! /usr/bin/env bash
# Run the basic configurations for easy half-cheetah
# Should be run from the repo root
# Saves runs in ./data and the following images in ./report:
#
# ./scripts/easy-cost

python mpc_bootstrap/main.py --onpol_iters 20 --agent mpc --exp_name mpc-hard2  --seed 1 5 10 --time --env_name hc-hard --mpc_horizon 50
python mpc_bootstrap/main.py --onpol_iters 20 --agent bootstrap --exp_name deterministic-hard2 --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --explore_std 0 --deterministic_learner --env_name hc-hard --mpc_horizon 50
python mpc_bootstrap/main.py --onpol_iters 20 --agent bootstrap --exp_name stochastic-hard2 --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --env_name hc-hard --mpc_horizon 50
python mpc_bootstrap/main.py --onpol_iters 20 --agent bootstrap --exp_name pure-hard2 --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --no_extra_explore --env_name hc-hard --mpc_horizon 50

python mpc_bootstrap/plot.py \
    "data/mpc-hard2_hc-hard:AverageReturn:MPC" \
    'data/deterministic-hard2_hc-hard:AverageReturn:$\delta$-BMPC' \
    "data/stochastic-hard2_hc-hard:AverageReturn:Gaussian BMPC" \
    "data/pure-hard2_hc-hard:AverageReturn:No-explore BMPC" \
    --outfile report/hard2-AverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc-hard2_hc-hard:AverageReturn:MPC" \
    'data/deterministic-hard2_hc-hard:LearnerAverageReturn:$\delta$-BMPC' \
    "data/stochastic-hard2_hc-hard:LearnerAverageReturn:Gaussian BMPC" \
    "data/pure-hard2_hc-hard:LearnerAverageReturn:No-explore BMPC" \
    --outfile report/hard2-LearnerAverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc-hard2_hc-hard:DynamicsMSE:MPC" \
    'data/deterministic-hard2_hc-hard:DynamicsMSE:$\delta$-BMPC' \
    "data/stochastic-hard2_hc-hard:DynamicsMSE:Gaussian BMPC" \
    "data/pure-hard2_hc-hard:DynamicsMSE:No-explore BMPC" \
    --outfile report/hard2-DynamicsMSE.pdf --yaxis "dynamics MSE" \
    --drop_iterations 5
