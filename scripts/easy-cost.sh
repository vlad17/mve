#! /usr/bin/env bash
# Run the basic configurations for easy half-cheetah
# Should be run from the repo root
#
# ./scripts/easy-cost

python mpc_bootstrap/main.py mpc --onpol_iters 15 --exp_name mpc --seed 1 --verbose --env_name hc-easy
python mpc_bootstrap/main.py zero_bootstrap --onpol_iters 15 --exp_name zero --seed 1 --verbose --env_name hc-easy
python mpc_bootstrap/main.py delta_bootstrap --onpol_iters 15 --exp_name delta --seed 1 --verbose --env_name hc-easy
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name gaussian --seed 1 --verbose --env_name hc-easy
python mpc_bootstrap/main.py gaussian_bootstrap --onpol_iters 15 --exp_name no-explore-gaussian --seed 1 --verbose --no_extra_explore --env_name hc-easy

python mpc_bootstrap/plot.py \
    "data/mpc_hc-easy:AverageReturn:MPC" \
    "data/zero_hc-easy:AverageReturn:Zero BMPC" \
    'data/delta_hc-easy:AverageReturn:$\delta$-BMPC' \
    "data/gaussian_hc-easy:AverageReturn:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-easy:AverageReturn:No-explore Gaussian BMPC" \
    --outfile report/easy-AverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc_hc-easy:AverageReturn:MPC" \
    "data/zero_hc-easy:LearnerAverageReturn:Zero BMPC" \
    'data/delta_hc-easy:LearnerAverageReturn:$\delta$-BMPC' \
    "data/gaussian_hc-easy:LearnerAverageReturn:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-easy:LearnerAverageReturn:No-explore Gaussian BMPC" \
    --outfile report/easy-LearnerAverageReturn.pdf --yaxis "average rollout return"
python mpc_bootstrap/plot.py \
    "data/mpc_hc-easy:DynamicsMSE:MPC" \
    "data/zero_hc-easy:DynamicsMSE:Zero BMPC" \
    'data/delta_hc-easy:DynamicsMSE:$\delta$-BMPC' \
    "data/gaussian_hc-easy:DynamicsMSE:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-easy:DynamicsMSE:No-explore Gaussian BMPC" \
    --outfile report/easy-DynamicsMSE.pdf --yaxis "dynamics MSE" \
    --drop_iterations 5
python mpc_bootstrap/plot.py \
    "data/mpc_hc-easy:StandardizedRewardMSE:MPC" \
    "data/zero-easy:StandardizedRewardMSE:Gaussian Zero" \
    'data/delta_hc-easy:StandardizedRewardMSE:$\delta$-BMPC' \
    "data/gaussian-easy:StandardizedRewardMSE:Gaussian BMPC" \
    "no-explore-gaussian_hc-easy:StandardizedRewardMSE:No-explore Gaussian BMPC" \
    --outfile report/easy-StandardizedRewardMSE.pdf --yaxis "reward MSE"
python mpc_bootstrap/plot.py \
    "data/mpc_hc-easy:StandardizedRewardBias:MPC" \
    "data/zero_hc-easy:StandardizedRewardBias:Zero BMPC" \
    'data/delta_hc-easy:StandardizedRewardBias:$\delta$-BMPC' \
    "data/gaussian_hc-easy:StandardizedRewardBias:Gaussian BMPC" \
    "data/no-explore-gaussian_hc-easy:StandardizedRewardBias:No-explore Gaussian BMPC" \
    --outfile report/easy-StandardizedRewardBias.pdf --yaxis "reward MSE"

python mpc_bootstrap/tune.py  --onpol_iters 15 --exp_name mpc --seed 1 --verbose --env_name hc-easy
