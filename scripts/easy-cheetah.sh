#! /usr/bin/env bash
# Runs HalfCheetah environment experiments, using the supervised (easy) cost function
# This script should be run from the repo root as follows:
#
# ./scripts/easy-cheetah.sh
#
# This generates the following images in the report/ folder:
#   hard-cheetah-return.pdf - returns for 1-3
#   hard-cheetah-learner-return.pdf - for 2-4
#   hard-cheetah-reward-bias.pdf - for 1-3
#   hard-cheetah-reward-mse.pdf - for 1-3
#   hard-cheetah-dynamics-mse.pdf - for 1-3
#
# (1) Uniform-sampling MPC (2) delta-BMPC (3) 0-BMPC (4) random


python mpc_bootstrap/main_bootstrapped_mpc.py delta --verbose --exp_name delta --env_name hc-easy --onpol_iters 20 --seed 1 5 10
python mpc_bootstrap/main_bootstrapped_mpc.py zero --verbose --exp_name zero --env_name hc-easy --onpol_iters 20 --seed 1 5 10
python mpc_bootstrap/main_mpc.py --verbose --exp_name mpc --env_name hc-easy --onpol_iters 20 --seed 1 5 10
python mpc_bootstrap/main_random_policy.py --verbose --exp_name random --env_name hc-easy --num_paths 20 --seed 1 5 10

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:avg-return:$\delta$-BMPC' \
       'data/mpc_hc-easy:avg-return:Uniform MPC' \
       'data/zero_hc-easy:avg-return:$0$-BMPC (learner)' \
       --yaxis "10-episode avg return" --outfile report/easy-cheetah-return.pdf

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:avg-learner-return:$\delta$-BMPC' \
       'data/zero_hc-easy:avg-learner-return:$0$-BMPC (learner)' \
       --hlines 'data/random_hc-easy:avg-return:random' \
       --yaxis "10-episode avg return" --outfile report/easy-cheetah-learner-return.pdf

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:standardized-reward-bias:$\delta$-BMPC' \
       'data/mpc_hc-easy:standardized-reward-bias:Uniform MPC' \
       'data/zero_hc-easy:standardized-reward-bias:$0$-BMPC (learner)' \
       --yaxis "10-episode standardized reward bias" --outfile report/easy-cheetah-reward-bias.pdf

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:standardized-reward-mse:$\delta$-BMPC' \
       'data/mpc_hc-easy:standardized-reward-mse:Uniform MPC' \
       'data/zero_hc-easy:standardized-reward-mse:$0$-BMPC (learner)' \
       --yaxis "10-episode standardized reward MSE" --outfile report/easy-cheetah-reward-mse.pdf

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:dynamics-mse:$\delta$-BMPC' \
       'data/mpc_hc-easy:dynamics-mse:Uniform MPC' \
       'data/zero_hc-easy:dynamics-mse:$0$-BMPC (learner)' \
       --yaxis "10-episode dynamics MSE" --outfile report/easy-cheetah-dynamics-mse.pdf
