#! /usr/bin/env bash
# Runs HalfCheetah environment experiments, using the supervised (easy) cost function
# This script should be run from the repo root as follows:
#
# ./scripts/easy-cheetah.sh
#
# This generates the following images in the report/ folder:
#   easy-cheetah-return.pdf - returns for 1-3
#   easy-cheetah-learner-return.pdf - for 2-4
#   easy-cheetah-reward-bias.pdf - for 1-3
#   easy-cheetah-reward-mse.pdf - for 1-3
#   easy-cheetah-dynamics-mse.pdf - for 1-3
#
# (1) Uniform-sampling MPC (2) delta-BMPC (3) 0-BMPC (4) random
#
# Results from running this experiment can be found in report/easy-cheetah.tgz


python mpc_bootstrap/main_bootstrapped_mpc.py delta --verbose --exp_name delta --env_name hc-easy --onpol_iters 20 --seed 1 5 10 15
python mpc_bootstrap/main_bootstrapped_mpc.py zero --verbose --exp_name zero --env_name hc-easy --onpol_iters 20 --seed 1 5 10 15
python mpc_bootstrap/main_mpc.py --verbose --exp_name mpc --env_name hc-easy --onpol_iters 20 --seed 1 5 10 15
python mpc_bootstrap/main_random_policy.py --verbose --exp_name random --env_name hc-easy --num_paths 20 --seed 1 5 10 15

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:avg-return:$\delta$ RS MPC' \
       'data/mpc_hc-easy:avg-return:Uniform RS MPC' \
       'data/zero_hc-easy:avg-return:$0$ RS MPC' \
       --yaxis "10-episode avg return" --outfile report/easy-cheetah-return.pdf

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:avg-learner-return:$\delta$ RS MPC' \
       'data/zero_hc-easy:avg-learner-return:$0$ RS MPC' \
       --hlines 'data/random_hc-easy:avg-return:random' \
       --yaxis "10-episode avg return" --outfile report/easy-cheetah-learner-return.pdf


python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:standardized-reward-bias:$\delta$ RS MPC' \
       'data/mpc_hc-easy:standardized-reward-bias:Uniform RS MPC' \
       'data/zero_hc-easy:standardized-reward-bias:$0$ RS MPC' \
       --drop_iterations 7 \
       --yaxis "10-episode standardized reward bias" --outfile report/easy-cheetah-reward-bias.pdf

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:standardized-reward-mse:$\delta$ RS MPC' \
       'data/mpc_hc-easy:standardized-reward-mse:Uniform RS MPC' \
       'data/zero_hc-easy:standardized-reward-mse:$0$ RS MPC' \
       --drop_iterations 7 \
       --yaxis "10-episode standardized reward MSE" --outfile report/easy-cheetah-reward-mse.pdf

python mpc_bootstrap/plot.py \
       'data/delta_hc-easy:dynamics-mse:$\delta$ RS MPC' \
       'data/mpc_hc-easy:dynamics-mse:Uniform RS MPC' \
       'data/zero_hc-easy:dynamics-mse:$0$ RS MPC' \
       --drop_iterations 7 \
       --yaxis "10-episode dynamics MSE" --outfile report/easy-cheetah-dynamics-mse.pdf
