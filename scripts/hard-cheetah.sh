#! /usr/bin/env bash
# Runs HalfCheetah environment experiments, using the usual (hard) reward function.
# This script should be run from the repo root as follows:
#
# ./scripts/hard-cost.sh
#
# This generates the following images in the report/ folder:
#   hard-cheetah-return.pdf - returns for 1-4
#   hard-cheetah-reward-bias.pdf - for 1-2
#   hard-cheetah-reward-mse.pdf - for 1-2
#   hard-cheetah-dynamics-mse.pdf - for 1-2
#
# (1) BMPC-DDPG, the MPC controller (2) BMPC-DDPG, the underlying DDPG learner
# (3) Uniform-sampling MPC (4) Raw DDPG with no exploration noise.


python mpc_bootstrap/main_bootstrapped_mpc.py ddpg --verbose --exp_name bmpc-ddpg --con_epochs 10 --onpol_iters 30 --seed 1 5 10 --con_width 64 --con_depth 2 --con_learning_rate 1e-4 --critic_lr 1e-4  --warmup_iterations_mpc 10 --action_noise_exploration 0.1 --onpol_paths 10

python mpc_bootstrap/main_mpc.py --verbose --exp_name mpc --onpol_iters 30 --seed 1 5 10  --warmup_iterations_mpc 10  --onpol_paths 10

# note ddpg does 1 rollout / iter, hence the diff in numbers
python mpc_bootstrap/main_ddpg.py --verbose --exp_name ddpg --training_batches 4000 --seed 1 5 10 --con_width 64 --con_depth 2 --con_learning_rate 1e-4 --critic_lr 1e-4  --warmup_iters 100 --onpol_paths 1 --onpol_iters 300 --log_every 10

python mpc_bootstrap/plot.py \
       'data/mpc_hc-hard:avg-return:Uniform MPC' \
       'data/bmpc-ddpg_hc-hard:avg-return:DDPG-BMPC (controller)' \
       'data/bmpc-ddpg_hc-hard:avg-learner-return:DDPG-BMPC (learner)' \
       'data/ddpg_hc-hard:avg-return:lone DDGP' \
       --yaxis "10-episode avg return" --outfile report/hard-cheetah-return.pdf

python mpc_bootstrap/plot.py \
       'data/mpc_hc-hard:standardized-reward-bias:Uniform MPC' \
       'data/bmpc-ddpg_hc-hard:standardized-reward-bias:DDPG-BMPC (controller)' \
       --yaxis "10-episode standardized reward bias" --outfile report/hard-cheetah-reward-bias.pdf

python mpc_bootstrap/plot.py \
       'data/mpc_hc-hard:standardized-reward-mse:Uniform MPC' \
       'data/bmpc-ddpg_hc-hard:standardized-reward-mse:DDPG-BMPC (controller)' \
       --yaxis "10-episode standardized reward MSE" --outfile report/hard-cheetah-reward-mse.pdf

python mpc_bootstrap/plot.py \
       'data/mpc_hc-hard:dynamics-mse:Uniform MPC' \
       'data/bmpc-ddpg_hc-hard:dynamics-mse:DDPG-BMPC (controller)' \
       --yaxis "10-episode dynamics MSE" --outfile report/hard-cheetah-dynamics-mse.pdf
