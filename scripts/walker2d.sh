#! /usr/bin/env bash

set -euo pipefail

# Runs Walker2d environment experiments, using the usual (hard) reward
# function. This script should be run from the repo root as follows:
#
#   ./scripts/walker2d.sh
#
# This generates the following images in the report/ folder:
#
#   walker2d-return.pdf - returns for 1-4
#   walker2d-reward-bias.pdf - for 1-2
#   walker2d-reward-mse.pdf - for 1-2
#   walker2d-dynamics-mse.pdf - for 1-2
#
# (1) BMPC-DDPG, the MPC controller (2) BMPC-DDPG, the underlying DDPG learner
# (3) Uniform-sampling MPC (4) Raw DDPG with no exploration noise.

main() {
    # If dry_run is true, we run the experiments for only a short period of
    # time, just to sanity check that everything runs correctly.
    local -r dry_run=false
    if $dry_run; then
        local -r ddpg_onpol_iters=3
        local -r horizon=20
        local -r log_every=1
        local -r mpc_onpol_iters=3
        local -r mpc_simulated_paths=10
        local -r num_paths=10
        local -r plot_flags="--notex"
        local -r training_batches=10
        local -r warmup_paths_mpc=1
        local -r warmup_iters=1
    else
        local -r ddpg_onpol_iters=300
        local -r horizon=1000
        local -r log_every=10
        local -r mpc_onpol_iters=30
        local -r mpc_simulated_paths=1000
        local -r num_paths=1000
        local -r plot_flags=""
        local -r training_batches=4000
        local -r warmup_paths_mpc=25
        local -r warmup_iters=25
    fi

    set -x
    python mpc_bootstrap/main_random_policy.py \
        --verbose \
        --exp_name random \
        --env_name walker2d \
        --num_paths "$num_paths" \
        --horizon "$horizon" \
        --seed 1 5 10

    python mpc_bootstrap/main_bootstrapped_mpc.py ddpg \
        --verbose \
        --exp_name bmpc-ddpg \
        --env_name walker2d \
        --horizon "$horizon" \
        --mpc_simulated_paths "$mpc_simulated_paths" \
        --con_epochs 10 \
        --onpol_iters "$mpc_onpol_iters" \
        --seed 1 5 10 \
        --con_width 64 \
        --con_depth 2 \
        --con_learning_rate 1e-4 \
        --critic_lr 1e-4 \
        --warmup_paths_mpc "$warmup_paths_mpc" \
        --action_noise_exploration 0.1 \
        --onpol_paths 10

    python mpc_bootstrap/main_mpc.py \
        --verbose \
        --exp_name mpc \
        --env_name walker2d \
        --horizon "$horizon" \
        --mpc_simulated_paths "$mpc_simulated_paths" \
        --onpol_iters "$mpc_onpol_iters" \
        --seed 1 5 10 \
        --warmup_paths_mpc "$warmup_paths_mpc" \
        --onpol_paths 10

    # note ddpg does 1 rollout / iter, hence the diff in numbers
    python mpc_bootstrap/main_ddpg.py \
        --verbose \
        --exp_name ddpg \
        --env_name walker2d \
        --horizon "$horizon" \
        --training_batches "$training_batches" \
        --seed 1 5 10 \
        --con_width 64 \
        --con_depth 2 \
        --con_learning_rate 1e-4 \
        --critic_lr 1e-4 \
        --warmup_iters "$warmup_iters" \
        --onpol_paths 1 \
        --onpol_iters "$ddpg_onpol_iters" \
        --log_every "$log_every"

    python mpc_bootstrap/plot.py \
       $plot_flags \
       'data/mpc_walker2d:avg-return:Uniform MPC' \
       'data/bmpc-ddpg_walker2d:avg-return:DDPG-BMPC (controller)' \
       'data/bmpc-ddpg_walker2d:avg-learner-return:DDPG-BMPC (learner)' \
       'data/ddpg_walker2d:avg-return:lone DDGP' \
       --yaxis "10-episode avg return" --outfile report/walker2d-return.pdf

    python mpc_bootstrap/plot.py \
       $plot_flags \
       'data/mpc_walker2d:standardized-reward-bias:Uniform MPC' \
       'data/bmpc-ddpg_walker2d:standardized-reward-bias:DDPG-BMPC (controller)' \
       --yaxis "10-episode standardized reward bias" --outfile report/walker2d-reward-bias.pdf

    python mpc_bootstrap/plot.py \
       $plot_flags \
       'data/mpc_walker2d:standardized-reward-mse:Uniform MPC' \
       'data/bmpc-ddpg_walker2d:standardized-reward-mse:DDPG-BMPC (controller)' \
       --yaxis "10-episode standardized reward MSE" --outfile report/walker2d-reward-mse.pdf

    python mpc_bootstrap/plot.py \
       $plot_flags \
       'data/mpc_walker2d:dynamics-mse:Uniform MPC' \
       'data/bmpc-ddpg_walker2d:dynamics-mse:DDPG-BMPC (controller)' \
       --yaxis "10-episode dynamics MSE" --outfile report/walker2d-dynamics-mse.pdf
    set +x
}

main
