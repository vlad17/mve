#! /usr/bin/env bash

# Very simple invocations that validate things don't blow up in all
# command-line configurations. Doesn't do any semantic checking, but will catch
# egregious errors. Don't source this. If arguments are specified, they are
# assumed to be the path for the MuJoCo key and installation.
#
#   ./scripts/tests.sh # uses ~/.mujoco/ paths
#   ./scripts/tests.sh mjkey.txt mjpro131 # uses specified paths

set -euo pipefail

if [[ "$#" -eq 2 ]]; then
    export MUJOCO_PY_MJKEY_PATH=$(readlink -f "$1")
    export MUJOCO_PY_MJPRO_PATH=$(readlink -f "$2")
fi

box() {
    msg="* $1 *"
    echo "$msg" | sed 's/./\*/g'
    echo "$msg"
    echo "$msg" | sed 's/./\*/g'
}

main() {
    cmd=""
    function note_failure {
        box "$cmd"
    }
    trap note_failure EXIT

    main_random="mpc_bootstrap/main_random_policy.py"
    main_mpc="mpc_bootstrap/main_mpc.py"
    main_bmpc="mpc_bootstrap/main_bootstrapped_mpc.py"

    tune_mpc="mpc_bootstrap/tune.py"
    experiment_flags="--exp_name basic_tests --verbose"
    random_flags="$experiment_flags --num_paths 8 --num_procs 2"
    dynamics_flags="--dyn_epochs 1 --dyn_depth 1 --dyn_width 8"
    mpc_flags="$experiment_flags $dynamics_flags --onpol_iters 1 --onpol_paths 3 --horizon 5 --mpc_simulated_paths 2 --mpc_horizon 3"
    warmup_flags="--warmup_paths_random 2"
    nn_learner_flags="--con_depth 1 --con_width 1 --con_epochs 1"

    cmds=()
    # Tune
    cmds+=("python $tune_mpc $mpc_flags $warmup_flags")
    # Random
    cmds+=("python $main_random $random_flags")
    cmds+=("python $main_random $random_flags --env_name ant")
    cmds+=("python $main_random $random_flags --env_name hc-easy")
    # MPC
    cmds+=("python $main_mpc $mpc_flags $warmup_flags")
    cmds+=("python $main_mpc $mpc_flags $warmup_flags --env_name hc-easy")
    cmds+=("python $main_mpc $mpc_flags $warmup_flags --onpol_iters 3 --exp_name plotexp")
    # BMPC
    cmds+=("python $main_bmpc delta $mpc_flags $nn_learner_flags $warmup_flags")
    cmds+=("python $main_bmpc delta $mpc_flags $nn_learner_flags $warmup_flags --warmup_iterations_mpc 1")
    cmds+=("python $main_bmpc delta $mpc_flags $nn_learner_flags $warmup_flags --explore_std 1")
    cmds+=("python $main_bmpc gaussian $mpc_flags $nn_learner_flags $warmup_flags")
    cmds+=("python $main_bmpc gaussian $mpc_flags $nn_learner_flags $warmup_flags --no_extra_explore")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags --param_noise_exploitation")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags --param_noise_exploration")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags --env_name ant")
    cmds+=("python $main_bmpc zero $mpc_flags")

    for cmd in "${cmds[@]}"; do
        box "$cmd"
        $cmd
    done

    instance="data/plotexp_hc-hard:DynamicsMSE:x"
    cmd="python mpc_bootstrap/plot.py $instance --outfile x.pdf --yaxis x --notex"
    $cmd
    rm x.pdf

    instance="data/plotexp_hc-hard:AverageReturn:x"
    cmd="python mpc_bootstrap/plot.py $instance --outfile y.pdf --yaxis y --notex"
    $cmd
    rm y.pdf

    trap '' EXIT
}

main
