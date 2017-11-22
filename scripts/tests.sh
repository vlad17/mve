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
        ray stop
    }
    trap note_failure EXIT

    ray start --head --num-gpus=1 2>&1 | tee ray-init.txt
    ray_addr="$(cat ray-init.txt | awk '/ray start --redis-address/ { print $4 }')"
    rm ray-init.txt

    tune_params_json='[{"smoothing": 3, "horizon": 5, "mpc_simulated_paths": 2, "mpc_horizon": 3, "onpol_paths": 3, "onpol_iters": 4, "warmup_paths_mpc": 1, "con_depth": 1, "con_width": 10, "con_epochs": 1, "dyn_depth": 1, "dyn_width": 8, "dyn_epochs": 1}, {"smoothing": 3, "horizon": 5, "mpc_simulated_paths": 2, "mpc_horizon": 3, "onpol_paths": 3, "onpol_iters": 5, "warmup_paths_mpc": 1, "con_depth": 1, "con_width": 10, "con_epochs": 1, "dyn_depth": 1, "dyn_width": 8, "dyn_epochs": 1}]'

    main_random="mpc_bootstrap/main_random_policy.py"
    main_mpc="mpc_bootstrap/main_mpc.py"
    main_bmpc="mpc_bootstrap/main_bootstrapped_mpc.py"
    main_ddpg="mpc_bootstrap/main_ddpg.py"
    tune="mpc_bootstrap/tune.py"

    experiment_flags="--exp_name basic_tests --verbose --horizon 5"
    random_flags="$experiment_flags --num_paths 8 --num_procs 2"
    dynamics_flags="--dyn_epochs 1 --dyn_depth 1 --dyn_width 8"
    mpc_flags="$experiment_flags $dynamics_flags --onpol_iters 2 --onpol_paths 3 --mpc_simulated_paths 2 --mpc_horizon 3"
    warmup_flags="--warmup_paths_random 2"
    nn_learner_flags="--con_depth 1 --con_width 1 --con_epochs 1"
    ddpg_flags="$experiment_flags $nn_learner_flags --onpol_iters 2 --onpol_paths 3 --warmup_iters 1"
    tune_flags="--ray_addr $ray_addr"

    cmds=()
    # Tune
    echo "$tune_params_json" > params.json
    cmds+=("python $tune $tune_flags --tunefile params.json")
    cmds+=("rm params.json")
    # Random
    cmds+=("python $main_random $random_flags")
    cmds+=("python $main_random $random_flags --env_name ant")
    cmds+=("python $main_random $random_flags --env_name walker2d")
    cmds+=("python $main_random $random_flags --env_name hc-easy")
    # MPC
    cmds+=("python $main_mpc $mpc_flags $warmup_flags")
    cmds+=("python $main_mpc $mpc_flags $warmup_flags --env_name hc-easy")
    cmds+=("python $main_mpc $mpc_flags $warmup_flags --onpol_iters 3 --exp_name plotexp")
    # DDPG
    cmds+=("python $main_ddpg $ddpg_flags")
    cmds+=("python $main_ddpg $ddpg_flags --training_batches 2")
    cmds+=("python $main_ddpg $ddpg_flags --training_batches 2 --log_every 1")
    # BMPC
    cmds+=("python $main_bmpc delta $mpc_flags $nn_learner_flags $warmup_flags")
    cmds+=("python $main_bmpc delta $mpc_flags $nn_learner_flags $warmup_flags --warmup_paths_mpc 1")
    cmds+=("python $main_bmpc delta $mpc_flags $nn_learner_flags $warmup_flags --explore_std 1")
    cmds+=("python $main_bmpc gaussian $mpc_flags $nn_learner_flags $warmup_flags")
    cmds+=("python $main_bmpc gaussian $mpc_flags $nn_learner_flags $warmup_flags --no_extra_explore")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags --param_noise_exploitation")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags --param_noise_exploration")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags --env_name ant")
    cmds+=("python $main_bmpc ddpg $mpc_flags $nn_learner_flags $warmup_flags --env_name walker2d")
    cmds+=("python $main_bmpc zero $mpc_flags")
    # Check that warmup caching doesn't break anything. These commands should
    # create two new cache directories.
    rm -rf data/test_warmup_cache
    cmds+=("python $main_mpc $mpc_flags --warmup_cache_dir data/test_warmup_cache --warmup_paths_random 3 --warmup_paths_mpc 2")
    cmds+=("python $main_mpc $mpc_flags --warmup_cache_dir data/test_warmup_cache --warmup_paths_random 3 --warmup_paths_mpc 2")
    cmds+=("python $main_mpc $mpc_flags --warmup_cache_dir data/test_warmup_cache --warmup_paths_random 4 --warmup_paths_mpc 2")

    for cmd in "${cmds[@]}"; do
        box "$cmd"
        $cmd
    done

    num_caches="$(ls data/test_warmup_cache | wc -l)"
    if [[ "$num_caches" -ne 2 ]]; then
        echo "num_caches = $num_caches"
        exit 1
    fi

    instance="data/plotexp_hc-hard:dynamics-mse:x"
    cmd="python mpc_bootstrap/plot.py $instance --outfile x.pdf --yaxis x --notex"
    $cmd
    rm x.pdf

    instance="data/plotexp_hc-hard:avg-return:x"
    hlines="--hlines data/plotexp_hc-hard:dynamics-mse:yy"
    cmd="python mpc_bootstrap/plot.py $instance --outfile y.pdf --yaxis y --notex $hlines"
    $cmd
    rm y.pdf

    ray stop
    trap '' EXIT
}

main
