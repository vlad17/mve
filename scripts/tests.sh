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

# Argument 1 is the generated file name, arg2 is the command that makes it
hermetic_file() {
    if [ -f "$1" ] ; then
        rm "$1"
    fi
    sh -c "$2"
    rm "$1"
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

    tune_params_json='[{"smoothing": 3, "horizon": 5, "simulated_paths": 2, "mpc_horizon": 3, "onpol_paths": 3, "onpol_iters": 4, "warmup_paths_mpc": 1, "learner_depth": 1, "learner_width": 10, "learner_nbatches": 1, "dyn_depth": 1, "dyn_width": 8, "dyn_epochs": 1, "warmup_paths_random": 1}, {"smoothing": 3, "horizon": 5, "simulated_paths": 2, "mpc_horizon": 3, "onpol_paths": 3, "onpol_iters": 5, "warmup_paths_mpc": 1, "learner_depth": 1, "learner_width": 10, "learner_nbatches": 1, "dyn_depth": 1, "dyn_width": 8, "dyn_epochs": 1, "warmup_paths_random": 1}]'

    main_random="cmpc/main_random_policy.py"
    main_cmpc="cmpc/main_cmpc.py"
    main_ddpg="cmpc/main_ddpg.py"
    tune="cmpc/tune.py"

    experiment_flags="--exp_name basic_tests --verbose --horizon 5"
    random_flags="$experiment_flags --num_paths 8"
    dynamics_flags="--dyn_epochs 1 --dyn_depth 1 --dyn_width 8"
    mpc_flags="$experiment_flags $dynamics_flags --onpol_iters 2 --mpc_horizon 3"
    rs_mpc_flags=" --onpol_paths 3 --simulated_paths 2"
    warmup_flags="--warmup_paths_random 1"
    nn_learner_flags="--learner_depth 1 --learner_width 1 --learner_nbatches 2"
    ddpg_flags="$experiment_flags $nn_learner_flags"
    tune_flags="--ray_addr $ray_addr"

    cmds=()
    # Random
    cmds+=("python $main_random $random_flags")
    cmds+=("python $main_random $random_flags --env_name ant")
    cmds+=("python $main_random $random_flags --env_name walker2d")
    cmds+=("python $main_random $random_flags --env_name hc-easy")
    # MPC
    cmds+=("python $main_cmpc none $rs_mpc_flags $warmup_flags")
    cmds+=("python $main_cmpc none $rs_mpc_flags --warmup_paths_random 2 --renormalize")
    cmds+=("python $main_cmpc none $rs_mpc_flags --warmup_paths_random 0 --renormalize")
    cmds+=("python $main_cmpc none $rs_mpc_flags $warmup_flags --env_name hc-easy")
    cmds+=("python $main_cmpc none $rs_mpc_flags $warmup_flags --onpol_iters 3 --exp_name plotexp")
    # DDPG
    cmds+=("python $main_ddpg $ddpg_flags --episodes 2")
    cmds+=("python $main_ddpg $ddpg_flags --critic_lr 1e-3 --episodes 2")
    cmds+=("python $main_ddpg $ddpg_flags --actor_lr 1e-3 --episodes 2")
    # CMPC
    cmds+=("python $main_cmpc cloner $rs_mpc_flags $nn_learner_flags $warmup_flags")
    cmds+=("python $main_cmpc cloner $rs_mpc_flags $nn_learner_flags $warmup_flags")
    cmds+=("python $main_cmpc ddpg $rs_mpc_flags $ddpg_flags $warmup_flags")
    cmds+=("python $main_cmpc zero $rs_mpc_flags $warmup_flags")
    shooter_flags="--planner shooter --opt_horizon 1"
    cmds+=("python $main_cmpc zero $rs_mpc_flags $warmup_flags $shooter_flags")
    colocation_flags="--planner colocation --coloc_opt_horizon 2 --coloc_primal_steps 2"
    colocation_flags="$colocation_flags --coloc_dual_steps 2 --coloc_primal_tol 1e-2"
    colocation_flags="$colocation_flags --coloc_primal_lr 1e-4 --coloc_dual_lr 1e-3"
    colocation_flags="--onpol_paths 1"
    cmds+=("python $main_cmpc zero $mpc_flags $warmup_flags $colocation_flags")
    # Check that warmup caching doesn't break anything. These commands should
    # create two new cache directories.
    rm -rf data/test_warmup_cache
    cmds+=("python $main_cmpc none $rs_mpc_flags --warmup_cache_dir data/test_warmup_cache --warmup_paths_random 3 --warmup_paths_mpc 2")
    cmds+=("python $main_cmpc none $rs_mpc_flags --warmup_cache_dir data/test_warmup_cache --warmup_paths_random 3 --warmup_paths_mpc 2")
    cmds+=("python $main_cmpc none $rs_mpc_flags --warmup_cache_dir data/test_warmup_cache --warmup_paths_random 4 --warmup_paths_mpc 2")

    for cmd in "${cmds[@]}"; do
        box "$cmd"
        $cmd
    done

    num_caches="$(ls data/test_warmup_cache | wc -l)"
    if [[ "$num_caches" -ne 2 ]]; then
        echo "num_caches = $num_caches"
        exit 1
    fi

    # Tune
    cmd="echo '$tune_params_json' > /tmp/params.json && python $tune $tune_flags --tunefile /tmp/params.json"
    hermetic_file /tmp/params.json "$cmd"

    instance="data/plotexp_hc-hard:dynamics mse:x"
    cmd="python cmpc/plot.py \"$instance\" --outfile /tmp/x.pdf --yaxis x --notex"
    hermetic_file "/tmp/x.pdf" "$cmd"

    instance="data/plotexp_hc-hard:reward mean:x"
    hlines="data/plotexp_hc-hard:dynamics mse:yy"
    cmd="python cmpc/plot.py \"$instance\" --outfile /tmp/y.pdf --yaxis y --notex --hlines \"$hlines\" --smoothing 2"
    hermetic_file "/tmp/y.pdf" "$cmd"

    ray stop
    trap '' EXIT
}

main
