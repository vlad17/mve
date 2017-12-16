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
        relative="../cmpc"
        box "${cmd/$relative/cmpc}"
        ray stop
        cd ..
        rm -rf _test
    }
    trap note_failure EXIT

    tune_params_json='[{"smoothing": 3, "horizon": 5, "simulated_paths": 2, "mpc_horizon": 3, "onpol_paths": 3, "onpol_iters": 4, "learner_depth": 1, "learner_width": 10, "learner_nbatches": 1, "dyn_depth": 1, "dyn_width": 8, "dyn_epochs": 1}, {"smoothing": 3, "horizon": 5, "simulated_paths": 2, "mpc_horizon": 3, "onpol_paths": 3, "onpol_iters": 5, "learner_depth": 1, "learner_width": 10, "learner_nbatches": 1, "dyn_depth": 1, "dyn_width": 8, "dyn_epochs": 1}]'
    
    if [ -d _test ] ; then
        rm -rf _test
    fi
    mkdir _test
    cd _test

    ray start --head --num-gpus=1 2>&1 | tee ray-init.txt
    ray_addr="$(cat ray-init.txt | awk '/ray start --redis-address/ { print $4 }')"
    rm ray-init.txt

    main_random="../cmpc/main_random_policy.py"
    main_cmpc="../cmpc/main_cmpc.py"
    main_ddpg="../cmpc/main_ddpg.py"
    tune="../cmpc/tune.py"
    cmpc_plot="../cmpc/plot.py"

    experiment_flags="--exp_name basic_tests --verbose --horizon 5"
    random_flags="$experiment_flags --num_paths 8"
    dynamics_flags="--dyn_epochs 1 --dyn_depth 1 --dyn_width 8"
    mpc_flags="$experiment_flags $dynamics_flags --onpol_iters 2 --mpc_horizon 3"
    mpc_flags="$mpc_flags --evaluation_envs 10"
    short_mpc_flags="$experiment_flags $dynamics_flags --onpol_iters 2 --mpc_horizon 6"
    short_mpc_flags="$short_mpc_flags --onpol_paths 2 --simulated_paths 2 --evaluation_envs 10"
    rs_mpc_flags="$mpc_flags --onpol_paths 3 --simulated_paths 2"
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
    cmds+=("python $main_cmpc rs $rs_mpc_flags")
    cmds+=("python $main_cmpc rs $short_mpc_flags")
    cmds+=("python $main_cmpc rs $rs_mpc_flags --render_every 1")
    cmds+=("python $main_cmpc rs $rs_mpc_flags --env_name hc-easy")
    cmds+=("python $main_cmpc rs $rs_mpc_flags --onpol_iters 3 --exp_name plotexp")
    cmds+=("python $main_cmpc rs $rs_mpc_flags --evaluation_envs 2")
    # DDPG
    cmds+=("python $main_ddpg $ddpg_flags --episodes 2")
    cmds+=("python $main_ddpg $ddpg_flags --critic_lr 1e-3 --episodes 2")
    cmds+=("python $main_ddpg $ddpg_flags --actor_lr 1e-3 --episodes 2")
    # CMPC
    cmds+=("python $main_cmpc rs_cloning $rs_mpc_flags $nn_learner_flags")
    cmds+=("python $main_cmpc rs_cloning $rs_mpc_flags $nn_learner_flags")
    cmds+=("python $main_cmpc rs_ddpg $rs_mpc_flags $ddpg_flags")
    cmds+=("python $main_cmpc rs_zero $rs_mpc_flags")
    shooter_flags="--opt_horizon 1"
    cmds+=("python $main_cmpc rs_zero $rs_mpc_flags $shooter_flags")
    colocation_flags="--coloc_primal_steps 2"
    colocation_flags="$colocation_flags --coloc_dual_steps 2 --coloc_primal_tol 1e-2"
    colocation_flags="$colocation_flags --coloc_primal_lr 1e-4 --coloc_dual_lr 1e-3"
    colocation_flags="$colocation_flags --onpol_paths 1"
    cmds+=("python $main_cmpc colocation $mpc_flags $colocation_flags --coloc_opt_horizon 2")
    cmds+=("python $main_cmpc colocation $mpc_flags $colocation_flags")
    # Test dynamics recovery
    cmds+=("python $main_cmpc rs $rs_mpc_flags --exp_name saved --save_every 2")
    expected_save="data/saved_hc-hard/3/checkpoints/dynamics.ckpt-00000002"
    cmds+=("python $main_cmpc rs $rs_mpc_flags --exp_name restored --restore_dynamics $expected_save")

    for cmd in "${cmds[@]}"; do
        box "$cmd"
        $cmd
    done

    # Tune
    cmd="echo '$tune_params_json' > params.json && python $tune $tune_flags --tunefile params.json"
    hermetic_file params.json "$cmd"

    instance="data/plotexp_hc-hard:reward mse:x"
    cmd="python $cmpc_plot \"$instance\" --outfile x.pdf --yaxis x --notex"
    hermetic_file "x.pdf" "$cmd"

    instance="data/plotexp_hc-hard:reward mean:x"
    hlines="data/plotexp_hc-hard:reward mse:yy"
    cmd="python $cmpc_plot \"$instance\" --outfile y.pdf --yaxis y --notex --hlines \"$hlines\" --smoothing 2"
    hermetic_file "y.pdf" "$cmd"

    cd ..
    rm -rf _test
    ray stop
    trap '' EXIT
}

main
