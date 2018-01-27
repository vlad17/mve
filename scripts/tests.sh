#! /usr/bin/env bash

# Very simple invocations that validate things don't blow up in all
# command-line configurations. Doesn't do any semantic checking, but will catch
# egregious errors. Don't source this. If arguments are specified, they are
# assumed to be the path for the MuJoCo key and installation.
#
#   ./scripts/tests.sh # uses ~/.mujoco/ paths
#   MUJOCO_DIRECTORY=DIR ./scripts/tests.sh # uses DIR instead of ~/.mujoco

set -eo pipefail

if [[ -z "${MUJOCO_DIRECTORY}" ]]; then
    sleep 0
else
    export MUJOCO_PY_MJKEY_PATH=$(readlink -f "${MUJOCO_DIRECTORY}/mjkey.txt")
    export MUJOCO_PY_MJPRO_PATH=$(readlink -f "${MUJOCO_DIRECTORY}/mjpro131")
fi

set -u

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

    tune_params_json='[{"smoothing": 3, "rs_learner": "ddpg", "horizon": 5, "simulated_paths": 2, "mpc_horizon": 3, "onpol_paths": 3, "onpol_iters": 4, "learner_depth": 1, "learner_width": 10, "learner_batches_per_timestep": 1, "dyn_depth": 1, "dyn_width": 8, "dynamics_batches_per_timestep": 1, "evaluation_envs": 10}, {"smoothing": 3, "rs_learner": "ddpg", "horizon": 5, "simulated_paths": 2, "mpc_horizon": 3, "onpol_paths": 3, "onpol_iters": 5, "learner_depth": 1, "learner_width": 10, "learner_batches_per_timestep": 1, "dyn_depth": 1, "dyn_width": 8, "dynamics_batches_per_timestep": 1, "evaluation_envs": 10}]'
    tune_params_run="python ../cmpc/main_cmpc.py --rs_learner ddpg --horizon 5 --simulated_paths 2"
    tune_params_run="$tune_params_run --mpc_horizon 3 --onpol_paths 3 --onpol_iters 4"
    tune_params_run="$tune_params_run --learner_depth 1 --learner_width 10 --learner_batches_per_timestep 1"
    tune_params_run="$tune_params_run --dyn_depth 1 --dyn_width 8 --dynamics_batches_per_timestep 1"
    tune_params_run="$tune_params_run --evaluation_envs 10"
    
    if [ -d _test ] ; then
        rm -rf _test
    fi
    mkdir _test
    cd _test

    ray start --head --num-gpus=1 --no-ui --node-ip-address 127.0.0.1 2>&1 | tee ray-init.txt
    ray_addr="$(cat ray-init.txt | awk '/ray start --redis-address/ { print $4 }')"
    rm ray-init.txt

    main_random="../cmpc/main_random_policy.py"
    main_cmpc="../cmpc/main_cmpc.py"
    main_ddpg="../cmpc/main_ddpg.py"
    tune="../cmpc/tune.py"
    cmpc_plot="../cmpc/plot.py"
    cmpc_plot_dyn="../cmpc/plot_dynamics.py"
    eval_q="../cmpc/main_evaluate_qval.py"

    experiment_flags="--exp_name basic_tests --verbose --horizon 5"
    random_flags="$experiment_flags --num_paths 8"
    dynamics_flags="--dynamics_batches_per_timestep 1 --dyn_depth 1 --dyn_width 8"
    mpc_flags="$experiment_flags $dynamics_flags --onpol_iters 2 --mpc_horizon 3"
    mpc_flags="$mpc_flags --evaluation_envs 10"
    short_mpc_flags="$experiment_flags $dynamics_flags --onpol_iters 2 --mpc_horizon 6"
    short_mpc_flags="$short_mpc_flags --onpol_paths 2 --simulated_paths 2 --evaluation_envs 10"
    rs_mpc_flags="$mpc_flags --onpol_paths 3 --simulated_paths 2"
    ddpg_only_flags="--learner_depth 1 --learner_width 1 --learner_batches_per_timestep 1 "
    ddpg_only_flags="$ddpg_only_flags --learner_batch_size 4"
    ddpg_flags="$experiment_flags $ddpg_only_flags --episodes 2"
    tune_flags="--ray_addr $ray_addr"

    cmds=()
    # Random
    cmds+=("python $main_random $random_flags")
    cmds+=("python $main_random $random_flags --env_name ant")
    cmds+=("python $main_random $random_flags --env_name walker2d")
    cmds+=("python $main_random $random_flags --env_name hc2")
    # MPC
    cmds+=("python $main_cmpc $rs_mpc_flags")
    cmds+=("python $main_cmpc $short_mpc_flags")
    cmds+=("python $main_cmpc $rs_mpc_flags --render_every 1")
    cmds+=("python $main_cmpc $rs_mpc_flags --evaluation_envs 2")
    cmds+=("python $main_cmpc $rs_mpc_flags --discount 0.9")
    # DDPG
    cmds+=("python $main_ddpg $ddpg_flags")
    cmds+=("python $main_ddpg $ddpg_flags --critic_lr 1e-4")
    cmds+=("python $main_ddpg $ddpg_flags --actor_lr 1e-4")
    cmds+=("python $main_ddpg $ddpg_flags --critic_l2_reg 1e-2")
    cmds+=("python $main_ddpg $ddpg_flags")
    cmds+=("python $main_ddpg $ddpg_flags --mixture_estimator oracle --q_target_mixture")
    cmds+=("python $main_ddpg $ddpg_flags --mixture_estimator oracle --actor_critic_mixture")
    mix_all="--mixture_estimator oracle --q_target_mixture --actor_critic_mixture"
    cmds+=("python $main_ddpg $ddpg_flags $mix_all")
    cmds+=("python $main_ddpg $ddpg_flags --ddpg_min_buf_size 200")
    # CMPC
    cloning="--mpc_optimizer random_shooter --rs_learner cloning"
    cloning="$cloning --cloning_learner_depth 1 --cloning_learner_width 1"
    cloning="$cloning --cloning_learner_batches_per_timestep 1"
    cmds+=("python $main_cmpc $cloning $rs_mpc_flags")
    cmds+=("python $main_cmpc $cloning $rs_mpc_flags --cloning_min_buf_size 200")
    rs_ddpg="--mpc_optimizer random_shooter --rs_learner ddpg"
    cmds+=("python $main_cmpc $rs_ddpg $rs_mpc_flags $ddpg_only_flags")
    rs_zero="--mpc_optimizer random_shooter --rs_learner zero"
    cmds+=("python $main_cmpc $rs_zero $rs_mpc_flags")
    shooter_flags="--opt_horizon 1"
    cmds+=("python $main_cmpc $rs_zero $rs_mpc_flags $shooter_flags")
    cmds+=("python $main_cmpc $rs_zero $rs_mpc_flags $shooter_flags --true_dynamics")
    cmds+=("python $main_cmpc $rs_zero $rs_mpc_flags $shooter_flags --true_dynamics --rs_n_envs 2")
    cmds+=("python $main_cmpc $rs_zero $rs_mpc_flags $shooter_flags --true_dynamics --simulated_paths 10 --rs_n_envs 5")
    colocation_flags="--coloc_primal_steps 2"
    colocation_flags="$colocation_flags --coloc_dual_steps 2 --coloc_primal_tol 1e-2"
    colocation_flags="$colocation_flags --coloc_primal_lr 1e-4 --coloc_dual_lr 1e-3"
    colocation_flags="$colocation_flags --onpol_paths 1"
    colocation_flags="$colocation_flags --mpc_optimizer colocation"
    cmds+=("python $main_cmpc $mpc_flags $colocation_flags --coloc_opt_horizon 2")
    cmds+=("python $main_cmpc $mpc_flags $colocation_flags")
    # Test recovery
    cmds+=("python $main_cmpc $rs_mpc_flags --exp_name saved --save_every 2")
    expected_dyn_save="data/saved_hc/3/checkpoints/dynamics.ckpt-00000002"
    expected_rb_save="data/saved_hc/3/checkpoints/persistable_dataset.ckpt-00000002"
    restore="--restore_dynamics $expected_dyn_save --restore_buffer $expected_rb_save"
    cmds+=("python $main_cmpc $rs_mpc_flags --exp_name restored $restore")
    cmds+=("python $main_ddpg $ddpg_flags --save_every 1 --exp_name ddpg_save")
    savedir="data/ddpg_save_hc/3/checkpoints"
    restore="--exp_name ddpg_restore --restore_buffer $savedir/persistable_dataset.ckpt-00000002"
    restore="$restore --restore_ddpg $savedir/ddpg.ckpt-00000002"
    cmds+=("python $main_ddpg $ddpg_flags $restore")
    # Plot tests
    restore_ddpg="--seed 3 --restore_ddpg $savedir/ddpg.ckpt-00000002"
    eval_q_flags="--horizon 10 --mixture_horizon 5 --notex"
    eval_q_flags="$eval_q_flags --episodes 3"
    cmds+=("python $eval_q $eval_q_flags $restore_ddpg $ddpg_only_flags")
    cmds+=("python $eval_q $eval_q_flags $restore_ddpg $ddpg_only_flags --evaluate_oracle_subsample 10")
    cmds+=("python $main_cmpc $rs_mpc_flags --onpol_iters 3 --exp_name plotexp")
    # Make sure the ray command isn't itself buggy
    cmds+=("$tune_params_run")
    for cmd in "${cmds[@]}"; do
        relative="../cmpc"
        box "${cmd/$relative/cmpc}"
        $cmd
    done

    # Tune
    cmd="echo '$tune_params_json' > params.json && python $tune $tune_flags --tunefile params.json"
    hermetic_file params.json "$cmd"

    instance="data/plotexp_hc:reward mean:x"
    cmd="python $cmpc_plot \"$instance\" --outfile x.pdf --yaxis x --notex --xaxis xx --xrange 0 30"
    hermetic_file "x.pdf" "$cmd"

    instance="data/plotexp_hc:reward mean:x"
    hlines="data/plotexp_hc:reward mean:yy"
    cmd="python $cmpc_plot \"$instance\" --outfile y.pdf --yaxis y --notex --hlines \"$hlines\" --smoothing 2"
    cmd="$cmd --yrange -1 100"
    hermetic_file "y.pdf" "$cmd"

    instance="data/plotexp_hc:label1 data/plotexp_hc:label2"
    cmd="python $cmpc_plot_dyn $instance --outfile z.pdf --notex --smoothing 2 --hsteps 1 2 3"
    hermetic_file "z.pdf" "$cmd"
    
    cmd="python $cmpc_plot_dyn $instance --outfile z.pdf --notex --yrange 0 1 --hsteps 2"
    hermetic_file "z.pdf" "$cmd"

    cd ..
    rm -rf _test
    ray stop
    trap '' EXIT
}

main
