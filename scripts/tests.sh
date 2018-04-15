#! /usr/bin/env bash

# Very simple invocations that validate things don't blow up in all
# command-line configurations. Doesn't do any semantic checking, but will catch
# egregious errors. Don't source this. 
#
#   ./scripts/tests.sh
#   ./scripts/tests.sh --dry-run

set -eo pipefail

set -u

if [ $# -gt 1 ] || [ $# -eq 1 ] && [ "$1" != "--dry-run" ] ; then
    echo 'usage: ./scripts/tests.sh [--dry-run]' 1>&2
    exit 1
fi

if [ $# -eq 1 ] ; then
    DRY_RUN="true"
else
    DRY_RUN="false"
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
    box "${cmd/$relative/mve}"
    if [ "$DRY_RUN" != "true" ] ; then
        sh -c "$2"
        rm "$1"
    fi
}

main() {
    cmd=""
    function note_failure {
        relative="../mve"
        box "${cmd/$relative/mve}"
        cd ..
        rm -rf _test
    }
    trap note_failure EXIT

    ray_yaml="
    main: ddpg
    learner_depth:
        grid_search: [1, 2]
    learner_width: 8
    learner_batches_per_timestep: 1
    exp_name: {\\\"eval\\\": \\\"'learner_depth{}'.format(spec.config.learner_depth)\\\"}
    learner_batch_size: 4
    dynamics_evaluation_envs: 4
    timesteps: 200
    "

    if [ -d _test ] ; then
        rm -rf _test
    fi
    mkdir _test
    cd _test

    main_random="../mve/main_random_policy.py"
    main_mve="../mve/main_mve.py"
    main_ddpg="../mve/main_ddpg.py"
    main_sac="../mve/main_sac.py"
    tune="../mve/main_ray.py"
    plot="../mve/plot.py"
    eval_q="../mve/main_evaluate_qval.py"
    ddpg_dyn_plot="../mve/ddpg_dynamics_plot.py"
    
    experiment_flags_no_ts="--exp_name basic_tests --verbose true --horizon 5"
    experiment_flags="$experiment_flags_no_ts"
    random_flags="$experiment_flags --num_paths 8"
    dynamics_flags="--dynamics_batches_per_timestep 1 --dyn_depth 1 --dyn_width 8"
    ddpg_only_flags="--learner_depth 1 --learner_width 8 --learner_batches_per_timestep 1 "
    ddpg_only_flags="$ddpg_only_flags --learner_batch_size 4"
    ddpg_flags="$experiment_flags_no_ts $ddpg_only_flags --timesteps 200 --dynamics_evaluation_envs 4"
    sac_flags="$ddpg_flags"

    cmds=()
    # Random
    cmds+=("python $main_random $random_flags --env_name humanoid --env_parallelism 1")
    # DDPG
    cmds+=("python $main_ddpg $ddpg_flags --render_every 1 --dynamics_evaluation_envs 2")
    cmds+=("python $main_ddpg $ddpg_flags --discount 0.9 --dyn_dropout 0.5 --dyn_l2_reg 0.1")
    cmds+=("python $main_ddpg $ddpg_flags --imaginary_buffer 1.0")
    cmds+=("python $main_ddpg $ddpg_flags --critic_lr 1e-4")
    cmds+=("python $main_ddpg $ddpg_flags --actor_lr 1e-4")
    cmds+=("python $main_ddpg $ddpg_flags --critic_l2_reg 1e-2")
    cmds+=("python $main_ddpg $ddpg_flags --drop_tdk true")
    cmds+=("python $main_ddpg $ddpg_flags --dynamics_type oracle --ddpg_mve true")
    cmds+=("python $main_ddpg $ddpg_flags --dynamics_type learned --ddpg_mve true $dynamics_flags")
    cmds+=("python $main_ddpg $ddpg_flags --dynamics_type learned --ddpg_mve true $dynamics_flags --dyn_bn true")
    cmds+=("python $main_ddpg $ddpg_flags --dynamics_type oracle --ddpg_mve true --model_horizon 3 --div_by_h true")
    cmds+=("python $main_ddpg $ddpg_flags --sample_interval 200")
    cmds+=("python $main_ddpg $ddpg_flags --ddpg_min_buf_size 200")
    cmds+=("python $main_ddpg $ddpg_flags --ddpg_min_buf_size 200 --ddpg_mve true $dynamics_flags --dynamics_early_stop 0.3")
    # SAC
    cmds+=("python $main_sac $sac_flags --reparameterization_trick false")
    cmds+=("python $main_sac $sac_flags --policy_lr 1e-4 --value_lr 1e-4 --temperature 2.0")
    cmds+=("python $main_sac $sac_flags --model_horizon 5 --sac_mve true")
    # Envs
    cmds+=("python $main_random $random_flags --env_name ant")
    cmds+=("python $main_random $random_flags --env_name walker2d")
    cmds+=("python $main_random $random_flags --env_name hc2")
    cmds+=("python $main_random $random_flags --env_name pusher")
    cmds+=("python $main_random $random_flags --env_name hopper")
    cmds+=("python $main_random $random_flags --env_name swimmer")
    cmds+=("python $main_random $random_flags --env_name humanoid")
    cmds+=("python $main_random $random_flags --env_name acrobot")
    # Test recovery
    persist_flags="--persist_replay_buffer true"
    ddpg_dyn_flags="--dynamics_type learned --ddpg_mve true"
    cmds+=("python $main_ddpg $ddpg_flags --save_every 1 --exp_name ddpg_save $persist_flags $ddpg_dyn_flags")
    savedir="data/ddpg_save_hc/1/checkpoints"
    restore="--exp_name ddpg_restore"
    restore="$restore --restore_ddpg $savedir/ddpg.ckpt-00000200"
    restore="--restore_dynamics data/ddpg_save_hc/1/checkpoints/dynamics.ckpt-00000200"
    restore="$restore $ddpg_only_flags $ddpg_dyn_flags $restore"
    restore="$restore --restore_normalization data/ddpg_save_hc/1/checkpoints/normalization.ckpt-00000200"
    restore_buffer=" --restore_buffer $savedir/persistable_dataset.ckpt-00000200"
    cmds+=("python $main_ddpg $ddpg_flags $restore $restore_buffer")
    # Plot tests
    eval_q_flags="--episodes 3 --output_path out.pdf --notex"
    eval_q_flags="$eval_q_flags --lims -1 1 -1 1 --title hello --episodes 1 --horizon 15"
    cmds+=("python $eval_q $eval_q_flags $restore")
    cmds+=("python $ddpg_dyn_plot --plot_nenvs 4 --plot_horizon 3 $restore")
    cmds+=("python $main_ddpg $ddpg_flags --exp_name plotexp --evaluate_every 100 --ddpg_mve true --dynamics_type learned")
    for cmd in "${cmds[@]}"; do
        relative="../mve"
        box "${cmd/$relative/mve}"
        if [ "$DRY_RUN" != "true" ] ; then
            $cmd
        fi
    done

    # Tune
    cmd="echo \"$ray_yaml\" > params.yaml && python $tune --median_stop -1 --experiment_name testray --config params.yaml --self_host"
    hermetic_file params.yaml "$cmd"

    instance="data/plotexp_hc:current policy reward mean:x"
    cmd="python $plot \"$instance\" --outfile x.pdf --yaxis x --notex --xaxis xx --xrange 0 200"
    hermetic_file "x.pdf" "$cmd"

    instance="data/plotexp_hc:current policy reward mean:x"
    hlines="data/plotexp_hc:current policy reward mean:yy"
    cmd="python $plot \"$instance\" --outfile y.pdf --yaxis y --notex --hlines \"$hlines\" --smoothing 2"
    cmd="$cmd --yrange -1 100"
    hermetic_file "y.pdf" "$cmd"

    cd ..
    rm -rf _test
    trap '' EXIT
}

main
