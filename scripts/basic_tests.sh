#! /usr/bin/env bash

# Very simple invocations that validate things don't blow up in all
# command-line configurations. Doesn't do any semantic checking, but will catch
# egregious errors. Don't source this. If arguments are specified, they are
# assumed to be the path for the MuJoCo key and installation.
#
#   ./scripts/basic_tests.sh # uses ~/.mujoco/ paths
#   ./scripts/basic_tests.sh mjkey.txt mjpro131 # uses specified paths

set -euo pipefail

if [[ "$#" -eq 2 ]]; then
    export MUJOCO_PY_MJKEY_PATH=$(readlink -f "$1")
    export MUJOCO_PY_MJPRO_PATH=$(readlink -f "$2")
fi

main() {
    cmd=""
    function note_failure {
        set +x
        msg="* $cmd *"
        echo "$msg" | sed 's/./\*/g'
        echo "$msg"
        echo "$msg" | sed 's/./\*/g'
    }
    trap note_failure EXIT

    exp_flags="--verbose"
    alg_flags="--onpol_iters 1 --onpol_paths 3 --random_paths 3 --horizon 5"
    dyn_flags="--dyn_epochs 1 --dyn_depth 1 --dyn_width 8"
    common_flags="$exp_flags $alg_flags $dyn_flags"
    mpc_flags="--mpc_simulated_paths 2 --mpc_horizon 3"
    con_flags="--con_depth 1 --con_width 1 --con_epochs 1"

    flags=()
    flags+=("mpc $common_flags $mpc_flags")
    flags+=("random $common_flags")
    flags+=("gaussian_bootstrap $common_flags $mpc_flags $con_flags")
    flags+=("delta_bootstrap $common_flags $mpc_flags $con_flags --explore_std 1")
    flags+=("gaussian_dagger $common_flags $mpc_flags $con_flags")
    flags+=("gaussian_dagger $common_flags $mpc_flags $con_flags --delay 5")
    flags+=("random $common_flags --env_name hc-easy")
    flags+=("delta_bootstrap $common_flags $mpc_flags $con_flags")
    flags+=("gaussian_bootstrap $common_flags $mpc_flags $con_flags --no_extra_explore")
    flags+=("gaussian_dagger $common_flags $mpc_flags $con_flags --con_stale_data 1")
    flags+=("gaussian_dagger $common_flags $mpc_flags $con_flags --con_stale_data 3")    
    flags+=("random $common_flags --onpol_iters 3 --exp_name plotexp")

    for flag in "${flags[@]}"; do
        cmd="python mpc_bootstrap/main.py $flag"
        set -x
        python mpc_bootstrap/main.py $flag
        set +x
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
