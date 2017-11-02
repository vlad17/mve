#!/bin/bash
# Very simple invocations that validate things don't blow up in all command-line configurations.
# Doesn't do any semantic checking, but will catch egregious errors.
# Don't source this.
# If arguments are specified, they are assumed to be the path for the
# MuJoCo key and installation.
#
# ./scripts/basic_tests.sh # uses ~/.mujoco/ paths
# ./scripts/basic_tests.sh mjkey.txt mjpro131 # uses specified paths

set -euo pipefail

export MUJOCO_PY_MJKEY_PATH=$(readlink -f "$1")
export MUJOCO_PY_MJPRO_PATH=$(readlink -f "$2")

flags="--dyn_epochs 1 --con_epochs 1 --ep_len 5 --mpc_horizon 3"
flags+=" --simulated_paths 2 --onpol_paths 3 --random_paths 3"
flags+=" --dyn_depth 1 --dyn_width 8 --con_depth 1 --con_width 8"
small_flags="$flags --onpol_iters 1"

cmd=""
instance=""
function note_failure {
    echo "****************************************"
    echo "FAILURE ($cmd): $instance"
    echo "****************************************"
}
trap note_failure EXIT

cd mpc_bootstrap

cmd="main.py"

instance="--agent mpc"
python main.py $small_flags $instance

instance="--agent random"
python main.py $small_flags $instance

instance="--agent bootstrap"
python main.py $small_flags $instance

instance="--agent bootstrap --explore_std 1"
python main.py $small_flags $instance

instance="--agent dagger"
python main.py $small_flags $instance

instance="--agent bootstrap --hard_cost"
python main.py $small_flags $instance

instance="--agent random --onpol_iters 3 --exp_name plotexp"
python main.py $flags $instance

cmd="plot.py"

instance="--value DynamicsMSE --legend x"
../scripts/fake-display.sh python plot.py data/plotexp_HalfCheetah-v1 $instance
rm DynamicsMSE.pdf

instance="--value AverageReturn --legend x"
../scripts/fake-display.sh python plot.py data/plotexp_HalfCheetah-v1 $instance
rm AverageReturn.pdf

trap '' EXIT
