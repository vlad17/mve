"""Generate MPC rollouts."""

import os
import json
import copy

# import mujoco for weird dlopen reasons
import mujoco_py # pylint: disable=unused-import
import tensorflow as tf

from dynamics_flags import DynamicsFlags
from mpc_flags import MpcFlags
from experiment_flags import ExperimentFlags
from flags import (convert_flags_to_json, parse_args)
from utils import (make_data_directory, seed_everything)
import log
import logz
import ray
from main_mpc import _train as train_mpc

def _main(args):
    log.init(args.experiment.verbose)

    exp_name = args.experiment.exp_name
    env_name = args.experiment.env_name
    datadir = "{}_{}".format(exp_name, env_name)
    logdir = make_data_directory(datadir)

    ray.init() # TODO: supply num_gpus as cmd line arg

    for seed in args.experiment.seed:
        # Save params to disk.
        logdir_seed = os.path.join(logdir, str(seed))
        logz.configure_output_dir(logdir_seed)
        with open(os.path.join(logdir_seed, 'params.json'), 'w') as f:
            json.dump(convert_flags_to_json(args), f, sort_keys=True, indent=4)

        @ray.remote
        def _train_mpc_inst(logdir_root, i):
            logdir_seed_proc = os.path.join(logdir_root, str(i))
            logz.configure_output_dir(logdir_seed_proc)

            args_inst = copy.deepcopy(args)
            args_inst.mpc.onpol_paths = 5 + (5 * i)
            print('testing: --onpol_paths', args_inst.mpc.onpol_paths)
            train_mpc(args_inst)

        # Run experiment.
        g = tf.Graph()
        with g.as_default():
            seed_everything(seed)
            ray.get([_train_mpc_inst.remote(logdir_seed, i) for i in range(3)])


if __name__ == "__main__":
    _args = parse_args([ExperimentFlags, MpcFlags, DynamicsFlags])
    _main(_args)
