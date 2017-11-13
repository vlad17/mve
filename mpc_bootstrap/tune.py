"""Search for best hyperparameters."""

import os
import json
import copy

# import mujoco for weird dlopen reasons
import mujoco_py # pylint: disable=unused-import
import tensorflow as tf
import ray

from dynamics import DynamicsFlags
from mpc_flags import MpcFlags
from experiment_flags import ExperimentFlags
from warmup import WarmupFlags
from flags import convert_flags_to_json as flags_to_json
from flags import parse_args
from utils import (make_data_directory, seed_everything)
import log
import logz
from main_mpc import _train as train_mpc

def _main(args):
    log.init(args.experiment.verbose)

    exp_name = args.experiment.exp_name
    env_name = args.experiment.env_name
    datadir = "{}_{}".format(exp_name, env_name)
    logdir = make_data_directory(datadir)

    ray.init() # TODO: supply redis ip as cmd line arg

    @ray.remote
    def _train_mpc(_logdir, seed, i):
        # Save params to disk.
        logdir_seed = os.path.join(_logdir, str(seed))
        logdir_seed_proc = os.path.join(logdir_seed, str(i))
        logz.configure_output_dir(logdir_seed_proc)
        with open(os.path.join(logdir_seed, 'params.json'), 'w') as f:
            json.dump(flags_to_json(args), f, sort_keys=True, indent=4)

        args_inst = copy.deepcopy(args)
        args_inst.mpc.onpol_paths = 5 + (5 * i)
        log.debug('testing: --onpol_paths', args_inst.mpc.onpol_paths)
        train_mpc(args_inst)

    for seed in args.experiment.seed:
        # Run experiment.
        g = tf.Graph()
        with g.as_default():
            seed_everything(seed)
            ray.get([_train_mpc.remote(logdir, seed, i) for i in range(3)])


if __name__ == "__main__":
    _args = parse_args([ExperimentFlags, MpcFlags, DynamicsFlags, WarmupFlags])
    _main(_args)
