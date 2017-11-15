"""Generate random rollouts."""

import multiprocessing
import os
import json

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np

from dataset import one_shot_dataset
from experiment_flags import ExperimentFlags
from flags import (Flags, convert_flags_to_json, parse_args)
from multiprocessing_env import mk_venv
from random_policy import RandomPolicy
from sample import sample_venv
from utils import (make_data_directory, seed_everything, timeit)
import log
import logz


class RandomPolicyFlags(Flags):
    """Random policy flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        random = parser.add_argument_group('random')
        random.add_argument(
            '--num_paths',
            type=int,
            default=100,
            help='number of paths',
        )
        random.add_argument(
            '--horizon',
            type=int,
            default=1000,
            help='path horizon',
        )
        random.add_argument(
            '--num_procs',
            type=int,
            default=multiprocessing.cpu_count(),
            help='number of procs for venv',
        )

    @staticmethod
    def name():
        return "random"

    def __init__(self, args):
        # For simplicity, we enforce that the number of paths a multiple of the
        # number of processors.
        assert args.num_paths % args.num_procs == 0, args

        self.num_paths = args.num_paths
        self.horizon = args.horizon
        self.num_procs = args.num_procs


def _train(args):
    venv = mk_venv(args.experiment.mk_env, args.random.num_procs)
    all_paths = []
    random_policy = RandomPolicy(venv)
    for itr in range(args.random.num_paths // args.random.num_procs):
        start = itr * args.random.num_procs
        stop = (itr + 1) * args.random.num_procs
        with timeit('generating rollouts {}-{}'.format(start, stop)):
            paths = sample_venv(venv, random_policy, args.random.horizon)
            all_paths += paths

    data = one_shot_dataset(all_paths)
    returns = data.per_episode_rewards()
    logz.log_tabular('AverageReturn', np.mean(returns))
    logz.log_tabular('StdReturn', np.std(returns))
    logz.log_tabular('MinimumReturn', np.min(returns))
    logz.log_tabular('MaximumReturn', np.max(returns))
    logz.dump_tabular()


def _main(args):
    log.init(args.experiment.verbose)

    exp_name = args.experiment.exp_name
    env_name = args.experiment.env_name
    datadir = "{}_{}".format(exp_name, env_name)
    logdir = make_data_directory(datadir)

    for seed in args.experiment.seed:
        # Save params to disk.
        logdir_seed = os.path.join(logdir, str(seed))
        logz.configure_output_dir(logdir_seed)
        with open(os.path.join(logdir_seed, 'params.json'), 'w') as f:
            json.dump(convert_flags_to_json(args), f, sort_keys=True, indent=4)

        # Run experiment.
        g = tf.Graph()
        with g.as_default():
            seed_everything(seed)
            _train(args)


if __name__ == "__main__":
    _args = parse_args([ExperimentFlags, RandomPolicyFlags])
    _main(_args)
