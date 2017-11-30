"""Generate random rollouts."""

import os
import json

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from dataset import one_shot_dataset
from experiment_flags import ExperimentFlags
from flags import (Flags, convert_flags_to_json, parse_args)
from multiprocessing_env import mk_venv
from random_policy import RandomPolicy
from sample import sample_venv
from utils import (make_data_directory, seed_everything, log_statistics)
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

    @staticmethod
    def name():
        return "random"

    def __init__(self, args):
        self.num_paths = args.num_paths


def _train(args):
    venv = mk_venv(args.experiment.mk_env, args.random.num_paths)
    random_policy = RandomPolicy(venv)
    paths = sample_venv(venv, random_policy, args.experiment.horizon)
    data = one_shot_dataset(paths)
    returns = data.per_episode_rewards()
    log_statistics('return', returns)
    logz.dump_tabular()


def _main(args):
    log.init(args.experiment.verbose)
    logdir_name = args.experiment.log_directory()
    logdir = make_data_directory(logdir_name)

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
