"""Generate DDPG rollouts and train on them"""

import json
import os

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from dataset import Dataset, one_shot_dataset
from ddpg_learner_flags import DDPGLearnerFlags
from experiment_flags import ExperimentFlags
from flags import (convert_flags_to_json, parse_args, Flags)
from multiprocessing_env import mk_venv
from sample import sample_venv
from utils import (make_data_directory, seed_everything, timeit,
                   create_tf_session, log_statistics)
import log
import logz


def _train(args):
    env = args.experiment.mk_env()
    data = Dataset.from_env(env, args.experiment.horizon,
                            args.experiment.bufsize)
    venv = mk_venv(args.experiment.mk_env, args.run.onpol_paths)
    sess = create_tf_session()

    learner = args.ddpg.make_learner(venv, sess, data)

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for itr in range(args.run.warmup_iters):
        with timeit('warmup round {}'.format(itr)):
            if data.obs.size:
                learner.fit(data)
            paths = sample_venv(venv, learner, args.experiment.horizon)
            data.add_paths(paths)

    agg_returns = []
    for itr in range(args.run.onpol_iters):
        with timeit('learner fit'):
            if data.obs.size:
                learner.fit(data)

        with timeit('sample learner'):
            paths = sample_venv(venv, learner, args.experiment.horizon)

        with timeit('adding paths to dataset'):
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = one_shot_dataset(paths)
            returns = most_recent.per_episode_rewards()

        agg_returns += returns
        if (itr + 1) % args.run.log_every == 0:
            log_itr = (itr + 1) // args.run.log_every
            logz.log_tabular('iteration', log_itr)
            log_statistics('return', agg_returns)
            logz.dump_tabular()
            agg_returns = []

    sess.__exit__(None, None, None)


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


class RunFlags(Flags):
    """Flags relevant for running DDPG over multiple iterations."""

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Adds flags to an argparse parser."""
        if argument_group is None:
            argument_group = parser.add_argument_group('mpc')
        argument_group.add_argument(
            '--onpol_iters',
            type=int,
            default=300,
            help='number of outermost on policy aggregation iterations',
        )
        argument_group.add_argument(
            '--onpol_paths',
            type=int,
            default=1,
            help='number of rollouts per on policy iteration',
        )
        argument_group.add_argument(
            '--warmup_iters',
            type=int,
            default=100,
            help='how many warmup iterations DDPG gets (not recorded)',
        )
        argument_group.add_argument(
            '--log_every',
            type=int,
            default=10,
            help='how frequently to log statistics',
        )

    @staticmethod
    def name():
        return "run"

    def __init__(self, args):
        self.onpol_iters = args.onpol_iters
        self.onpol_paths = args.onpol_paths
        self.warmup_iters = args.warmup_iters
        self.log_every = args.log_every


if __name__ == "__main__":
    flags = [ExperimentFlags, RunFlags, DDPGLearnerFlags]
    _args = parse_args(flags)
    _main(_args)
