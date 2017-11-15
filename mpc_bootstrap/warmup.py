"""
Methods and flags relating to creating a warmup dataset
with standard MPC data.

Unfortunately, caching requires a lot of care for replicable experiments:
we need to make sure that we only re-use things from the same seed we started
with, that we re-seed after using random data to fill the cache the first time
so that the resulting run starts from the same seed future experiments,
which re-use the cache do, and that we take care to only re-use the cache
in the same settings. This is why we don't cache warmup data currently.
TODO: consider caching warmup data. Be careful to only use the cache
when all relevant (MPC, Dyanmics, experiment env) parameters are the same.
"""

import numpy as np
import tensorflow as tf

from dataset import one_shot_dataset
from dynamics import NNDynamicsModel
from flags import Flags
from log import debug
from mpc import MPC
from multiprocessing_env import mk_venv
from random_policy import RandomPolicy
from sample import sample_venv
from utils import create_tf_session, seed_everything


class WarmupFlags(Flags):
    """Flags specifying MPC-warmup related settings"""

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Add flags for warmup to the parser"""
        if argument_group is None:
            argument_group = parser.add_argument_group('warmup')
        argument_group.add_argument(
            '--warmup_paths_random',
            type=int,
            default=10,
            help='run this many random rollouts to add to the '
            'starting dataset of transitions')
        argument_group.add_argument(
            '--warmup_iterations_mpc',
            type=int,
            default=0,
            help='run this many iterations of MPC for warmup')

    def __init__(self, args):
        self.warmup_paths_random = args.warmup_paths_random
        self.warmup_iterations_mpc = args.warmup_iterations_mpc

    @staticmethod
    def name():
        return "warmup"


def _sample_random(venv, data):
    random_policy = RandomPolicy(venv)
    paths = sample_venv(venv, random_policy, data.max_horizon)
    data.add_paths(paths)
    debug('random agent warmup complete')


def _mpc_loop(iterations, data, dynamics, controller, venv):
    for i in range(iterations):
        dynamics.fit(data)
        paths = sample_venv(venv, controller, data.max_horizon)
        data.add_paths(paths)
        most_recent = one_shot_dataset(paths)
        ave_return = np.mean(most_recent.per_episode_rewards())
        mse = dynamics.dataset_mse(most_recent)
        debug('completed {: 4d} MPC warmup iterations (returns {:5.0f} '
              'dynamics mse {:8.2f})', i + 1, ave_return, mse)


def _sample_mpc(tf_reward, venv, warmup_flags, mpc_flags, dyn_flags, data):
    # sample in a different graph to avoid variable shadowing
    seed = tf.get_default_graph().seed
    g = tf.Graph()
    with g.as_default():
        with create_tf_session() as sess:
            seed_everything(seed)
            dynamics = NNDynamicsModel(venv, sess, data, dyn_flags)
            controller = MPC(
                venv, dynamics, mpc_flags.mpc_horizon, tf_reward,
                mpc_flags.mpc_simulated_paths, sess, learner=None)
            sess.run(tf.global_variables_initializer())
            g.finalize()

            _mpc_loop(
                warmup_flags.warmup_iterations_mpc,
                data, dynamics, controller, venv)


def add_warmup_data(flags, data):
    """
    Creates warmup data, adding it to the parameter dataset.

    Flags should contain warmup, mpc, dynamics, experiment flags.
    """

    if flags.warmup.warmup_paths_random > 0:
        venv = mk_venv(flags.experiment.mk_env,
                       flags.warmup.warmup_paths_random)
        _sample_random(venv, data)

    if flags.warmup.warmup_iterations_mpc > 0:
        venv = mk_venv(flags.experiment.mk_env, flags.mpc.onpol_paths)
        tf_reward = flags.experiment.mk_env().tf_reward
        _sample_mpc(
            tf_reward, venv, flags.warmup, flags.mpc, flags.dynamics, data)
