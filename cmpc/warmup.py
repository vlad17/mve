"""
Methods and flags relating to creating a warmup dataset with standard MPC data.
"""

import hashlib
import json
import os
import time

import numpy as np
import tensorflow as tf

from dataset import (Dataset, one_shot_dataset)
from dynamics import NNDynamicsModel
from flags import Flags
from log import debug
from mpc import MPC
from multiprocessing_env import make_venv
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
            '--warmup_paths_mpc',
            type=int,
            default=0,
            help='run this many total paths of MPC for warmup')
        argument_group.add_argument(
            '--warmup_cache_dir',
            type=str,
            default='data/warmup_cache',
            help='directory in which cached warmup data is stored')

    def __init__(self, args):
        self.warmup_paths_random = args.warmup_paths_random
        self.warmup_paths_mpc = args.warmup_paths_mpc
        self.warmup_cache_dir = args.warmup_cache_dir

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
            dynamics = NNDynamicsModel(
                venv, data, dyn_flags)
            controller = MPC(
                venv, dynamics, mpc_flags.mpc_horizon, tf_reward,
                mpc_flags.mpc_simulated_paths, learner=None)
            sess.run(tf.global_variables_initializer())
            g.finalize()
            with sess.as_default():
                _mpc_loop(
                    warmup_flags.warmup_paths_mpc,
                    data, dynamics, controller, venv)


def _hash_dict(d):
    """
    Return a hex digest of the hash of d. We want _hash_dict(d) to return the
    same value for the same value of d, even across different invocations of
    Python. By default, Python randomizes the hashes returned by hash, so we
    cannot use it. Instead we use a hashlib hash which is constant across
    different invocations.
    """
    hasher = hashlib.sha256()
    # Hashes can be order sensitive. We have to make sure to pass in items in
    # sorted order.
    for k, v in sorted(d.items()):
        # hasher takes in bytes, not strings.
        hasher.update(bytes(str(k), "utf-8"))
        hasher.update(bytes(str(v), "utf-8"))
    return hasher.hexdigest()


def add_warmup_data(flags, data):
    """
    Creates warmup data, adding it to the parameter dataset.

    Flags should contain warmup, mpc, dynamics, experiment flags.
    """
    assert data.size == 0, data.size

    # warmup_params includes every flag that can affect warmup data generation.
    # That is, warmup_flags completeley determines the contents of data after
    # `add_warmup_data` is called.
    warmup_params = {
        # Experiment flags.
        "seed": tf.get_default_graph().seed,
        "env_name": flags.experiment.env_name,
        "frame_skip": flags.experiment.frame_skip,
        "horizon": flags.experiment.horizon,
        "bufsize": flags.experiment.bufsize,
        # Warmup flags.
        "warmup_paths_random": flags.warmup.warmup_paths_random,
        "warmup_paths_mpc": flags.warmup.warmup_paths_mpc,
        # Dynamics flags.
        "dyn_depth": flags.dynamics.dyn_depth,
        "dyn_width": flags.dynamics.dyn_width,
        "dyn_learning_rate": flags.dynamics.dyn_learning_rate,
        "dyn_epochs": flags.dynamics.dyn_epochs,
        "dyn_batch_size": flags.dynamics.dyn_batch_size,
        # MPC flags.
        "mpc_simulated_paths": flags.mpc.mpc_simulated_paths,
        "mpc_horizon": flags.mpc.mpc_horizon,
    }
    warmup_params_str = json.dumps(warmup_params, sort_keys=True, indent=4)
    warmup_params_hash = _hash_dict(warmup_params)
    debug("warmup_params_hash = {}", warmup_params_hash)
    debug("warmup_params =\n{}", warmup_params_str)

    # Let warmup_cache_dir = "data/warmup_cache". "data/warmup_cache" will look
    # something like this (hashes truncated here for brevity):
    #
    #    data
    #    └── warmup_cache
    #        ├── 1c003ddeec53ef3771e6855793...
    #        │   ├── dataset.pickle
    #        │   └── params.json
    #        ├── 46adc6fb3cfce204b9be605832...
    #        │   ├── dataset.pickle
    #        │   └── params.json
    #        └── b420b1df3e6e20dd46514f9b5f...
    #            ├── dataset.pickle
    #            └── params.json
    #
    # A dataset with a given warmup_params is placed in a directory named
    # hash(warmup_params). Inside of this directory, we store the dataset in
    # "dataset.pickle" and a JSON version of the warmup_params in
    # "params.json". There are actually a couple more files stored as well; see
    # below for details.
    dataset_dir = os.path.join(flags.warmup.warmup_cache_dir,
                               str(warmup_params_hash))
    dataset_filename = os.path.join(dataset_dir, "dataset.pickle")
    params_filename = os.path.join(dataset_dir, "params.json")

    # Handle a cache hit.
    if os.path.isdir(dataset_dir):
        debug("Warmup cache hit! Directory {} exists.", dataset_dir)
        with open(params_filename, "r") as f:
            cached_warmup_params = json.load(f)
            assert warmup_params == cached_warmup_params, \
                (warmup_params, cached_warmup_params)
        cached_data = Dataset.load(dataset_filename)
        data.clone(cached_data)
        return
    debug("Warmup cache miss! Directory {} doesn't exist.", dataset_dir)

    if flags.warmup.warmup_paths_random > 0:
        venv = make_venv(flags.experiment.make_env,
                         flags.warmup.warmup_paths_random)
        _sample_random(venv, data)

    if flags.warmup.warmup_paths_mpc > 0:
        venv = make_venv(flags.experiment.make_env, 1)
        tf_reward = flags.experiment.make_env().tf_reward
        _sample_mpc(
            tf_reward, venv, flags.warmup, flags.mpc, flags.dynamics, data)

    # Update the cache.
    debug("Caching dataset and warmup params to {}.", dataset_dir)
    os.makedirs(dataset_dir)
    data.dump(dataset_filename)
    with open(params_filename, "w") as f:
        f.write(warmup_params_str + "\n")
    # In addition to the warmup params, we also write the time at which the
    # dataset was created and the git hash of the code that was used to create
    # it. These are useful to sanity check whether or not a cached dataset
    # should be deleted.
    with open(os.path.join(dataset_dir, "time_created.txt"), "w") as f:
        f.write(time.strftime("%l:%M%p on %b %d, %Y" + "\n"))
    with open(os.path.join(dataset_dir, "git_hash.txt"), "w") as f:
        f.write(str(flags.experiment.git_hash) + "\n")
