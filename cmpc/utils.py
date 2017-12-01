"""
Common recipes for dealing with TF and gym classes, as well as common
Python patterns
"""

from contextlib import contextmanager
import os
import random
import shutil
import time

import tensorflow as tf
import gym
import numpy as np

import log


def create_tf_session(gpu=True):
    """Create a TF session that doesn't eat all GPU memory."""
    if gpu:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
    config.gpu_options.allow_growth = True
    opt_opts = config.graph_options.optimizer_options
    opt_opts.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    return sess


def seed_everything(seed):
    """Seed random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


@contextmanager
def timeit(name):
    """Enclose a with-block with to debug-print out block runtime"""
    t = time.time()
    yield
    t = time.time() - t
    log.debug('{} took {:0.1f} seconds', name, t)


def get_ac_dim(env):
    """Retrieve action dimension, assuming a continuous space"""
    ac_space = env.action_space
    assert isinstance(ac_space, gym.spaces.Box), type(ac_space)
    assert len(ac_space.shape) == 1, ac_space.shape
    return ac_space.shape[0]


def get_ob_dim(env):
    """Retrieve observation dimension, assuming a continuous space"""
    ob_space = env.observation_space
    assert isinstance(ob_space, gym.spaces.Box), type(ob_space)
    assert len(ob_space.shape) == 1, ob_space.shape
    return ob_space.shape[0]


def create_random_tf_policy(ac_space):
    """
    Given an environment `env` with states of length s and actions of length a,
    `create_random_tf_policy(env.action_space)` will return a function `f` that
    takes in an n x s tensor and returns an n x a tensor drawn uniformly at
    random from `env.action_space`.
    """
    def _policy(state_ns, **_):
        n = tf.shape(state_ns)[0]
        ac_dim = ac_space.low.shape
        ac_na = tf.random_uniform((n,) + ac_dim)
        ac_na *= (ac_space.high - ac_space.low)
        ac_na += ac_space.low
        return ac_na
    return _policy


def build_mlp(input_placeholder,
              output_size=1,
              scope='mlp',
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None,
              reuse=None):
    """
    Create an MLP with the corresponding hyperparameters. Make sure to keep
    the scope the same and to set reuse=True to reuse the same weight
    parameters between invocations.
    """
    out = input_placeholder
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


def make_data_directory(name):
    """
    make_data_directory(name) will create a directory data/name and return the
    directory's name. If a data/name directory already exists, then it will be
    renamed data/name-i where i is the smallest integer such that data/name-i
    does not already exist. For example, imagine the data/ directory has the
    following contents:

        data/foo
        data/foo-1
        data/foo-2
        data/foo-3

    Then, make_data_directory("foo") will rename data/foo to data/foo-4 and
    then create a fresh data/foo directory.
    """
    # Make the data directory if it does not already exist.
    if not os.path.exists('data'):
        os.makedirs('data')

    name = os.path.join('data', name)
    ctr = 0
    logdir = name
    while os.path.exists(logdir):
        logdir = name + '-{}'.format(ctr)
        ctr += 1
    if ctr > 0:
        log.debug('Experiment already exists, moved old one to {}.', logdir)
        shutil.move(name, logdir)

    os.makedirs(name)
    with open(os.path.join(name, 'starttime.txt'), 'w') as f:
        print(time.strftime("%d-%m-%Y_%H-%M-%S"), file=f)

    return name
