"""
Common recipes for dealing with TF and gym classes, as well as common
Python patterns
"""

from contextlib import contextmanager
import random
import time

from terminaltables import SingleTable
import tensorflow as tf
import gym
import numpy as np

from controller import Controller
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


def _finite_env(space):
    return all(np.isfinite(space.low)) and all(np.isfinite(space.high))


def scale_to_box(space, relative):
    """
    Given a hyper-rectangle specified by a gym box space, scale
    relative coordinates between 0 and 1 to the box's coordinates,
    such that the relative vector of all zeros has the smallest
    coordinates (the "bottom left" corner) and vice-versa for ones.

    If environment is infinite, no scaling is performed.
    """
    if not _finite_env(space):
        return relative
    relative *= (space.high - space.low)
    relative += space.low
    return relative


def scale_from_box(space, absolute):
    """
    Given a hyper-rectangle specified by a gym box space, scale
    exact coordinates from within that space to
    relative coordinates between 0 and 1.

    If environment is infinite, no scaling is performed.
    """
    if not _finite_env(space):
        return absolute
    absolute -= space.low
    absolute /= (space.high - space.low)
    return absolute


def create_random_tf_action(ac_space):
    """
    Given an environment `env` with states of length s and actions of length a,
    `create_random_tf_action(env.action_space)` will return a function `f` that
    takes in an n x s tensor and returns an n x a tensor drawn uniformly at
    random from `env.action_space`.
    """
    def _policy(state_ns):
        n = tf.shape(state_ns)[0]
        ac_dim = ac_space.low.shape
        ac_na = tf.random_uniform((n,) + ac_dim)
        return scale_to_box(ac_space, ac_na)
    return _policy


def build_mlp(input_placeholder,
              output_size=1,
              scope='mlp',
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None,
              reuse=None,
              l2reg=0,
              activation_norm=None):
    """
    Create an MLP with the corresponding hyperparameters. Make sure to keep
    the scope the same and to set reuse=True to reuse the same weight
    parameters between invocations.

    activation_norm should be a unary function like
    tf.layers.batch_normalization.
    """
    out = input_placeholder
    if l2reg > 0:
        l2reg = tf.contrib.layers.l2_regularizer(l2reg)
    else:
        l2reg = None
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            if activation_norm is None:
                out = tf.layers.dense(out, size, activation=activation,
                                      kernel_regularizer=l2reg)
            else:
                # apply norm before activation, for consistency with papers
                # and OpenAI
                out = tf.layers.dense(out, size, activation=None,
                                      kernel_regularizer=l2reg)
                out = activation_norm(out)
                if activation is not None:
                    out = activation(out)
        out = tf.layers.dense(out, output_size, activation=output_activation,
                              kernel_regularizer=l2reg)
    return out


def rate_limit(limit, fn, *args_n):
    """
    Suppose args_n is a list of arguments for the function fn, each of which
    is a numpy array of dimension at least 1. Further, assume that fn
    returns a tuple of numpy arrays of dimension at least 1, where
    the length along the first axis of the returned arrays is the same as
    that of the arguments given to fn.

    Then rate_limit(limit, fn, *args_n) returns the result one
    would expect from fn(*args_n) but with multiple function calls
    so that fn is never called with arrays whose first axis
    length is greater than limit.
    """
    n = len(args_n[0])
    if n <= limit:
        return fn(*args_n)

    partial_slices = [arg[:limit] for arg in args_n]
    partial_returns = fn(*partial_slices)
    returns = [np.empty((n,) + ret.shape[1:], dtype=ret.dtype)
               for ret in partial_returns]
    for dst, src in zip(returns, partial_returns):
        dst[:limit] = src

    for i in range(limit, n, limit):
        loc = slice(i, min(i + limit, n))
        partial_slices = [arg[loc] for arg in args_n]
        partial_returns = fn(*partial_slices)
        for dst, src in zip(returns, partial_returns):
            dst[loc] = src

    return returns


class AssignableStatistic:
    """
    Need stateful statistics so we don't have to feed
    stats through a feed_dict every time we want to use the
    dynamics.

    This class keeps track of the mean and std as given.
    """

    def __init__(self, suffix, initial_mean, initial_std):
        initial_mean = np.asarray(initial_mean).astype('float32')
        initial_std = np.asarray(initial_std).astype('float32')
        self._mean_var = tf.get_variable(
            name='mean_' + suffix, trainable=False,
            initializer=initial_mean)
        self._std_var = tf.get_variable(
            name='std_' + suffix, trainable=False,
            initializer=initial_std)
        self._mean_ph = tf.placeholder(tf.float32, initial_mean.shape)
        self._std_ph = tf.placeholder(tf.float32, initial_std.shape)
        self._assign_both = tf.group(
            tf.assign(self._mean_var, self._mean_ph),
            tf.assign(self._std_var, self._std_ph))

    def update_statistics(self, mean, std):
        """
        Update the stateful statistics using the default session.
        """
        tf.get_default_session().run(
            self._assign_both, feed_dict={
                self._mean_ph: mean,
                self._std_ph: std})

    def tf_normalize(self, x):
        """Normalize a value according to these statistics"""
        return (x - self._mean_var) / (self._std_var)

    def tf_denormalize(self, x):
        """Denormalize a value according to these statistics"""
        return x * self._std_var + self._mean_var


def as_controller(act):
    """
    Generates a controller interface to a single act() method,
    which should accept a tensor of states and return a tensor of actions.
    """
    return _ActorAsController(act)


class _ActorAsController(Controller):
    def __init__(self, act):
        self._act = act

    def act(self, states_ns):
        return self._act(states_ns), None, None

    def planning_horizon(self):
        return 0


def print_table(data):
    """
    Use terminaltables to pretty-print a table to the terminal, where the
    table should be an even nested list of strings.
    """
    table = SingleTable(data)
    table.inner_column_border = False
    print(table.table)


def timesteps(paths):
    """Return the total number of timesteps in a list of trajectories"""
    return sum(len(path.rewards) for path in paths)
