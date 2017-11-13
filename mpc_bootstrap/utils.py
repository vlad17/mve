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


def create_tf_session():
    """Create a TF session that doesn't eat all GPU memory."""
    config = tf.ConfigProto()
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


class Path(object):
    """Store rewards and transitions from a single fixed-horizon rollout"""

    def __init__(self, env, initial_obs, horizon):
        self._obs = np.empty((horizon, get_ob_dim(env)))
        self._next_obs = np.empty((horizon, get_ob_dim(env)))
        self._acs = np.empty((horizon, get_ac_dim(env)))
        self._rewards = np.empty(horizon)
        self._predicted_rewards = np.empty(horizon)
        self._idx = 0
        self._horizon = horizon
        self._obs[0] = initial_obs

    def next(self, next_obs, reward, ac, pred_reward):
        """Append a new transition to currently-stored ones"""
        assert self._idx < self._horizon, (self._idx, self._horizon)
        self._next_obs[self._idx] = next_obs
        self._rewards[self._idx] = reward
        self._acs[self._idx] = ac
        self._predicted_rewards[self._idx] = pred_reward
        self._idx += 1
        if self._idx < self._horizon:
            self._obs[self._idx] = next_obs
            return False
        return True

    def _check_all_data_has_been_collected(self):
        assert self._idx == self._horizon, (self._idx, self._horizon)

    @property
    def obs(self):
        """All observed states so far."""
        self._check_all_data_has_been_collected()
        return self._obs

    @property
    def acs(self):
        """All actions so far."""
        self._check_all_data_has_been_collected()
        return self._acs

    @property
    def rewards(self):
        """All rewards so far."""
        self._check_all_data_has_been_collected()
        return self._rewards

    @property
    def next_obs(self):
        """All states transitioned into so far."""
        self._check_all_data_has_been_collected()
        return self._next_obs

    @property
    def predicted_rewards(self):
        """All predicted rewards so far."""
        assert self._idx == self._horizon, (self._idx, self._horizon)
        return self._predicted_rewards

# pylint: disable=too-many-instance-attributes


class Dataset(object):
    """
    Stores all data for transitions across several rollouts.

    The order of actions, observations, and rewards returned by the
    stationary* methods is internally consistent: the action taken
    in dataset.stationary_acs()[i] is the action taken from state
    dataset.stationary_obs()[i], resulting in the i-th reward, etc.
    """

    def __init__(self, env, horizon):
        # not time by batch by state/action dimension order
        self.ac_dim = get_ac_dim(env)
        self.ob_dim = get_ob_dim(env)
        self.obs = np.empty((horizon, 0, self.ob_dim))
        self.next_obs = np.empty((horizon, 0, self.ob_dim))
        self.acs = np.empty((horizon, 0, self.ac_dim))
        self.rewards = np.empty((horizon, 0))
        self.predicted_rewards = np.empty((horizon, 0))
        self.labelled_acs = np.empty((horizon, 0, self.ac_dim))

    @property
    def horizon(self):
        """Environment horizon"""
        return self.obs.shape[0]

    def add_paths(self, paths):
        """Aggregate data from a list of paths"""
        obs = [path.obs[:, np.newaxis, :] for path in paths]
        obs.append(self.obs)
        acs = [path.acs[:, np.newaxis, :] for path in paths]
        acs.append(self.acs)
        rewards = [path.rewards[:, np.newaxis] for path in paths]
        rewards.append(self.rewards)
        next_obs = [path.next_obs[:, np.newaxis, :] for path in paths]
        next_obs.append(self.next_obs)
        predicted_rewards = [path.predicted_rewards[:, np.newaxis]
                             for path in paths]
        predicted_rewards.append(self.predicted_rewards)
        self.obs = np.concatenate(obs, axis=1)
        self.acs = np.concatenate(acs, axis=1)
        self.rewards = np.concatenate(rewards, axis=1)
        self.predicted_rewards = np.concatenate(predicted_rewards, axis=1)
        self.next_obs = np.concatenate(next_obs, axis=1)

    def unlabelled_obs(self):
        """Return observations for which there is no labelled action"""
        # assumes data is getting appended (so prefix stays the same)
        # as long as items are only modified through methods of this class
        # this should be ok

        obs = self.stationary_obs()
        acs = self.stationary_labelled_acs()
        num_unlabelled = len(obs) - len(acs)
        return obs[-num_unlabelled:]

    def label_obs(self, acs):
        """Label the first len(acs) observations with the given actions"""
        if acs is None:
            return
        env_horizon = self.obs.shape[0]
        num_acs, ac_dim = acs.shape
        assert ac_dim == self.ac_dim, (ac_dim, self.ac_dim)
        assert num_acs % env_horizon == 0, (num_acs, env_horizon)
        acs = acs.reshape(env_horizon, -1, ac_dim)
        self.labelled_acs = np.concatenate(
            [self.labelled_acs, acs], axis=1)

    def stationary_labelled_acs(self):
        """Return all labelled actions across all rollouts"""
        return self.labelled_acs.reshape(-1, self.ac_dim)

    def stationary_rewards(self):
        """Return all rewards all rollouts"""
        return self.rewards.reshape(-1)

    def stationary_obs(self):
        """Return all observations across all rollouts"""
        return self.obs.reshape(-1, self.ob_dim)

    def stationary_next_obs(self):
        """Return all resulting observations across all rollouts"""
        return self.next_obs.reshape(-1, self.ob_dim)

    def stationary_acs(self):
        """Return all taken actions across all rollouts"""
        return self.acs.reshape(-1, self.ac_dim)

    def reward_bias(self, prediction_horizon):
        """
        Report observed prediction_horizon-step reward bias
        and the h-step reward
        """
        agg_rewards = np.cumsum(self.rewards.T, axis=1)
        h_step_rew = agg_rewards[:, prediction_horizon - 1:]
        h_step_rew[:, 1:] -= agg_rewards[:, :-prediction_horizon]
        if prediction_horizon > 1:
            pred_rew = self.predicted_rewards.T[:, :-(prediction_horizon - 1)]
        else:
            pred_rew = self.predicted_rewards.T
        return h_step_rew - pred_rew, h_step_rew


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
