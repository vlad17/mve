"""
Common recipes for dealing with TF and gym classes, as well as common
Python patterns
"""

from contextlib import contextmanager
import time
import types

import tensorflow as tf
import gym
import numpy as np


@contextmanager
def timeit(name, print_time):
    """Enclose a with-block with this to print out block runtime"""
    t = time.time()
    yield
    t = time.time() - t
    if print_time:
        print('{} took {:0.1f} seconds'.format(name, t))


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


class Path(object):
    """Store rewards and transitions from a single fixed-horizon rollout"""

    def __init__(self, env, initial_obs, horizon):
        self._obs = np.empty((horizon, get_ob_dim(env)))
        self._next_obs = np.empty((horizon, get_ob_dim(env)))
        self._acs = np.empty((horizon, get_ac_dim(env)))
        self._rewards = np.empty(horizon)
        self._idx = 0
        self._horizon = horizon
        self._obs[0] = initial_obs

    def next(self, next_obs, reward, ac):
        """Append a new transition to currently-stored ones"""
        assert self._idx < self._horizon, (self._idx, self._horizon)
        self._next_obs[self._idx] = next_obs
        self._rewards[self._idx] = reward
        self._acs[self._idx] = ac
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
        self.labelled_acs = np.empty((horizon, 0, self.ac_dim))

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
        self.obs = np.concatenate(obs, axis=1)
        self.acs = np.concatenate(acs, axis=1)
        self.rewards = np.concatenate(rewards, axis=1)
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

    def stationary_obs(self):
        """Return all observations across all rollouts"""
        return self.obs.reshape(-1, self.ob_dim)

    def stationary_next_obs(self):
        """Return all resulting observations across all rollouts"""
        return self.next_obs.reshape(-1, self.ob_dim)

    def stationary_acs(self):
        """Return all taken actions across all rollouts"""
        return self.acs.reshape(-1, self.ac_dim)


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


def inherit_doc(cls):
    """
    From SO: https://stackoverflow.com/questions/8100166

    This will copy all the missing documentation for methods from the parent
    classes.

    :param type cls: class to fix up.
    :return type: the fixed class.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
        elif isinstance(func, property) and not func.fget.__doc__:
            for parent in cls.__bases__:
                parprop = getattr(parent, name, None)
                if parprop and getattr(parprop.fget, '__doc__', None):
                    newprop = property(fget=func.fget,
                                       fset=func.fset,
                                       fdel=func.fdel,
                                       doc=parprop.fget.__doc__)
                    setattr(cls, name, newprop)
                    break

    return cls
