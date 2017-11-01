from contextlib import contextmanager
import time

import tensorflow as tf
import gym
import numpy as np

@contextmanager
def timeit(name, print_time):
    t = time.time()
    yield
    t = time.time() - t
    if print_time:
        print('{} took {:0.1f} seconds'.format(name, t))

def get_ac_dim(env):
    ac_space = env.action_space
    assert isinstance(ac_space, gym.spaces.Box), type(ac_space)
    assert len(ac_space.shape) == 1, ac_space.shape
    return ac_space.shape[0]


def get_ob_dim(env):
    ob_space = env.observation_space
    assert isinstance(ob_space, gym.spaces.Box), type(ob_space)
    assert len(ob_space.shape) == 1, ob_space.shape
    return ob_space.shape[0]


class Path:
    def __init__(self, env, initial_obs, horizon):
        super().__init__()
        self._obs = np.empty((horizon, get_ob_dim(env)))
        self._next_obs = np.empty((horizon, get_ob_dim(env)))
        self._acs = np.empty((horizon, get_ac_dim(env)))
        self._rewards = np.empty(horizon)
        self._idx = 0
        self._horizon = horizon
        self._obs[0] = initial_obs

    def next(self, next_obs, reward, ac):
        assert self._idx < self._horizon, (self._idx, self._horizon)
        self._next_obs[self._idx] = next_obs
        self._rewards[self._idx] = reward
        self._acs[self._idx] = ac
        self._idx += 1
        if self._idx < self._horizon:
            self._obs[self._idx] = next_obs
            return False
        return True

    @property
    def obs(self):
        assert self._idx == self._horizon, (self._idx, self._horizon)
        return self._obs

    @property
    def acs(self):
        assert self._idx == self._horizon, (self._idx, self._horizon)
        return self._acs

    @property
    def rewards(self):
        assert self._idx == self._horizon, (self._idx, self._horizon)
        return self._rewards

    @property
    def next_obs(self):
        assert self._idx == self._horizon, (self._idx, self._horizon)
        return self._next_obs


class Dataset:
    """ Stores all data in time x batch x state/action dim order """

    def __init__(self, env, horizon):
        super().__init__()
        self.ac_dim = get_ac_dim(env)
        self.ob_dim = get_ob_dim(env)
        self.obs = np.empty((horizon, 0, self.ob_dim))
        self.next_obs = np.empty((horizon, 0, self.ob_dim))
        self.acs = np.empty((horizon, 0, self.ac_dim))
        self.rewards = np.empty((horizon, 0))
        self.labelled_acs = np.empty((horizon, 0, self.ac_dim))

    def add_paths(self, paths):
        """ Aggregate data """
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
        # assumes data is getting appended (so prefix stays the same)
        # as long as items are only modified through methods of this class
        # this should be ok

        obs = self.stationary_obs()
        acs = self.stationary_labelled_acs()
        num_unlabelled = len(obs) - len(acs)
        return obs[-num_unlabelled:]

    def label_obs(self, acs):
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
        return self.labelled_acs.reshape(-1, self.ac_dim)

    def stationary_obs(self):
        return self.obs.reshape(-1, self.ob_dim)

    def stationary_next_obs(self):
        return self.next_obs.reshape(-1, self.ob_dim)

    def stationary_acs(self):
        return self.acs.reshape(-1, self.ac_dim)


def build_mlp(input_placeholder,
              output_size=1,
              scope='mlp',
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None,
              reuse=None
             ):
    out = input_placeholder
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out
