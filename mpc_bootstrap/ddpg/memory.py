"""Implement a replay buffer storing transitions"""

import numpy as np

# RingBuffer-based implementation, requires some adaptation for vectorized
# environments


class _RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        """return items at requested indices"""
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        """Add v to ring buffer, evicting old data if necessary"""
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def _array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class _Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = _RingBuffer(limit, shape=observation_shape)
        self.actions = _RingBuffer(limit, shape=action_shape)
        self.rewards = _RingBuffer(limit, shape=(1,))
        self.terminals1 = _RingBuffer(limit, shape=(1,))
        self.observations1 = _RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        """Sample from memory"""
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(
            self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': _array_min2d(obs0_batch),
            'obs1': _array_min2d(obs1_batch),
            'rewards': _array_min2d(reward_batch),
            'actions': _array_min2d(action_batch),
            'terminals1': _array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        """Append transition to memory"""
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        """returns number of entries in the replay buffer"""
        return len(self.observations0)


class Memory(object):
    """
    An OpenAI-compatible memory class that wraps an mpc_bootstrap
    utils.Dataset.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def sample(self, batch_size):
        """Draw a sample from the replay buffer"""
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(
            self.nb_entries - 2, size=batch_size)

        obs0_batch = self._dataset.stationary_obs()[batch_idxs]
        obs1_batch = self._dataset.stationary_next_obs()[batch_idxs]
        action_batch = self._dataset.stationary_acs()[batch_idxs]
        reward_batch = self._dataset.stationary_rewards()[batch_idxs]
        terminal1_batch = np.zeros(batch_size)

        result = {
            'obs0': _array_min2d(obs0_batch),
            'obs1': _array_min2d(obs1_batch),
            'rewards': _array_min2d(reward_batch),
            'actions': _array_min2d(action_batch),
            'terminals1': _array_min2d(terminal1_batch),
        }
        # TODO: consider normalizing obs / rewards here
        return result

    @property
    def nb_entries(self):
        """return number of available transitions in buffer"""
        return len(self._dataset.stationary_obs())
