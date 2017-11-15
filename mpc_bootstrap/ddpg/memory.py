"""Implement a replay buffer storing transitions"""

import numpy as np


class RingBuffer(object):
    """
    A RingBuffer is a finite-size cylic buffer that supports
    fast insertion and access.
    """

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

    def all_data(self):
        """return all data in the buffer"""
        if self.length == self.maxlen:
            return self.data
        assert self.start == 0, self.start
        return self.data[:self.length]

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

    def append_all(self, vs):
        """Append an array of values"""
        if len(vs) > self.maxlen:
            raise ValueError('cannot add array of size {} to ringbuf size {}'
                             .format(len(vs), self.length))
        if self.length < self.maxlen:
            assert self.start == 0, self.start
            room = self.maxlen - self.length
            to_fill = min(len(vs), room)
            self.data[self.length:self.length + to_fill] = vs[:to_fill]
            self.length += to_fill
            if to_fill == len(vs):
                return
            vs = vs[to_fill:]

        assert self.length == self.maxlen, (self.length, self.maxlen)
        to_fill = min(self.maxlen - self.start, len(vs))
        self.data[self.start:self.start + to_fill] = vs[:to_fill]

        self.start = (self.start + to_fill) % self.maxlen
        if to_fill == len(vs):
            return
        assert self.start == 0, self.start
        assert len(vs) < self.maxlen, (len(vs), self.maxlen)
        vs = vs[to_fill:]
        self.data[:len(vs)] = vs
        self.start = len(self.vs)


def _array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    """
    An OpenAI-compatible memory class that wraps an mpc_bootstrap
    utils.Dataset.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def sample(self, batch_size):
        """Draw a sample from the replay buffer"""
        batch_idxs = np.random.random_integers(
            self.nb_entries - 1, size=batch_size)

        # TODO: dataset should just have a sample method, or even better
        # to amortize random int gen time, a sample_many
        # method (dedup with all the *_learners fit() functions and
        # inside ddpg fit)
        obs0_batch = self._dataset.obs[batch_idxs]
        obs1_batch = self._dataset.next_obs[batch_idxs]
        action_batch = self._dataset.acs[batch_idxs]
        reward_batch = self._dataset.rewards[batch_idxs]
        terminal1_batch = self._dataset.terminals[batch_idxs]

        result = {
            'obs0': _array_min2d(obs0_batch),
            'obs1': _array_min2d(obs1_batch),
            'rewards': _array_min2d(reward_batch),
            'actions': _array_min2d(action_batch),
            'terminals1': _array_min2d(terminal1_batch),
        }
        return result

    @property
    def nb_entries(self):
        """return number of available transitions in buffer"""
        return len(self._dataset.obs)
