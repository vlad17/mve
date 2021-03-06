"""Implementation of a transition dataset (aka replay buffer)"""

import numpy as np

from context import flags
import env_info
from .ringbuffer import RingBuffer
from utils import get_ob_dim, get_ac_dim


class Path(object):
    """
    Store rewards and transitions from a single rollout with a maximum
    horizon length.
    """

    def __init__(self, env, initial_obs, max_horizon):
        self._obs = np.empty((max_horizon, get_ob_dim(env)))
        self._next_obs = np.empty((max_horizon, get_ob_dim(env)))
        self._acs = np.empty((max_horizon, get_ac_dim(env)))
        self._rewards = np.empty(max_horizon)
        self._terminals = np.empty(max_horizon)
        self._idx = 0
        self._obs[0] = initial_obs

    @property
    def max_horizon(self):
        """Maximum path horizon"""
        return len(self._rewards)

    def next(self, next_obs, reward, done, ac):
        """Append a new transition to currently-stored ones"""
        assert self._idx < self.max_horizon, (self._idx, self.max_horizon)
        self._next_obs[self._idx] = next_obs
        self._rewards[self._idx] = reward
        self._acs[self._idx] = ac
        self._idx += 1
        done = done or self._idx == self.max_horizon
        self._terminals[self._idx - 1] = done
        if not done:
            self._obs[self._idx] = next_obs
        return done

    @property
    def obs(self):
        """All observed states so far."""
        return self._obs[:self._idx]

    @property
    def acs(self):
        """All actions so far."""
        return self._acs[:self._idx]

    @property
    def rewards(self):
        """All rewards so far."""
        return self._rewards[:self._idx]

    @property
    def terminals(self):
        """All booleans for whether a state was terminal so far."""
        return self._terminals[:self._idx]

    @property
    def next_obs(self):
        """All states transitioned into so far."""
        return self._next_obs[:self._idx]


class Dataset(object):
    """
    Stores all data for transitions across several rollouts.

    The order of actions, observations, and rewards returned by the
    accessor methods is internally consistent: the i-th action
    in dataset.acs is the one taken in state dataset.obs[i] resulting in
    a new state dataset.next_obs[i], etc.

    Stores up to maxlen transitions.
    """

    def __init__(self, maxlen, mask_probs=None):
        if mask_probs is None:
            mask_probs = np.empty([], dtype='float32')
        else:
            mask_probs = np.asarray(mask_probs, dtype='float32')
        ac_dim, ob_dim = env_info.ac_dim(), env_info.ob_dim()
        self.max_horizon = flags().experiment.horizon
        self.maxlen = maxlen
        self._obs = RingBuffer(maxlen, (ob_dim,))
        self._next_obs = RingBuffer(maxlen, (ob_dim,))
        self._rewards = RingBuffer(maxlen, tuple())
        self._acs = RingBuffer(maxlen, (ac_dim,))
        self._terminals = RingBuffer(maxlen, tuple())
        num_masks = len(mask_probs)
        self._masks = RingBuffer(maxlen, (num_masks,), dtype='bool')
        self._mask_probs = mask_probs

    @staticmethod
    def from_paths(paths):
        """
        Generate a dataset from a collection of trajectories (with size
        equal to the total amount of trajectories given).
        """
        tot_transitions = sum(len(path.rewards) for path in paths)
        dataset = Dataset(tot_transitions)
        dataset.add_paths(paths)
        return dataset

    @property
    def size(self):
        """
        Number of observations. This is at most the maximum buffer length
        specified at initialization.
        """
        return self._obs.length

    @property
    def obs(self):
        """All observed states."""
        return self._obs.all_data()

    @property
    def acs(self):
        """All actions."""
        return self._acs.all_data()

    @property
    def rewards(self):
        """All rewards."""
        return self._rewards.all_data()

    @property
    def terminals(self):
        """All booleans for whether a state was terminal."""
        return self._terminals.all_data()

    @property
    def next_obs(self):
        """All states transitioned into."""
        return self._next_obs.all_data()

    @property
    def masks(self):
        """All binary sampled masks (it's up to the user to keep track of
        the one they'd like to use; the i-th mask was sampled with the i-th
        probability passed in during dataset initialization)."""
        return self._masks.all_data()

    def add_paths(self, paths):
        """Aggregate data from a list of paths"""
        for path in paths:
            assert path.max_horizon == self.max_horizon, \
                (path.max_horizon, self.max_horizon)
            self._obs.append_all(path.obs)
            self._next_obs.append_all(path.next_obs)
            self._rewards.append_all(path.rewards)
            self._acs.append_all(path.acs)
            self._terminals.append_all(path.terminals)
            n = len(path.obs)
            masks = np.zeros((n, len(self._mask_probs)), dtype='bool')
            for i, p in enumerate(self._mask_probs):
                masks[:, i] = np.random.choice(
                    [True, False], size=n, p=[p, 1 - p])
            self._masks.append_all(masks)

    def get_state(self):
        """return all state in a dictionary mapping to numpy arrays"""
        return {
            'obs': self.obs,
            'next_obs': self.next_obs,
            'rewards': self.rewards,
            'acs': self.acs,
            'terminals': self.terminals}

    def set_state(self, dictionary):
        """
        Set the internal ringbuffers to contain the values in the dictionary
        (which should be keyed to the corresponding property).
        """
        for attr, arr in dictionary.items():
            rb = getattr(self, '_' + attr)
            rb = RingBuffer(rb.maxlen, rb.data.shape[1:], rb.data.dtype)
            setattr(self, '_' + attr, rb)
            rb.append_all(arr)

    def batches_per_epoch(self, batch_size):
        """
        Given a fixed batch size, returns the number of batches necessary
        to do a full pass over all data in this replay buffer.
        """
        return max(self.size // batch_size, 1)

    def next(self, ob, next_ob, reward, done, action):
        """Update RingBuffers directly using results of a step."""
        self._obs.append(ob)
        self._next_obs.append(next_ob)
        self._rewards.append(reward)
        self._acs.append(action)
        self._terminals.append(done)
        self._masks.append(
            np.random.rand(len(self._mask_probs)) < self._mask_probs)

    def sample_many(self, nbatches, batch_size):
        """
        Generates nbatches worth of batches, iid - sampled from the internal
        buffers.

        Values returned are transitions, which are tuples of:
        (current) observations, next observations, rewards, actions,
        and terminal state indicators.
        """
        assert batch_size > 0, batch_size
        if self.size == 0 or nbatches == 0:
            return
        batch_idxs = np.random.randint(self.size, size=(
            nbatches, batch_size))
        transitions = [self.obs, self.next_obs, self.rewards, self.acs,
                       self.terminals, self.masks]
        for batch_idx in batch_idxs:
            yield [transition_item[batch_idx] for transition_item
                   in transitions]
