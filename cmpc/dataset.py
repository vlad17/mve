"""Implementation of a transition dataset (aka replay buffer)"""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import numpy as np

from ddpg.memory import RingBuffer
from utils import get_ob_dim, get_ac_dim


class Path(object):
    """
    Store rewards and transitions from a single rollout with a maximum
    horizon length.

    May also store planning information.
    """

    def __init__(self, env, initial_obs, max_horizon, planning_horizon):
        self._obs = np.empty((max_horizon, get_ob_dim(env)))
        self._next_obs = np.empty((max_horizon, get_ob_dim(env)))
        self._acs = np.empty((max_horizon, get_ac_dim(env)))
        self._rewards = np.empty(max_horizon)
        self._planned_acs = np.empty(
            (max_horizon, planning_horizon, get_ac_dim(env)))
        self._planned_obs = np.empty(
            (max_horizon, planning_horizon, get_ob_dim(env)))
        self._terminals = np.empty(max_horizon)
        self._idx = 0
        self._obs[0] = initial_obs

    @property
    def max_horizon(self):
        """Maximum path horizon"""
        return len(self._rewards)

    def next(self, next_obs, reward, done, ac, planned_acs, planned_obs):
        """Append a new transition to currently-stored ones"""
        assert self._idx < self.max_horizon, (self._idx, self.max_horizon)
        self._next_obs[self._idx] = next_obs
        self._rewards[self._idx] = reward
        self._acs[self._idx] = ac
        if planned_acs is not None:
            self._planned_acs[self._idx] = planned_acs
        else:
            assert self._planned_acs.size == 0
            assert self._planned_obs.size == 0
        if planned_obs is not None:
            self._planned_obs[self._idx] = planned_obs
        else:
            assert self._planned_obs.size == 0
            assert self._planned_acs.size == 0
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

    @property
    def planned_acs(self):
        """All planned actions so far."""
        return self._planned_acs[:self._idx]

    @property
    def planned_obs(self):
        """All planned observations so far."""
        return self._planned_obs[:self._idx]


class Dataset(object):
    """
    Stores all data for transitions across several rollouts.

    The order of actions, observations, and rewards returned by the
    accessor methods is internally consistent: the i-th action
    in dataset.acs is the one taken in state dataset.obs[i] resulting in
    a new state dataset.next_obs[i], etc.

    Stores up to maxlen transitions.

    Note by default a dataset DOES NOT store plans. Change planning_horizon > 0
    if you want it to do so.
    """

    def __init__(self, ac_dim, ob_dim, max_horizon, maxlen,
                 planning_horizon=0):
        self.max_horizon = max_horizon
        self.maxlen = maxlen
        self._obs = RingBuffer(maxlen, (ob_dim,))
        self._next_obs = RingBuffer(maxlen, (ob_dim,))
        self._rewards = RingBuffer(maxlen, tuple())
        self._acs = RingBuffer(maxlen, (ac_dim,))
        self._terminals = RingBuffer(maxlen, tuple())
        self._planned_acs = RingBuffer(maxlen, (planning_horizon, ac_dim))
        self._planned_obs = RingBuffer(maxlen, (planning_horizon, ob_dim))

    @staticmethod
    def from_env(env, max_horizon, maxlen):
        """
        Generate a dataset with action/observation as specified by an
        environment.
        """
        return Dataset(get_ac_dim(env), get_ob_dim(env), max_horizon, maxlen)

    @staticmethod
    def from_paths(paths):
        """
        Generate a dataset from a collection of trajectories (with size
        equal to the total amount of trajectories given).
        """
        tot_transitions = sum(len(path.rewards) for path in paths)
        ac_dim = paths[0].acs.shape[1]
        ob_dim = paths[0].obs.shape[1]
        plan_horizon = paths[0].planned_acs.shape[1]
        dataset = Dataset(
            ac_dim, ob_dim, paths[0].max_horizon, tot_transitions,
            plan_horizon)
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
    def planned_acs(self):
        """All planned actions."""
        return self._planned_acs.all_data()

    @property
    def planned_obs(self):
        """All planned observations."""
        return self._planned_obs.all_data()

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
            plan_horizon = self.planned_acs.shape[1]
            self._planned_acs.append_all(path.planned_acs[:, :plan_horizon, :])
            self._planned_obs.append_all(path.planned_obs[:, :plan_horizon, :])

    def set_state(self, dictionary):
        """
        Set the internal ringbuffers to contain the values in the dictionary
        (which should be keyed to the corresponding property).
        """
        for attr, arr in dictionary.items():
            # technically, we should wipe what the ringbuffer currently
            # contains to "really" set the state. We don't have to do this
            # if the ringbuffer is currently empty, which we assume
            # is the case always
            rb = getattr(self, '_' + attr)
            assert rb.length == 0, len(rb)
            rb.append_all(arr)

    def batches_per_epoch(self, batch_size):
        """
        Given a fixed batch size, returns the number of batches necessary
        to do a full pass over all data in this replay buffer.
        """
        return max(self.size // batch_size, 1)

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
                       self.terminals]
        for batch_idx in batch_idxs:
            yield [transition_item[batch_idx] for transition_item
                   in transitions]
