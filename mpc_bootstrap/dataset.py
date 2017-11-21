"""Implementation of a transition dataset (aka replay buffer)"""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import numpy as np

from ddpg.memory import RingBuffer
from utils import get_ob_dim, get_ac_dim
from heap import MaxHeap, SumTree


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
        self._predicted_rewards = np.empty(max_horizon)
        self._terminals = np.empty(max_horizon)
        self._idx = 0
        self._obs[0] = initial_obs

    @property
    def max_horizon(self):
        """Maximum path horizon"""
        return len(self._predicted_rewards)

    def next(self, next_obs, reward, done, ac, pred_reward):
        """Append a new transition to currently-stored ones"""
        assert self._idx < self.max_horizon, (self._idx, self.max_horizon)
        self._next_obs[self._idx] = next_obs
        self._rewards[self._idx] = reward
        self._acs[self._idx] = ac
        self._predicted_rewards[self._idx] = pred_reward
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
    def predicted_rewards(self):
        """All predicted rewards so far."""
        return self._predicted_rewards[:self._idx]


class Dataset(object):
    """
    Stores all data for transitions across several rollouts.

    The order of actions, observations, and rewards returned by the
    accessor methods is internally consistent: the i-th action
    in dataset.acs is the one taken in state dataset.obs[i] resulting in
    a new state dataset.next_obs[i], etc.

    Stores up to maxlen transitions.
    """

    def __init__(self, ac_dim, ob_dim, max_horizon, maxlen):
        self.max_horizon = max_horizon
        self._obs = RingBuffer(maxlen, (ob_dim,))
        self._next_obs = RingBuffer(maxlen, (ob_dim,))
        self._rewards = RingBuffer(maxlen, tuple())
        self._acs = RingBuffer(maxlen, (ac_dim,))
        self._terminals = RingBuffer(maxlen, tuple())
        self._predicted_rewards = RingBuffer(maxlen, tuple())

    @staticmethod
    def from_env(env, max_horizon, maxlen):
        """
        Generate a dataset with action/observation as specified by an
        environment.
        """
        return Dataset(get_ac_dim(env), get_ob_dim(env), max_horizon, maxlen)

    @property
    def size(self):
        """
        Number of observations. This is at most the maximum buffer length
        specified at initialization.
        """
        return self._obs.length

    @property
    def obs(self):
        """All observed states so far."""
        return self._obs.all_data()

    @property
    def acs(self):
        """All actions so far."""
        return self._acs.all_data()

    @property
    def rewards(self):
        """All rewards so far."""
        return self._rewards.all_data()

    @property
    def terminals(self):
        """All booleans for whether a state was terminal so far."""
        return self._terminals.all_data()

    @property
    def next_obs(self):
        """All states transitioned into so far."""
        return self._next_obs.all_data()

    @property
    def predicted_rewards(self):
        """All predicted rewards so far."""
        return self._predicted_rewards.all_data()

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
            self._predicted_rewards.append_all(path.predicted_rewards)

    def _check_not_full(self):
        # TODO: this requirement is necessary for this implementation
        # but it's not actually that critical in general: if an episode
        # got chopped off by the ring buffer we could just drop it
        # in _split_array. But that'd require an unnecessarily complex
        # implementation, since we only ever call _split_array
        # on small buffers.
        assert self._rewards.length < self._rewards.maxlen, \
            'dataset buffer may have already dropped data'

    def _split_array(self, arr):
        self._check_not_full()
        terminal_locs = np.flatnonzero(self.terminals) + 1
        sarr = np.split(arr, terminal_locs)
        assert sarr[-1].size == 0, sarr[-1].size
        return sarr[:-1]

    def reward_bias(self, prediction_horizon):
        """
        Report observed prediction_horizon-step reward bias
        and the h-step reward
        """
        eps_rewards = self._split_array(self.rewards)
        pred_rewards = self._split_array(self.predicted_rewards)
        agg_rewards = [np.cumsum(ep_rewards) for ep_rewards in eps_rewards]
        h_step_rews = []
        for ep_agg_rew in agg_rewards:
            h_step_rew = ep_agg_rew[prediction_horizon - 1:]
            h_step_rew[1:] -= ep_agg_rew[:-prediction_horizon]
            h_step_rews.append(h_step_rew)
        if prediction_horizon > 1:
            pred_rewards = [ep_predictions[:-(prediction_horizon - 1)]
                            for ep_predictions in pred_rewards]
        h_step_actual = np.concatenate(h_step_rews)
        h_step_pred = np.concatenate(pred_rewards)
        return h_step_actual - h_step_pred, h_step_actual

    def per_episode_rewards(self):
        """Return a list with the total reward for each seen episode"""
        all_rewards = self._split_array(self.rewards)
        return [r.sum() for r in all_rewards]

    def batches_per_epoch(self, batch_size):
        """
        Given a fixed batch size, returns the number of batches necessary
        to do a full pass over all data in this replay buffer.
        """
        return max(self.size // batch_size, 1)

    def sample_many(self, nbatches, batch_size):
        """
        Generates nbatches worth of batches, iid-sampled from the internal
        buffers.

        Values returned are transitions, which are tuples of:
        (current) observations, next observations, rewards, actions,
        and terminal state indicators.
        """
        batch_idxs = np.random.randint(self.size, size=(
            nbatches, batch_size))
        transitions = [self.obs, self.next_obs, self.rewards, self.acs,
                       self.terminals]
        for batch_idx in batch_idxs:
            yield [transition_item[batch_idx] for transition_item
                   in transitions]


def one_shot_dataset(paths):
    """Create a Dataset just big enough for the parameter paths"""
    # add an extra slot so that Dataset._check_not_full isn't worried about
    # using a full buffer, which it conservatively interprets as a buffer
    # where part of an episode may have been dropped.
    tot_transitions = sum(len(path.rewards) for path in paths) + 1
    ac_dim = paths[0].acs.shape[1]
    ob_dim = paths[0].obs.shape[1]
    dataset = Dataset(ac_dim, ob_dim, paths[0].max_horizon, tot_transitions)
    dataset.add_paths(paths)
    return dataset


class PrioritizedDataset(Dataset):
    """
    Offers all of the functionality of the base class and
    allows for additional prioritization in optimization.
    """

    def __init__(self, ac_dim, ob_dim, max_horizon, maxlen):
        super().__init__(ac_dim, ob_dim, max_horizon, maxlen)
        self._batch_idx = None
        self._weight_heap = MaxHeap(maxlen)
        self._sum_tree = SumTree(maxlen)
        self._buffer_idx = 0

    def prioritized_sample_many(self, nbatches, batch_size):
        """
        A generator analogous to sample_many, but this adds an
        importance weight per Schaul et al 2015 to the end of the
        batch tuple.

        See https://arxiv.org/abs/1511.05952 for algorithm details.

        If you use this method in a for-loop, you should co-operate
        by calling prioritized_dataset.update_errs(...) with
        the errors that each item in the batch generated for the
        most recent batch. This lets the dataset updates its
        importance weights. E.g.:

        generator = data.prioritized_sample_many(...)
        for batch in generator:
            objective_errors = train(batch)
            data.update_errs(objective_errors)
        """
        batch_idxs = self._sum_tree.sample(
            nbatches * batch_size).reshape(nbatches, batch_size)
        transitions = [self.obs, self.next_obs, self.rewards, self.acs,
                       self.terminals]
        for batch_idx in batch_idxs:
            self._batch_idx = batch_idx
            transition = [transition_item[batch_idx] for transition_item
                          in transitions]
            weights = self._weight_heap.get_values(batch_idx)
            max_weight = self._weight_heap.max()
            assert max_weight > 0, max_weight
            transition.append(weights / max_weight)
            yield transition

        self._batch_idx = None

    def add_paths(self, paths):
        super().add_paths(paths)
        # _buffer_idx tracks where in each replay buffer (stored in the
        # base class) that we are. Now that we just added the new
        # paths we should update them.
        num_new_transitions = sum(len(path.obs) for path in paths)
        maxlen = self._sum_tree.size
        prio_max = self._sum_tree.max()
        if prio_max == 0:
            prio_max = 1  # first add_paths we need to start w/ nonzero prio

        while num_new_transitions > 0:
            buffer_add_len = min(
                maxlen - self._buffer_idx, num_new_transitions)
            idxs = np.arange(buffer_add_len, dtype=int)
            idxs += self._buffer_idx
            vals = np.full(buffer_add_len, prio_max)
            self._update_priorities(idxs, vals)
            num_new_transitions -= buffer_add_len
            self._buffer_idx = (self._buffer_idx + buffer_add_len) % maxlen

    def _update_priorities(self, idxs, prios):
        self._sum_tree.update_priorities(idxs, prios)
        weights = np.sqrt(  # beta = 0.5 from the paper
            self._weight_heap.size *
            self._sum_tree.probabilities(idxs)) + 1e-6
        assert np.all(weights) > 0, weights[weights <= 0]
        weights = 1 / weights
        self._weight_heap.modify_values(idxs, weights)

    def update_errs(self, errs):
        """
        This must be called from within a for-loop iterating over the
        generator created by self.prioritized_sample_many.

        This updates the corresponding importance weights for the
        batch that was sent (and received the parameter absolute errors).
        """
        errs = np.squeeze(errs)
        blended_errs = np.sqrt(errs)  # alpha = 0.5 from the paper
        self._update_priorities(self._batch_idx, blended_errs)

    @staticmethod
    def from_env(env, max_horizon, maxlen):
        """
        Generate a dataset with action/observation as specified by an
        environment.
        """
        return PrioritizedDataset(
            get_ac_dim(env), get_ob_dim(env), max_horizon, maxlen)
