"""
An immutable dataset is a Dataset as in dataset.py, but paths
can't be added to it.

Its main distinction is that it keeps track of individual paths,
as opposed to the Dataset, which views transitions as an amorphous
glob.
"""

import numpy as np

from dataset import Dataset
import reporter


class ImmutableDataset:
    """
    Like a dataset.Dataset, but immutable and with fine-grained access to
    the transitions from a single path.
    """

    def __init__(self, paths):
        tot_transitions = sum(len(path.rewards) for path in paths)
        ac_dim = paths[0].acs.shape[1]
        ob_dim = paths[0].obs.shape[1]
        plan_horizon = paths[0].planned_acs.shape[1]
        dataset = Dataset(
            ac_dim, ob_dim, paths[0].max_horizon, tot_transitions,
            plan_horizon)
        dataset.add_paths(paths)
        self._dataset = dataset

    def _split_array(self, arr):
        terminal_locs = np.flatnonzero(self._dataset.terminals) + 1
        sarr = np.split(arr, terminal_locs)
        assert sarr[-1].size == 0, sarr[-1].size
        return sarr[:-1]

    def log_reward(self):
        """
        Report the reward for the episodes contained here.
        """
        eps_rewards = self._split_array(self._dataset.rewards)
        rewards = [rew.sum() for rew in eps_rewards]
        reporter.add_summary_statistics('reward', rewards)

    def log_reward_bias(self, prediction_horizon):
        """
        Report observed prediction_horizon-step reward bias
        and the h-step reward
        """
        eps_rewards = self._split_array(self._dataset.rewards)
        pred_rewards = self._split_array(self._dataset.predicted_rewards)
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
        bias = h_step_actual - h_step_pred
        if bias.size == 0:
            bias = [0]
        ave_bias = np.mean(bias)
        ave_sqerr = np.square(bias).mean()
        reporter.add_summary('reward bias', ave_bias)
        reporter.add_summary('reward mse', ave_sqerr)

    def episode_obs_planned_obs(self):
        """
        Return a pair of lists, the first for all observations in an episode,
        and the second for all observations expected to come after it
        by the planner.
        """
        obs = self._split_array(self._dataset.obs)
        pobs = self._split_array(self._dataset.planned_obs)
        return obs, pobs

    @property
    def obs(self):
        """Observations across all paths"""
        return self._dataset.obs

    @property
    def planned_acs(self):
        """Planned actions across all paths"""
        return self._dataset.planned_acs

    @property
    def max_horizon(self):
        """Environment maximum horizon"""
        return self._dataset.max_horizon

    @property
    def planned_obs(self):
        """All planned observations."""
        return self._dataset.planned_obs
