"""
This module defines the one-stop normalization shop for all observed transition
related normalization statistics.

Henceforth, normalization refers to mean-centering and std-rescaling ops while
scaling refers to converting a value in a certain range into [-1, 1]
"""

import tensorflow as tf
import numpy as np

from .assignable_statistic import AssignableStatistic
from context import flags
import env_info
from flags import Flags, ArgSpec
import reporter
from tfnode import TFNode


class NormalizationFlags(Flags):
    """Flags having to do with normalization of observed transitions."""

    @staticmethod
    def _generate_arguments():
        yield ArgSpec(
            name='restore_normalization',
            default=None,
            type=str,
            help='restore normalization from the given path')

    def __init__(self):
        super().__init__('normalization', 'normalization',
                         list(NormalizationFlags._generate_arguments()))


class Normalizer(TFNode):
    """
    A normalizer handles the normalization of three items:

    * observations
    * actions
    * deltas between observations

    The use of "deltas" implicits a notion of continuity between timesteps,
    so MDPs where this change isn't coherent may not work well.
    """

    def __init__(self):
        stats = _Statistics(None)
        with tf.variable_scope('normalization'):
            self._ob_tf_stats = AssignableStatistic(
                'ob', stats.mean_ob, stats.std_ob)
            self._ac_tf_stats = AssignableStatistic(
                'ac', stats.mean_ac, stats.std_ac)
            self._delta_tf_stats = AssignableStatistic(
                'delta', stats.mean_delta, stats.std_delta)
        super().__init__(
            'normalization', flags().normalization.restore_normalization)

    def norm_obs(self, obs):
        """normalize observations"""
        return self._ob_tf_stats.tf_normalize(obs)

    def norm_acs(self, acs):
        """normalize actions"""
        return self._ac_tf_stats.tf_normalize(acs)

    def denorm_acs(self, acs):
        """denormalize actions"""
        return self._ac_tf_stats.tf_denormalize(acs)

    def norm_delta(self, deltas):
        """normalize deltas"""
        return self._delta_tf_stats.tf_normalize(deltas)

    def denorm_delta(self, deltas):
        """denormalize deltas"""
        return self._delta_tf_stats.tf_denormalize(deltas)

    def update_stats(self, data):
        """update the stateful normalization statistics"""
        stats = _Statistics(data)
        self._ob_tf_stats.update_statistics(stats.mean_ob, stats.std_ob)
        self._ac_tf_stats.update_statistics(stats.mean_ac, stats.std_ac)
        self._delta_tf_stats.update_statistics(
            stats.mean_delta, stats.std_delta)

    def log_stats(self):
        """report normalization statistics"""
        stats = [
            (self._ob_tf_stats, 'observations'),
            (self._delta_tf_stats, 'deltas')]

        prefix = 'dynamics statistics/'
        for stat, name in stats:
            reporter.add_summary_statistics(
                prefix + name + '/mean magnitude',
                np.absolute(stat.mean()),
                hide=True)
            reporter.add_summary_statistics(
                prefix + name + '/std',
                stat.std(),
                hide=True)


class DummyNormalizer:
    """
    Adheres to the normalizer interface but does no actual normalization.
    """

    @staticmethod
    def norm_obs(obs):
        """normalize observations"""
        return obs

    @staticmethod
    def norm_acs(acs):
        """normalize actions"""
        return acs

    @staticmethod
    def denorm_acs(acs):
        """denormalize actions"""
        return acs

    @staticmethod
    def norm_delta(deltas):
        """normalize deltas"""
        return deltas

    @staticmethod
    def denorm_delta(deltas):
        """denormalize deltas"""
        return deltas

    @staticmethod
    def update_stats(data):
        """update the stateful normalization statistics"""
        pass

    @staticmethod
    def log_stats():
        """report normalization statistics"""
        pass


class _Statistics:
    def __init__(self, data):
        if not data:
            self.mean_ob = np.zeros(env_info.ob_dim())
            self.std_ob = np.ones(env_info.ob_dim())
            self.mean_delta = np.zeros(env_info.ob_dim())
            self.std_delta = np.ones(env_info.ob_dim())
            self.mean_ac = np.zeros(env_info.ac_dim())
            self.std_ac = np.ones(env_info.ac_dim())
        else:
            self.mean_ob = np.mean(data.obs, axis=0)
            self.std_ob = np.std(data.obs, axis=0)
            diffs = data.next_obs - data.obs
            self.mean_delta = np.mean(diffs, axis=0)
            self.std_delta = np.std(diffs, axis=0)
            self.mean_ac = np.mean(data.acs, axis=0)
            self.std_ac = np.std(data.acs, axis=0)


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
