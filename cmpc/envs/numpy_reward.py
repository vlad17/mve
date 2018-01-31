"""
For one-off reward calculations, this class provides a convenient wrapper
around the TensorFlow reward functions provided by the environments.
"""

import tensorflow as tf

from utils import get_ob_dim, get_ac_dim


class NumpyReward:
    """
    Wraps an environment to provide a numpy-based reward function instead
    of a TensorFlow non-eager one.
    """

    def __init__(self, env):
        self._state_ph_ns = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)])
        self._action_ph_na = tf.placeholder(
            tf.float32, [None, get_ac_dim(env)])
        self._next_state_ph_ns = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)])
        self._reward_n = env.tf_reward(
            self._state_ph_ns,
            self._action_ph_na,
            self._next_state_ph_ns)

    def np_reward(self, state_ns, action_na, next_state_ns):
        """
        Returns the reward for the given vector of transitions.
        """
        return tf.get_default_session().run(self._reward_n, feed_dict={
            self._state_ph_ns: state_ns,
            self._action_ph_na: action_na,
            self._next_state_ph_ns: next_state_ns})
