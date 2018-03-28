"""
Record stateful order statistics with AssignableStatistic.
"""

import numpy as np
import tensorflow as tf


class AssignableStatistic:
    """
    Need stateful statistics so we don't have to feed
    stats through a feed_dict every time we want to use the
    dynamics.

    This class keeps track of the mean and std as given.

    This class also does some funny business when the standard
    deviation it's given ever falls below epsilon: at this point
    standard deviation is no longer used in normalization.
    """

    def __init__(self, suffix, initial_mean, initial_std, epsilon=1e-4):
        initial_mean = np.asarray(initial_mean).astype('float32')
        initial_std = np.asarray(initial_std).copy().astype('float32')
        self._avoid_std_normalization = np.zeros(initial_std.shape, dtype=bool)
        self._epsilon = epsilon
        below_eps = initial_std < epsilon
        initial_std[np.where(below_eps)] = 1.0
        self._avoid_std_normalization[np.where(below_eps)] = True
        self._mean_var = tf.get_variable(
            name='mean_' + suffix, trainable=False,
            initializer=initial_mean)
        self._std_var = tf.get_variable(
            name='std_' + suffix, trainable=False,
            initializer=initial_std)
        self._mean_ph = tf.placeholder(tf.float32, initial_mean.shape)
        self._std_ph = tf.placeholder(tf.float32, initial_std.shape)
        self._assign_both = tf.group(
            tf.assign(self._mean_var, self._mean_ph),
            tf.assign(self._std_var, self._std_ph))

    def update_statistics(self, mean, std):
        """
        Update the stateful statistics using the default session.
        """
        std = np.copy(std)
        below_eps = std < self._epsilon
        self._avoid_std_normalization[np.where(below_eps)] = True
        std[np.where(self._avoid_std_normalization)] = 1.0
        tf.get_default_session().run(
            self._assign_both, feed_dict={
                self._mean_ph: mean,
                self._std_ph: std})

    def mean(self):
        """Returns current mean"""
        return tf.get_default_session().run(self._mean_var)

    def std(self):
        """Returns current std"""
        return tf.get_default_session().run(self._std_var)

    def tf_normalize(self, x):
        """Normalize a value according to these statistics"""
        return (x - self._mean_var) / (self._std_var)

    def tf_denormalize(self, x):
        """Denormalize a value according to these statistics"""
        return x * self._std_var + self._mean_var
