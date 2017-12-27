"""A learner that acts randomly on the first action, then just returns 0."""

import tensorflow as tf
import numpy as np

import env_info
from learner import Learner


class ZeroLearner(Learner):
    """Acts randomly on the first action, then just returns 0."""

    def __init__(self):
        pass

    def tf_action(self, states_ns):
        return tf.zeros([tf.shape(states_ns)[0], env_info.ac_dim()])

    def fit(self, data, timesteps):
        pass

    def act(self, states_ns):
        acs = np.zeros((len(states_ns), env_info.ac_dim()))
        return acs
