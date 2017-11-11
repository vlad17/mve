"""A learner that acts randomly on the first action, then just returns 0."""

import tensorflow as tf
import numpy as np

from learner import Learner
from utils import (create_random_tf_policy, get_ac_dim)

class ZeroLearner(Learner):
    """Acts randomly on the first action, then just returns 0."""

    def __init__(self, env):
        self.ac_space = env.action_space
        self.ac_dim = get_ac_dim(env)

    def tf_action(self, states_ns, is_initial=True):
        if is_initial:
            random_policy = create_random_tf_policy(self.ac_space)
            return random_policy(states_ns)
        return tf.zeros([tf.shape(states_ns)[0], self.ac_dim])

    def fit(self, data, **kwargs):
        return

    def act(self, states_ns):
        acs = np.zeros((len(states_ns), self.ac_dim))
        rws = np.zeros(len(states_ns))
        return acs, rws
