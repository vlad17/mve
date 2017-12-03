"""A learner that acts randomly on the first action, then just returns 0."""

import tensorflow as tf
import numpy as np

from learner import Learner
from learner_flags import LearnerFlags
from utils import get_ac_dim


class ZeroLearnerFlags(LearnerFlags):
    """ZeroLearner flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        # ZeroLearner doesn't have any flags!
        pass

    @staticmethod
    def name():
        return 'zero'

    def __init__(self, _args):
        pass

    def make_learner(self, env):
        return ZeroLearner(env)


class ZeroLearner(Learner):
    """Acts randomly on the first action, then just returns 0."""

    def __init__(self, env):
        self.ac_space = env.action_space
        self.ac_dim = get_ac_dim(env)

    def tf_action(self, states_ns):
        return tf.zeros([tf.shape(states_ns)[0], self.ac_dim])

    def fit(self, data):
        pass

    def act(self, states_ns):
        acs = np.zeros((len(states_ns), self.ac_dim))
        rws = np.zeros(len(states_ns))
        return acs, rws
