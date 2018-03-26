"""
This module defines the learner interface.
"""


class Learner:
    """
    A learner ingests data, manipulates internal state by learning off this
    data, supplies agents for taking further actions, and provides an opaque
    interface to internal evaluation.
    """

    def train(self, data, timesteps):
        """
        Fit the learner using the given dataset of transitions.

        data is the current ringbuffer and timesteps indicates how many
        new transitions we have just sampled.
        """
        raise NotImplementedError

    def agent(self):
        """
        Return an agent.Agent to take some actions.
        """
        raise NotImplementedError

    def tf_action(self, states_ns):
        """
        Accept a TF state and return a TF tensor for the action this learner
        would have taken. This is used to evaluate the dynamics quality,
        so take the action that is used to expand dynamics if the learner uses
        a dynamics model for planning or learning.
        """
        raise NotImplementedError

    def evaluate(self, data):
        """
        Report evaluation information.
        """
        raise NotImplementedError
