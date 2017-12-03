"""Base learner class."""

from policy import Policy


class Learner(Policy):
    """
    A learner enables constrained model-predictive control by forcing MPC to
    only plan around trajectories near the learner in some way.

    The learner chooses a policy trajectory based on the history of MPC
    trajectories in some fashion. This policy is used as the center around
    which MPC trajectories are planned. In other words, the learner forms
    a regularizer for MPC planning.

    This policy can also be run directly through the learner's act() method.

    Note that all such policies are deterministic mappings from states
    to actions.
    """

    def tf_action(self, states_ns):
        """
        Return the TF tensor for the action that the learner would take.
        """
        raise NotImplementedError

    def fit(self, data):
        """Fit the learner using the given dataset of transitions"""
        raise NotImplementedError

    def log(self, most_recent_rollouts):
        """
        A learner might log statistics about the most recent
        rollouts (performed by the MPC controller) here.
        """
        pass
