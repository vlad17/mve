"""
Base learner class for generating learned sampling distributions for
the random shooter MPC.
"""

from controller import Controller


class Learner:
    """
    A learner enables constrained model-predictive control by forcing MPC to
    only plan around trajectories near the learner in some way.

    The learner chooses a policy trajectory based on the history of MPC
    trajectories in some fashion. This policy is used as the center around
    which MPC trajectories are planned. In other words, the learner forms
    a regularizer for MPC planning.

    This policy can also be run directly through the learner's act() method.

    Each Learner subclass should also have a FLAGS class attribute specifying
    its parameters. Each learner should be constructible with the arguments
    (env, flags).
    """

    def tf_action(self, states_ns):
        """
        Return the TF tensor for the action that the learner would take.
        """
        raise NotImplementedError

    def fit(self, data):
        """Fit the learner using the given dataset of transitions"""
        raise NotImplementedError

    def act(self, states_ns):
        """Return the actions to play in the given states."""
        raise NotImplementedError


def as_controller(learner):
    """
    Generates a controller interface to a learner.
    """
    return _LearnerAsController(learner)


class _LearnerAsController(Controller):
    def __init__(self, learner):
        self._learner = learner

    def act(self, states_ns):
        return self._learner.act(states_ns), None, None, None

    def planning_horizon(self):
        return 0
