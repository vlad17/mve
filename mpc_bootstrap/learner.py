"""Base learner class."""

from policy import Policy


class Learner(Policy):  # pylint: disable=abstract-method
    """
    A learner acts in a manner that is most conducive to its own learning,
    as long as the resulting states are labelled with correct actions to have
    taken by an expert. Given such a labelled dataset, it can also learn from
    it.
    """

    def tf_action(self, states_ns, is_initial=False):
        """
        Return the TF tensor for the action that the learner would take.
        The learner may choose to take different actions depending on whether
        is_initial is true or not, which when set indicates that this is
        the first action in a simulated rollout.
        """
        raise NotImplementedError

    def fit(self, data):
        """Fit the learner using the given dataset of transitions"""
        raise NotImplementedError

    def log(self, most_recent_rollouts):
        """
        A learner might log statistics about the most recent
        rollouts here.
        """
        pass
