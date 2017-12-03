"""Base policy class."""


class Policy(object):
    """
    A Policy is an interfaces that represents a possibly stateful agent,
    which offers a mapping from states to actions.

    This policy is naturally vectorizable. If the agent is making decisions
    for n instances of an environment at the same time, then the agent may
    maintain internal decision-making state for each instance by assuming
    that the instance corresponds to a fixed index [0, ..., n).

    To refresh the state across all indices, and indicate what n is,
    the reset() method should be called.

    """

    def act(self, states_ns):
        """
        This method returns a tuple of two items.

        The first is the action for every state in states_ns, where the batch
        size is n and the state shape is s.

        The second is some estimate of predicted reward (which may be all zeros
        if the agent does not have this information).
        """
        raise NotImplementedError

    def reset(self, n):
        """
        Indicate that the agent should reset to n fresh internal states,
        to be used by subsequent invocations of act().

        A stateless agent may do nothing here.
        """
        pass