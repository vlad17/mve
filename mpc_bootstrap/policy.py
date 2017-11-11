"""Base policy class."""


class Policy(object):
    """A Policy represents a possibly stateful agent."""

    def act(self, states_ns):
        """
        This method returns a tuple of two items.

        The first is the action for every state in states_ns, where the batch
        size is n and the state shape is s.

        The second is a predicted reward or zeros, depending on whether the
        policy is based on MPC.
        """
        raise NotImplementedError

    def reset(self, nstates):
        """
        For stateful RNN-based policies, this should be called at the
        beginning of each rollout to reset the state.
        """
        pass

    def log(self, **kwargs):
        """A policy might log additional stats here."""
        pass
