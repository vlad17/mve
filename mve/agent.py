"""
The agent provides a simple interface to policies in reinforcement
learning. It is assumed to be stateless, though stateful agents
can probably be added with a reset() method between rollouts.
"""

class Agent:
    """
    An agent exposes two policies for acting in RL environments, an exploratory
    and exploitative one.

    Methods should be vectorized; that is, the zeroth axis of passed-in
    observation tensors should correspond to a batch size, where as-many
    actions are then returned.
    """

    def explore_act(self, states_ns):
        """
        Returns a matrix of n actions each of dimension a following an
        explorative policy.
        """
        raise NotImplementedError

    def exploit_act(self, states_ns):
        """
        Returns a matrix of n actions each of dimension a following an
        exploitative policy.
        """
        raise NotImplementedError

    @staticmethod
    def wrap(explore, exploit):
        """
        Wrap a pair of closures for explore/exploit into an agent object
        """
        return _DummyAgent(explore, exploit)

class _DummyAgent(Agent):
    def __init__(self, explore, exploit):
        super().__init__()
        self._explore = explore
        self._exploit = exploit

    def explore_act(self, states_ns):
        return self._explore(states_ns)

    def exploit_act(self, states_ns):
        return self._exploit(states_ns)
