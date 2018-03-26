"""A policy which acts uniformly at random."""

import numpy as np

from agent import Agent


class RandomPolicy(Agent):
    """A policy that acts uniformly randomly in the action space."""

    def __init__(self, env):
        self.ac_space = env.action_space

    def exploit_act(self, states_ns):
        nstates = len(states_ns)
        return self._sample_n(nstates)

    def explore_act(self, states_ns):
        nstates = len(states_ns)
        return self._sample_n(nstates)

    def _sample_n(self, n):
        return np.random.uniform(
            low=self.ac_space.low,
            high=self.ac_space.high,
            size=(n,) + self.ac_space.shape)
