"""A policy which acts uniformly at random."""

import numpy as np

from controller import Controller


# TODO(mwhittaker): Make a Policy instead of a Controller.
class RandomPolicy(Controller):
    """A policy that acts uniformly randomly in the action space."""

    def __init__(self, env):
        self.ac_space = env.action_space

    def act(self, states_ns):
        nstates = len(states_ns)
        return self._sample_n(nstates), np.zeros(nstates)

    def _sample_n(self, n):
        return np.random.uniform(
            low=self.ac_space.low,
            high=self.ac_space.high,
            size=(n,) + self.ac_space.shape)
