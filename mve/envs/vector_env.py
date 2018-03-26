"""
A vectorized environment can be viewed formally as a Cartesian product of
identical MDPs. It lets us evaluate several trajectories in parallel.
"""

import os
import numpy as np

import gym


class VectorEnv(gym.Env):
    """
    Abstract base class for vectorized environments. These environments
    are structured identically to gym environments but they omit the
    following information:

    * The "info" dictionary, which is the last return value in the tuple
      from step() will be empty.
    * Rendering is unsupported.
    * Environment wrappers are not supported (the environments being
      vectorized should be wrapped instead).

    However, vectorized envs offer a couple additional useful pieces
    of functionality.

    * set_state_from_obs - does what the name says, as long as the
      underlying env has a set_state_from_ob method (this presumes a
      fully observable MDP).
    * multi_step - given an up-front open loop action plan, execute it
      and return the resulting states, rewards, and done indicators.

    Since some environments may terminate early while others are
    iterating, the caller is still responsible for tracking which environment
    indices are active, since there is no guarantee on the returned
    values for environments that are done but are still stepped.

    Moreover, vectorized environments handle being passed in arguments
    with smaller rank gracefully. E.g., if a vectorized environment has
    rank n (n simultaneous vectorized environments), it will accept
    actions of length m <= n and only return the corresponding
    actions for the first m environments.
    """

    def set_state_from_ob(self, obs):
        """
        Set the state for each environment with the given observations.
        Requires underlying environments have the set_state_from_ob method.
        """
        raise NotImplementedError

    def _seed(self, seed=None):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError

    def _seed_uncorr(self, n):
        rand_bytes = os.urandom(n * 4)
        rand_uints = np.frombuffer(rand_bytes, dtype=np.uint32)
        self.seed([int(x) for x in rand_uints])

    def multi_step(self, acs_hna):
        """
        Evaluate a set of open-loop actions. Returns a tuple of the resulting
        states, rewards, and done indicators (no info). The actions should be
        a tensor with axes in the following order:
        0. length of horizon to multi-step
        1. number of environments to vectorize over
        2. the action shape (or observation shape)
        The returned states, rewards, and done indicators follow the above
        specification, but rewards and dones have a null shape so axis 2 is
        not present.
        """
        raise NotImplementedError
