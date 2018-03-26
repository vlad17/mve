"""
Fully observable environments are environments where the state space fully
captures the information necessary for the next transition and reward
distributions. As such, they reveal additional information: they have
a TensorFlow tensor-to-tensor mapping for the reward function, and, for
diagnostic purposes, allow state to be explicitly set.

Currently only deterministic rewards are supported.
"""

import numpy as np

class FullyObservable:
    """
    A mixin class with a white-box incremental reward function.

    In addition, such an environment is "settable": given an observation
    this environment can reset to that observation's state for
    debugging purposes. Thus this implicitly requires a fully observable
    MDP.
    """

    def tf_reward(self, state, action, next_state):
        """
        Given tensors(with the 0th dimension as the batch dimension) for a
        transition during a rollout, this returns the corresponding reward
        as a rank - 1 tensor(vector).
        """
        raise NotImplementedError

    def np_reward(self, state, action, next_state):
        """
        Numpy analogoue for tf_reward.
        """
        raise NotImplementedError

    def set_state_from_ob(self, ob):
        """
        Reset the environment, starting at the state corresponding to the
        observation ob.
        """
        raise NotImplementedError

    def _mjc_step(self, action):
        # common mujoco stepping functionality
        state_before = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        state_after = self._get_obs()
        reward = self.np_reward(
            state_before[np.newaxis, ...],
            action[np.newaxis, ...],
            state_after[np.newaxis, ...])
        return reward, state_after
