"""
Gym environments might withhold some items from the observation which
make it impossible to compute the reward from the observation alone.

This file contains amended environments with sufficient information
to compute reward from observations (so MPC can use the observations
directly).
"""

from gym.envs.mujoco import MujocoEnv
from gym.utils import EzPickle
import numpy as np
import tensorflow as tf

from utils import inherit_doc


class WithReward:
    """
    A mixin class with a white-box incremental reward function.
    """

    def tf_reward(self, state, action, next_state, curr_reward):
        """
        Given tensors (with the 0th dimension as the batch dimension) for a
        transition during a rollout, and the current reward up to the point in
        time where the transitions passed as arguments have occurred, this
        returns the resulting reward for the trajectories after the transition.
        """
        raise NotImplementedError

@inherit_doc
class WhiteBoxHalfCheetahEasy(MujocoEnv, EzPickle, WithReward):
    """White box HalfCheetah, with frameskip 1 and easy reward"""
    # mostly copied from openai gym

    def __init__(self, frame_skip):  # pylint: disable=super-init-not-called
        MujocoEnv.__init__(self, 'half_cheetah.xml', frame_skip)  # pylint: disable=non-parent-init-called
        EzPickle.__init__(self)  # pylint: disable=non-parent-init-called

    def _incremental_reward(self, state, _, next_state, reward, to_float):
        heading_penalty_factor = 10
        front_leg = state[:, 6]
        reward -= to_float(front_leg >= 0.2) * heading_penalty_factor
        front_shin = state[:, 7]
        reward -= to_float(front_shin >= 0.) * heading_penalty_factor
        front_foot = state[:, 8]
        reward -= to_float(front_foot >= 0.) * heading_penalty_factor
        reward += (next_state[:, 0] - state[:, 0]) / self.dt
        return reward

    def _np_incremental_reward(self, state, action, next_state):
        reward = 0
        reward = self._incremental_reward(state, action, next_state, reward,
                                          lambda x: x.astype(np.float32))
        return reward

    def tf_reward(self, state, action, next_state, curr_reward):
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.cast(x, tf.float32))

    def _step(self, action):
        state_before = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        state_after = self._get_obs()
        reward = self._np_incremental_reward(
            state_before[np.newaxis, ...],
            action[np.newaxis, ...],
            state_after[np.newaxis, ...])
        done = False
        return state_after, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
               self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

@inherit_doc
class WhiteBoxHalfCheetahHard(MujocoEnv, EzPickle, WithReward):
    """White box HalfCheetah, with frameskip 1 and hard reward"""
    # mostly copied from openai gym

    def __init__(self, frame_skip):  # pylint: disable=super-init-not-called
        MujocoEnv.__init__(self, 'half_cheetah.xml', frame_skip)  # pylint: disable=non-parent-init-called
        EzPickle.__init__(self)  # pylint: disable=non-parent-init-called

    def _incremental_reward(self, state, action, next_state, reward, sumaxis1):
        ac_reg = sumaxis1(action * action)
        ac_reg *= 0.1
        reward -= ac_reg
        reward += (next_state[:, 0] - state[:, 0]) / self.unwrapped.dt
        return reward

    def _np_incremental_reward(self, state, action, next_state):
        reward = 0
        reward = self._incremental_reward(state, action, next_state, reward,
                                          lambda x: np.sum(x, axis=1))
        return reward

    def tf_reward(self, state, action, next_state, curr_reward):
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.reduce_sum(x, axis=1))

    def _step(self, action):
        state_before = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        state_after = self._get_obs()
        reward = self._np_incremental_reward(
            state_before[np.newaxis, ...],
            action[np.newaxis, ...],
            state_after[np.newaxis, ...])
        done = False
        return state_after, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
               self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()
