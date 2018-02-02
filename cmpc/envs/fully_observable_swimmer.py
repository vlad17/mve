"""
This file was made in the style of fully_observable_half_cheetah.py,
mimicking the analogous gym environment.
"""

import numpy as np
import tensorflow as tf

from .fully_observable import FullyObservable
from .render_free_mjc import RenderFreeMJC


class FullyObservableSwimmer(RenderFreeMJC, FullyObservable):
    """A fully-observable version of swimmer."""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('swimmer.xml', 4)

    # gym code
    # def step(self, a):
    #     ctrl_cost_coeff = 0.0001
    #     xposbefore = self.sim.data.qpos[0]
    #     self.do_simulation(a, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0]
    #     reward_fwd = (xposafter - xposbefore) / self.dt
    #     reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
    #     reward = reward_fwd + reward_ctrl
    #     ob = self._get_obs()
    #     return ob, reward, False, dict(reward_fwd=reward_fwd,
    #                                    reward_ctrl=reward_ctrl)

    def _incremental_reward(self, state, action, next_state, reward, sqsum):
        ctrl_cost_coeff = 0.0001
        xposbefore = state[:, 0]
        xposafter = next_state[:, 0]
        new_reward = (xposafter - xposbefore) / self.dt
        new_reward += - ctrl_cost_coeff * sqsum(action)
        reward += new_reward
        return reward

    def np_reward(self, state, action, next_state):
        reward = 0
        reward = self._incremental_reward(state, action, next_state, reward,
                                          lambda x: np.sum(x, axis=1))
        return reward

    def tf_reward(self, state, action, next_state):
        curr_reward = tf.zeros([tf.shape(state)[0]])
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.reduce_sum(x, axis=1))

    def _step(self, action):
        reward, state_after = self._mjc_step(action)
        done = False
        return state_after, reward, done, {}

    # gym code
    # def _get_obs(self):
    #     qpos = self.sim.data.qpos
    #     qvel = self.sim.data.qvel
    #     return np.concatenate([qpos.flat[2:], qvel.flat])

    def _get_obs(self):
        return np.concatenate([
            # difference from gym: need qpos x value for reward
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    # gym code
    # def reset_model(self):
    #     self.set_state(
    #         self.init_qpos + self.np_random.uniform(
    #             low=-.1, high=.1, size=self.model.nq),
    #         self.init_qvel + self.np_random.uniform(
    #             low=-.1, high=.1, size=self.model.nv)
    #     )
    #     return self._get_obs()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(
                low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(
                low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def set_state_from_ob(self, ob):
        self.reset()
        split = self.init_qpos.size
        qpos = ob[:split].reshape(self.init_qpos.shape)
        qvel = ob[split:].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)
