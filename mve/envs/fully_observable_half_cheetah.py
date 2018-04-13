"""
The usual gym environments might withhold some items from the observation which
make it impossible to compute the reward from the observation alone.

This file contains amended environments with sufficient information
to compute reward from observations to make the settings a proper
fully-observable MDPs. Besides adding the additional dimensions to the
observations space, the environments are equivalent to the OpenAI gym
versions at commit 4c460ba6c8959dd8e0a03b13a1ca817da6d4074f.
"""

import numpy as np
import tensorflow as tf

from .fully_observable import FullyObservable
from .render_free_mjc import RenderFreeMJC


class FullyObservableHalfCheetah(RenderFreeMJC, FullyObservable):
    """A fully-observable version of half cheetah."""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('half_cheetah.xml', 5)

    # gym code
    # def _step(self, action):
    #     xposbefore = self.sim.data.qpos[0, 0]
    #     self.do_simulation(action, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0, 0]
    #     ob = self._get_obs()
    #     reward_ctrl = - 0.1 * np.square(action).sum()
    #     reward_run = (xposafter - xposbefore)/self.dt
    #     reward = reward_ctrl + reward_run
    #     done = False
    #     return ob, reward, done, dict(reward_run=reward_run,
    #         reward_ctrl=reward_ctrl)

    def _incremental_reward(self, state, action, next_state, reward, sumaxis1):
        ac_reg = sumaxis1(action * action)
        ac_reg *= 0.1
        reward -= ac_reg
        reward += (next_state[:, 0] - state[:, 0]) / self.unwrapped.dt
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

    def step(self, action):
        reward, state_after = self._mjc_step(action)
        done = False
        return state_after, reward, done, {}

    # gym code
    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[1:],
    #         self.sim.data.qvel.flat,
    #     ])

    def _get_obs(self):
        return np.concatenate([
            # difference from gym: need qpos x value for reward
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    # gym code
    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1,
    #         size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def set_state_from_ob(self, ob):
        self.reset()
        split = self.init_qpos.size
        qpos = ob[:split].reshape(self.init_qpos.shape)
        qvel = ob[split:].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)
