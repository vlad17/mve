"""
This file was made in the style of fully_observable_half_cheetah.py,
mimicking the analogous gym environment.
"""

import numpy as np
import tensorflow as tf

from .fully_observable import FullyObservable
from .render_free_mjc import RenderFreeMJC


class FullyObservableHopper(RenderFreeMJC, FullyObservable):
    """A fully-observable version of hopper."""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('hopper.xml', 4)

    # gym code
    # def step(self, a):
    #     posbefore = self.sim.data.qpos[0]
    #     self.do_simulation(a, self.frame_skip)
    #     posafter, height, ang = self.sim.data.qpos[0:3]
    #     alive_bonus = 1.0
    #     reward = (posafter - posbefore) / self.dt
    #     reward += alive_bonus
    #     reward -= 1e-3 * np.square(a).sum()
    #     s = self.state_vector()
    #     done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
    #                 (height > .7) and (abs(ang) < .2))
    #     ob = self._get_obs()
    #     return ob, reward, done, {}

    def _incremental_reward(self, state, action, next_state, reward, sqsum):
        posbefore = state[:, 0]
        posafter = next_state[:, 0]
        alive_bonus = 1.0
        new_reward = (posafter - posbefore) / self.dt
        new_reward += alive_bonus
        new_reward -= 1e-3 * sqsum(action)
        reward += new_reward
        return reward

    def np_reward(self, state, action, next_state):
        reward = 0
        reward = self._incremental_reward(
            state, action, next_state, reward,
            lambda x: np.sum(np.square(x), axis=1))
        return reward

    def tf_reward(self, state, action, next_state):
        curr_reward = tf.zeros([tf.shape(state)[0]])
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.reduce_sum(tf.square(x), axis=1))

    def step(self, action):
        reward, state_after = self._mjc_step(action)
        s = self.state_vector()
        height, ang = state_after[1:3]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        return state_after, reward, done, {}

    # gym code
    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[1:],
    #         np.clip(self.sim.data.qvel.flat, -10, 10)
    #     ])

    def _get_obs(self):
        return np.concatenate([
            # difference from gym: need to add in qpos x value
            # for reward
            self.sim.data.qpos.flat,
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    # gym code
    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(
    #         low=-.005, high=.005, size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.uniform(
    #         low=-.005, high=.005, size=self.model.nv)
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def set_state_from_ob(self, ob):
        self.reset()
        split = self.init_qpos.size
        qpos = ob[:split].reshape(self.init_qpos.shape)
        qvel = ob[split:].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)
