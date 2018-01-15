"""
This file was made in the style of fully_observable_half_cheetah.py,
mimicking the analogous gym environment.
"""

import gym.envs.mujoco
import numpy as np
import tensorflow as tf

from .fully_observable import FullyObservable

class FullyObservableWalker2d(
        gym.envs.mujoco.mujoco_env.MujocoEnv, FullyObservable):
    """A fully-observable version of Walker2d"""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__("walker2d.xml", 4)

    # gym code
    # def _step(self, a):
    #     posbefore = self.model.data.qpos[0, 0]
    #     self.do_simulation(a, self.frame_skip)
    #     posafter, height, ang = self.model.data.qpos[0:3, 0]
    #     alive_bonus = 1.0
    #     reward = ((posafter - posbefore) / self.dt)
    #     reward += alive_bonus
    #     reward -= 1e-3 * np.square(a).sum()
    #     done = not (height > 0.8 and height < 2.0 and
    #                 ang > -1.0 and ang < 1.0)
    #     ob = self._get_obs()
    #     return ob, reward, done, {}

    def _incremental_reward(self, state, action, next_state, reward,
                            sqsum_axis1):
        # For reference, see _step for the original cost calculation that this
        # method copies.
        posbefore = state[:, 0]
        posafter = next_state[:, 0]
        alive_bonus = 1.0
        new_reward = ((posafter - posbefore) / self.dt)
        new_reward += alive_bonus
        new_reward -= 1e-3 * sqsum_axis1(action)
        reward += new_reward
        return reward

    def np_reward(self, state, action, next_state):
        reward = 0
        reward = self._incremental_reward(state, action, next_state, reward,
                                          lambda x: np.square(x).sum(axis=1))
        return reward

    def tf_reward(self, state, action, next_state):
        curr_reward = tf.zeros([tf.shape(state)[0]])
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.reduce_sum(tf.square(x), axis=1))

    def _step(self, action):
        reward, state_after = self._mjc_step(action)
        height, ang = state_after[1:3]
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return state_after, reward, done, {}

    # gym code
    # def _get_obs(self):
    #     qpos = self.model.data.qpos
    #     qvel = self.model.data.qvel
    #     return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        # Note that qpos has shape 9x1 and qvel has shape 9x1 (determined by
        # printing them out). The orignal Walker2dEnv returns the following:
        #
        #   return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        #
        # That is, it returns all but the first entry of qpos. The first entry
        # of qpos is needed to compute reward, so we return it.
        return np.concatenate([qpos, np.clip(qvel, -10, 10)]).ravel()

    # gym code
    # def reset_model(self):
    #     self.set_state(
    #         self.init_qpos + self.np_random.uniform(low=-.005, high=.005,
    #         size=self.model.nq),
    #         self.init_qvel + self.np_random.uniform(low=-.005, high=.005,
    #         size=self.model.nv)
    #     )
    #     return self._get_obs()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(
                low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(
                low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def set_state_from_ob(self, ob):
        self.reset()
        split = self.init_qpos.size
        qpos = ob[:split].reshape(self.init_qpos.shape)
        qvel = ob[split:].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)
