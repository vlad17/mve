"""
This file was made in the style of fully_observable_half_cheetah.py,
mimicking the analogous gym environment.
"""

import gym.envs.mujoco
import numpy as np
import tensorflow as tf

from .fully_observable import FullyObservable

class FullyObservableAnt(
        gym.envs.mujoco.mujoco_env.MujocoEnv, FullyObservable):
    """A fully-observable version of Ant"""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('ant.xml', 5)

    # gym code
    # def _step(self, a):
    #     xposbefore = self.get_body_com("torso")[0]
    #     self.do_simulation(a, self.frame_skip)
    #     xposafter = self.get_body_com("torso")[0]
    #     forward_reward = (xposafter - xposbefore)/self.dt
    #     ctrl_cost = .5 * np.square(a).sum()
    #     contact_cost = 0.5 * 1e-3 * np.sum(
    #         np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
    #     survive_reward = 1.0
    #     reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    #     state = self.state_vector()
    #     notdone = np.isfinite(state).all() \
    #         and state[2] >= 0.2 and state[2] <= 1.0
    #     done = not notdone
    #     ob = self._get_obs()
    #     return ob, reward, done, dict(
    #         reward_forward=forward_reward,
    #         reward_ctrl=-ctrl_cost,
    #         reward_contact=-contact_cost,
    #         reward_survive=survive_reward)

    def _incremental_reward(self, state, action, next_state, reward,
                            sqsum_axis1):
        xposbefore = state[:, 0]
        xposafter = next_state[:, 0]
        cfrc_coords = 6
        cfrc_len = cfrc_coords * self.model.nbody
        trunc_cfrc_ext = next_state[:, -cfrc_len:]

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * sqsum_axis1(action)
        contact_cost = 0.5 * 1e-3 * sqsum_axis1(trunc_cfrc_ext)
        survive_reward = 1.0
        reward += forward_reward - ctrl_cost - contact_cost + survive_reward
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
        qpos_state = self.state_vector()
        notdone = np.isfinite(qpos_state).all() \
            and qpos_state[2] >= 0.2 and qpos_state[2] <= 1.0
        done = not notdone
        return state_after, reward, done, {}

    # gym code
    # def _get_obs(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flat[1:],
    #         self.model.data.qvel.flat,
    #     ])

    def _get_obs(self):
        return np.concatenate([
            # difference from gym: need torso x value for reward
            # also the cfrc_ext, an 84-length vector of I-don't-know-what
            self.get_body_com('torso')[:1],
            # difference from gym: need to add in qpos x value
            # so that states are re-settable
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
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
            self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def set_state_from_ob(self, ob):
        self.reset()
        split = 1 + self.init_qpos.size
        end = split + self.init_qvel.size
        qpos = ob[1:split].reshape(self.init_qpos.shape)
        qvel = ob[split:end].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)
