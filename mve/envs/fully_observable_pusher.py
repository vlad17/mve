"""
This file was made in the style of fully_observable_half_cheetah.py,
mimicking the analogous gym environment.
"""

import numpy as np
import tensorflow as tf

from .fully_observable import FullyObservable
from .render_free_mjc import RenderFreeMJC


class FullyObservablePusher(RenderFreeMJC, FullyObservable):
    """A fully-observable version of Pusher"""

    # gym code
    # def __init__(self):
    #     utils.EzPickle.__init__(self)
    #     mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

    def __init__(self):
        super().__init__("pusher.xml", 5)
        self.goal_pos = self.cylinder_pos = None

    # gym code
    # def step(self, a):
    #     vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
    #     vec_2 = self.get_body_com("object") - self.get_body_com("goal")
    #
    #     reward_near = - np.linalg.norm(vec_1)
    #     reward_dist = - np.linalg.norm(vec_2)
    #     reward_ctrl = - np.square(a).sum()
    #     reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
    #
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(reward_dist=reward_dist,
    #                                   reward_ctrl=reward_ctrl)

    @staticmethod
    def _incremental_reward(state, action, _, reward,
                            sqsum_axis1, norm):
        body_com_goal = state[:, -3:]
        body_com_object = state[:, -6:-3]
        body_com_tips_arm = state[:, -9:-6]

        vec_1 = body_com_object - body_com_tips_arm
        vec_2 = body_com_object - body_com_goal

        reward_near = - norm(vec_1)
        reward_dist = - norm(vec_2)
        reward_ctrl = - sqsum_axis1(action)
        reward += reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        return reward

    def np_reward(self, state, action, next_state):
        reward = 0
        reward = self._incremental_reward(
            state, action, next_state, reward,
            lambda x: np.square(x).sum(axis=1),
            lambda x: np.linalg.norm(x, axis=1))
        return reward

    def tf_reward(self, state, action, next_state):
        curr_reward = tf.zeros([tf.shape(state)[0]])
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.reduce_sum(tf.square(x), axis=1),
            lambda x: tf.norm(x, axis=1))

    def step(self, action):
        reward, state_after = self._mjc_step(action)
        done = False
        return state_after, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    # gym code
    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[:7],
    #         self.sim.data.qvel.flat[:7],
    #         self.get_body_com("tips_arm"),
    #         self.get_body_com("object"),
    #         self.get_body_com("goal"),
    #     ])

    def _get_obs(self):
        return np.concatenate([
            # difference from gym: need all qpos so states are re-settable
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    # gym code
    # def reset_model(self):
    #     qpos = self.init_qpos
    #
    #     self.goal_pos = np.asarray([0, 0])
    #     while True:
    #         self.cylinder_pos = np.concatenate([
    #             self.np_random.uniform(low=-0.3, high=0, size=1),
    #             self.np_random.uniform(low=-0.2, high=0.2, size=1)])
    #         if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
    #             break
    #
    #     qpos[-4:-2] = self.cylinder_pos
    #     qpos[-2:] = self.goal_pos
    #     qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
    #                                                    high=0.005,
    #                                                    size=self.model.nv)
    #     qvel[-4:] = 0
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def set_state_from_ob(self, ob):
        self.reset()
        split = self.init_qpos.size
        end = split + len(self.init_qvel)
        qpos = ob[:split].reshape(self.init_qpos.shape)
        qvel = ob[split:end].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)
