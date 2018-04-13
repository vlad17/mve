"""
This file was made in the style of fully_observable_half_cheetah.py,
mimicking the analogous gym environment.
"""

from gym import spaces
import numpy as np
import tensorflow as tf

from .fully_observable import FullyObservable
from .render_free_mjc import RenderFreeMJC
from .rescale import scale_from_unit


class FullyObservableHumanoid(RenderFreeMJC, FullyObservable):
    """A fully-observable version of humanoid (actions rescaled to [-1, 1])"""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('humanoid.xml', 5)
        self._original_action_space = self.action_space
        self.action_space = spaces.Box(
            -1, 1, self._original_action_space.low.shape)

    # def mass_center(model, sim):
    #     mass = np.expand_dims(model.body_mass, 1)
    #     xpos = sim.data.xipos
    #     return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def _com(self, xipos, sumaxis1, reshape, ax0len):
        # xipos is batch x nbody x 3 com position for every body in the sims
        # it gives the com coordinates
        mass = self.model.body_mass.reshape(1, self.model.nbody, 1)
        xipos = reshape(xipos, [ax0len(xipos), -1, 3])
        return sumaxis1(mass * xipos) / np.sum(mass)

    # gym code
    # def step(self, a):
    #     pos_before = mass_center(self.model, self.sim)
    #     self.do_simulation(a, self.frame_skip)
    #     pos_after = mass_center(self.model, self.sim)
    #     alive_bonus = 5.0
    #     data = self.sim.data
    #     lin_vel_cost = 0.25 * (pos_after - pos_before) / \
    #         self.model.opt.timestep
    #     quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
    #     quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
    #     quad_impact_cost = min(quad_impact_cost, 10)
    #     reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + \
    #         alive_bonus
    #     qpos = self.sim.data.qpos
    #     done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
    #     return self._get_obs(), reward, done, dict(
    #         reward_linvel=lin_vel_cost,
    #         reward_quad ctrl=-quad_ctrl_cost,
    #         reward_alive=alive_bonus,
    #         reward_impact=-quad_impact_cost)

    def _incremental_reward(self, state, action, next_state, reward,
                            sumaxis1, minimum, reshape, ax0len):
        # see _get_obs for order -- xipos is at the end
        xipos_start = -self.model.nbody * 3
        xipos_before = state[:, xipos_start:]
        xipos_after = next_state[:, xipos_start:]
        xpos_before = self._com(xipos_before, sumaxis1, reshape, ax0len)[:, 0]
        xpos_after = self._com(xipos_after, sumaxis1, reshape, ax0len)[:, 0]
        alive_bonus = 5.0
        lin_vel_cost = 0.25 * (xpos_after - xpos_before) / \
            self.model.opt.timestep
        # note action == data.ctrl
        quad_ctrl_cost = 0.1 * sumaxis1(action * action)
        # see _get_obs for order -- cfrc right befor xipos
        cfrc_start = xipos_start - self.model.nbody * 6
        cfrc_ext_after = next_state[:, cfrc_start:xipos_start]
        quad_impact_cost = .5e-6 * sumaxis1(cfrc_ext_after * cfrc_ext_after)
        quad_impact_cost = minimum(quad_impact_cost, 10)
        reward += (
            lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus)
        return reward

    def np_reward(self, state, action, next_state):
        # this call assumes action is already rescaled (not for external use)
        reward = 0
        reward = self._incremental_reward(
            state, action, next_state, reward,
            lambda x: np.sum(x, axis=1),
            np.minimum,
            np.reshape,
            lambda x: x.shape[0])
        return reward

    def tf_reward(self, state, action, next_state):
        action = scale_from_unit(self._original_action_space, action)
        curr_reward = tf.zeros([tf.shape(state)[0]])
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.reduce_sum(x, axis=1),
            tf.minimum,
            tf.reshape,
            lambda x: tf.shape(x)[0])

    def step(self, action):
        if hasattr(self, '_original_action_space'):
            # this attribute may not be defined for weird gym init reasons,
            # don't worry about it (gym is just trying to get the
            # observation size
            action = scale_from_unit(self._original_action_space, action)
        reward, state_after = self._mjc_step(action)
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return state_after, reward, done, {}

    # gym code
    # def _get_obs(self):
    #     data = self.sim.data
    #     return np.concatenate([data.qpos.flat[2:],
    #                            data.qvel.flat,
    #                            data.cinert.flat,
    #                            data.cvel.flat,
    #                            data.qfrc_actuator.flat,
    #                            data.cfrc_ext.flat])

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([
            # difference from gym: need full qpos to make env markov
            data.qpos.ravel(),
            data.qvel.ravel(),
            data.cinert.ravel(),
            data.cvel.ravel(),
            data.qfrc_actuator.ravel(),
            # difference from gym: clip the contact forces as done in
            # the rllab env. We need to do this so that the baseline,
            # Soft Actor Critic, performs well (its paper demonstrates
            # rllab humanoid performance).
            np.clip(data.cfrc_ext.ravel(), -1, 1),
            # difference from gym: need to add in com pos for reward
            data.xipos.ravel()
        ])

    # gym code
    # def reset_model(self):
    #     c = 0.01
    #     self.set_state(
    #         self.init_qpos +
    #         self.np_random.uniform(low=-c, high=c, size=self.model.nq),
    #         self.init_qvel +
    #         self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
    #     )
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    # gym code
    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 1
    #     self.viewer.cam.distance = self.model.stat.extent * 1.0
    #     self.viewer.cam.lookat[2] += .8
    #     self.viewer.cam.elevation = -20

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def set_state_from_ob(self, ob):
        self.reset()
        split = self.init_qpos.size
        qpos = ob[:split].reshape(self.init_qpos.shape)
        end = self.init_qvel.size + split
        qvel = ob[split:end].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)
