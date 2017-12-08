"""
Gym environments might withhold some items from the observation which
make it impossible to compute the reward from the observation alone.

This file contains amended environments with sufficient information
to compute reward from observations (so MPC can use the observations
directly).

This file is mostly environments copied from the gym package,
but with reward functions exposed.
"""

from gym.envs.mujoco import MujocoEnv
from gym.utils import EzPickle
import numpy as np
import tensorflow as tf


class SettableWithReward:
    """
    A mixin class with a white-box incremental reward function.

    In addition, such an environment is "settable": given an observation
    this environment can reset to that observation's state for
    debugging purposes. Thus this implicitly requires a fully observable
    MDP.
    """

    def tf_reward(self, state, action, next_state, curr_reward):
        """
        Given tensors (with the 0th dimension as the batch dimension) for a
        transition during a rollout, and the current reward up to the point in
        time where the transitions passed as arguments have occurred, this
        returns the resulting reward for the trajectories after the transition.
        """
        raise NotImplementedError

    def set_state_from_ob(self, ob):
        """
        Reset the environment, starting at the state corresponding to the
        observation ob.
        """
        raise NotImplementedError


class WhiteBoxHalfCheetahEasy(MujocoEnv, EzPickle, SettableWithReward):
    """White box HalfCheetah, with frameskip 1 and easy reward"""
    # same as gym except where highlighted

    def __init__(self, frame_skip):  # pylint: disable=super-init-not-called
        MujocoEnv.__init__(self, 'half_cheetah.xml',
                           frame_skip)  # pylint: disable=non-parent-init-called
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
            # difference from gym: need qpos x value for reward
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

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



class WhiteBoxHalfCheetahHard(MujocoEnv, EzPickle, SettableWithReward):
    """White box HalfCheetah, with frameskip 1 and hard reward"""
    # same as gym except as noted

    def __init__(self, frame_skip):  # pylint: disable=super-init-not-called
        MujocoEnv.__init__(self, 'half_cheetah.xml',
                           frame_skip)  # pylint: disable=non-parent-init-called
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
            # difference from gym: need qpos x value for reward
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

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


class WhiteBoxAntEnv(MujocoEnv, EzPickle, SettableWithReward):
    """
    The ant environment, with extra observation values for calculating
    reward.
    """

    def __init__(self, frame_skip):  # pylint: disable=super-init-not-called
        MujocoEnv.__init__(
            self, 'ant.xml', frame_skip)  # pylint: disable=non-parent-init-called
        EzPickle.__init__(self)  # pylint: disable=non-parent-init-called

    def _incremental_reward(self, state, action, next_state, reward,
                            sqsum_axis1):
        xposbefore = state[:, 0]
        xposafter = next_state[:, 0]
        trunc_cfrc_ext = next_state[:, -84:]

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * sqsum_axis1(action)
        contact_cost = 0.5 * 1e-3 * sqsum_axis1(trunc_cfrc_ext)
        survive_reward = 1.0
        reward += forward_reward - ctrl_cost - contact_cost + survive_reward
        return reward

    def _np_incremental_reward(self, state, action, next_state):
        reward = 0
        reward = self._incremental_reward(state, action, next_state, reward,
                                          lambda x: np.square(x).sum(axis=1))
        return reward

    def tf_reward(self, state, action, next_state, curr_reward):
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.reduce_sum(tf.square(x), axis=1))

    def _step(self, action):
        state_before = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        state_after = self._get_obs()
        reward = self._np_incremental_reward(
            state_before[np.newaxis, ...],
            action[np.newaxis, ...],
            state_after[np.newaxis, ...])
        qpos_state = self.state_vector()
        notdone = np.isfinite(qpos_state).all() \
            and qpos_state[2] >= 0.2 and qpos_state[2] <= 1.0
        done = not notdone
        return state_after, reward, done, {}

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


class WhiteBoxWalker2dEnv(MujocoEnv, EzPickle, SettableWithReward):
    """
    The Walker2d environment [1], with extra observation values for calcuating
    reward. Everything is copied from [1] unless otherwise noted.

    [1]: https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d.py
    """
    def __init__(self, frame_skip):
        MujocoEnv.__init__(self, "walker2d.xml", frame_skip)
        EzPickle.__init__(self)

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

    def _np_incremental_reward(self, state, action, next_state):
        reward = 0
        reward = self._incremental_reward(state, action, next_state, reward,
                                          lambda x: np.square(x).sum(axis=1))
        return reward

    def tf_reward(self, state, action, next_state, curr_reward):
        return self._incremental_reward(
            state, action, next_state, curr_reward,
            lambda x: tf.reduce_sum(tf.square(x), axis=1))

    def _step(self, action):
        # For reference, here is the original _step function:
        #
        #     def _step(self, a):
        #         posbefore = self.model.data.qpos[0, 0]
        #         self.do_simulation(a, self.frame_skip)
        #         posafter, height, ang = self.model.data.qpos[0:3, 0]
        #         alive_bonus = 1.0
        #         reward = ((posafter - posbefore) / self.dt)
        #         reward += alive_bonus
        #         reward -= 1e-3 * np.square(a).sum()
        #         done = not (height > 0.8 and height < 2.0 and
        #                     ang > -1.0 and ang < 1.0)
        #         ob = self._get_obs()
        #         return ob, reward, done, {}
        state_before = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        state_after = self._get_obs()
        reward = self._np_incremental_reward(
            state_before[np.newaxis, ...],
            action[np.newaxis, ...],
            state_after[np.newaxis, ...])
        height, ang = state_after[1:3]
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return state_after, reward, done, {}

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
