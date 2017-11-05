"""
White-box environment (wrappers) which reveal relevant physical info and
their cost functions.
"""

import gym
from gym import ObservationWrapper, spaces
from gym.envs.mujoco import HalfCheetahEnv, MujocoEnv
import numpy as np
import tensorflow as tf

from utils import inherit_doc, get_ob_dim, get_ac_dim


class WrapperWithCost(ObservationWrapper):
    """
    An observation wrapper for an environment, where
    the cost function is specified in the form indicated below.
    """

    def tf_cost(self, state, action, next_state, curr_cost):
        """
        Given tensors (with the 0th dimension as the batch dimension) for a
        transition during a rollout, and the current cost up to the point in
        time where the transitions passed as arguments have occurred, this
        returns the resulting cost for the trajectories after the transition.
        """
        raise NotImplementedError


class CostCalculator:
    """
    A helper class which generates a full-trajectory cost function given
    a TF cost function in the format specified by WrapperWithCost
    """

    def __init__(self, horizon, env):
        # t - time horizon n - batch size, s/a - state/action dim
        ob_dim = get_ob_dim(env)
        self._state_ph_tns = tf.placeholder(
            tf.float32, [horizon, None, ob_dim], 'cost_state_input')
        self._next_state_ph_tns = tf.placeholder(
            tf.float32, [horizon, None, ob_dim], 'cost_next_state_input')
        self._action_ph_tna = tf.placeholder(
            tf.float32, [horizon, None, get_ac_dim(env)], 'cost_action_input')
        initial_cost_n = tf.zeros([tf.shape(self._state_ph_tns)[1]])
        elems = (self._state_ph_tns,
                 self._action_ph_tna,
                 self._next_state_ph_tns)
        self._trajectory_cost_n = tf.scan(
            lambda c, s_a_ns: env.tf_cost(*(s_a_ns + (c,))),
            elems,
            initial_cost_n,
            back_prop=False)[-1]

    def trajectory_cost(self, state, action, next_state):
        """
        Given numpy arrays for transitions (with dimensions being time first,
        then batch dimension, then the action/state dimensions), compute
        the resulting vector of trajectory costs for each rollout (length
        will be the batch size).
        """
        return tf.get_default_session().run(
            self._trajectory_cost_n, feed_dict={
                self._state_ph_tns: state,
                self._action_ph_tna: action,
                self._next_state_ph_tns: next_state})


class HalfCheetahEnvFS(HalfCheetahEnv):
    """HalfCheetah, with frameskip 1 (for consistency with HW4)"""

    def __init__(self, frame_skip):  # pylint: disable=super-init-not-called
        MujocoEnv.__init__(self, 'half_cheetah.xml', frame_skip)  # pylint: disable=non-parent-init-called
        gym.utils.EzPickle.__init__(self)  # pylint: disable=non-parent-init-called

@inherit_doc
class WhiteBoxMuJoCo(WrapperWithCost):
    """
    An observation wrapper for the HalfCheetahEnv with agents with a
    torso, where the observation vector
    contains slightly more information than usual. The usual
    environment filters out some coordinate information. This might make
    model-free policy learning easier, but it makes dynamics prediction
    unreasonably hard (by removing x-coordinate info).

    Optionally, this enviroment can make dynamics learning even easier
    by wiring in center-of-mass data.

    Finally, users can select an easy (extra supervision) or hard
    (raw negative reward function) cost. If using the hard cost,
    users can further specify the regularization coefficient for the
    action magnitude. To match the reward function, it should be the
    default value, 0.1.
    """

    def __init__(self, env, com_pos=False, com_vel=False,
                 easy_cost=False, action_regularization=0.1):
        super().__init__(env)
        self._com_pos = com_pos
        self._com_vel = com_vel
        self._easy_cost = easy_cost
        self._action_regularization = action_regularization
        gym_env = self.unwrapped
        assert isinstance(gym_env, HalfCheetahEnvFS), gym_env
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._observation(None).shape)

    def _observation(self, _):
        observations = [
            self.unwrapped.model.data.qpos.flat,
            self.unwrapped.model.data.qvel.flat]
        if self._com_pos:
            observations.append(self.unwrapped.get_body_com('torso').flat)
        if self._com_vel:
            observations.append(self.unwrapped.get_body_comvel('torso').flat)
        return np.concatenate(observations)

    def tf_cost(self, state, action, next_state, curr_cost):
        cost = curr_cost
        if self._easy_cost:
            heading_penalty_factor = tf.constant(10.)

            front_leg = state[:, 6]
            my_range = 0.2
            cost += tf.cast(front_leg >= my_range, tf.float32) * \
                heading_penalty_factor

            front_shin = state[:, 7]
            my_range = 0.
            cost += tf.cast(front_shin >= my_range, tf.float32) * \
                heading_penalty_factor

            front_foot = state[:, 8]
            my_range = 0.
            cost += tf.cast(front_foot >= my_range, tf.float32) * \
                heading_penalty_factor

            cost -= (next_state[:, 0] - state[:, 0]) / self.unwrapped.dt
        else:
            ac_reg = tf.reduce_sum(tf.square(action), axis=1)
            ac_reg *= self._action_regularization
            cost += ac_reg
            cost -= (next_state[:, 0] - state[:, 0]) / self.unwrapped.dt
        return cost
