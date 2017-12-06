"""
Optimize the planning problem explicitly with colocation.
"""

import numpy as np
import tensorflow as tf

from optimizer import AdamOptimizer
from policy import Policy
from utils import (get_ac_dim, get_ob_dim)
import log


class Colocation(Policy):  # pylint: disable=too-many-instance-attributes
    """
    Colocation based optimization of the planning objective
    looking mpc_horizon steps ahead.

    By default, with flags.opt_horizon == None or equal to mpc_horizon,
    random shooter performs
    unconstrained MPC planning. Otherwise, learner must not be
    None, and only the first flags.opt_horizon planning steps
    are optimized. The rest are taken according to the learner.

    The colocation objective optimizes:
    max(s_t,a_t) sum_t=0^h-1 reward(s_t, a_t, s_t+1)
    st for all t, ||s_t+1 - dynamics(s_t, a_t)||^2 = 0
       for t >= o, ||a_t - learner(s_t)||^2 = 0

    We softly solve this with Dual Gradient Ascent, optimizing the primal
    loss and then taking one dual gradient step per action. We save
    current primal and dual variable values between actions for warm starts.

    This uses the penalty method to enforce the constraints.
    TODO: consider Augmented Lagrangian for better numerical properties
    """

    def __init__(self, env, dyn_model, reward_fn, learner, mpc_horizon, flags):
        if flags.opt_horizon is None:
            flags.opt_horizon = mpc_horizon
        assert flags.opt_horizon <= mpc_horizon, (
            flags.opt_horizon, mpc_horizon)
        assert mpc_horizon > 0, mpc_horizon
        assert flags.opt_horizon > 0, flags.opt_horizon
        if flags.opt_horizon < mpc_horizon:
            assert learner is not None, learner
            learner = learner.tf_action
        else:
            learner = _dummy_learner(env)
        self._mpc_horizon = mpc_horizon
        self._opt_horizon = flags.opt_horizon

        # a = action dim
        # s = state dim
        # h = mpc_horiz
        # o = opt_horizon
        # l = h - o (learner constrained)
        #
        # the optimization variables are a_1, ..., a_H in _actions_ha
        # and s_2, ..., s_H+1 in _states_hs
        # s_1 is given to us at act() time by a placeholder.
        # between actions in the same episode, we warm-start our optimization
        # variables. a_i starts with the previous a_i+1 and the dual
        # variable values stay the same.
        self._actions_ha = np.zeros((mpc_horizon, get_ac_dim(env)))
        self._states_hs = np.zeros((mpc_horizon, get_ob_dim(env)))
        self._actions_ph_ha = tf.placeholder(
            tf.float32, self._actions_ha.shape)
        self._states_ph_hs = tf.placeholder(tf.float32, self._states_hs.shape)

        self._dynamics_dual_ph_h = tf.placeholder(tf.float32, [mpc_horizon])
        self._learner_dual_ph_l = tf.placeholder(
            tf.float32, [mpc_horizon - flags.opt_horizon])
        self._input_state_ph_s = tf.placeholder(
            tf.float32, [get_ob_dim(env)])

        self._primal_optimizer = AdamOptimizer(
            lambda states, actions: self._primal(
                reward_fn, dyn_model, learner, states, actions),
            [self._states_hs.shape, self._actions_ha.shape],
            flags.primal_steps,
            flags.tol,
            flags.primal_lr)

        self._dynamics_dual_h = np.ones(mpc_horizon)
        self._learner_dual_l = np.ones(mpc_horizon - flags.opt_horizon)

        self._reified_debug_tensors = self._debug_tensors(
            reward_fn, dyn_model, learner, self._states_ph_hs,
            self._actions_ph_ha,)
        self._reified_reward = self._reward(
            reward_fn, self._states_ph_hs, self._actions_ph_ha)
        self._step = None
        self._input_state = None

        learner_violations = self._learner_violations_l(
            learner, self._states_ph_hs, self._actions_ph_ha)
        dyn_violations = self._dyn_violations_h(
            dyn_model, self._states_ph_hs, self._actions_ph_ha)
        self._reified_violations = learner_violations, dyn_violations
        self._dual_lr = flags.dual_lr
        self._dual_steps = flags.dual_steps

    def reset(self, n):
        self._dynamics_dual_h = np.ones_like(self._dynamics_dual_h)
        self._learner_dual_l = np.ones_like(self._learner_dual_l)
        self._step = 0
        self._input_state = None

    def _update_to_new_state(self, states_ns):
        assert len(states_ns) == 1, 'batch size {} > 1 disallowed'.format(len(
            states_ns))
        self._input_state = states_ns[0]
        self._step += 1

        # need to always re-center on current state to not diverge in solution
        # space
        # we also need to always add a random perturb
        # without this perturbation some numerical ill-conditioning
        # shit totally breaks
        next_states = np.repeat(states_ns, self._mpc_horizon, axis=0)
        perturb = np.random.uniform(-1, 1, next_states.shape)
        # TODO: hacky, fix
        perturb *= (np.fabs(self._input_state) / 100) + 0.01
        next_states += perturb
        self._states_hs = next_states
        if self._step == 1:
            self._actions_ha[:] = 0
            # TODO: hacky, fix
            # actions similarly need perturbation on the first step
            self._actions_ha[self._opt_horizon:] += np.random.uniform(
                -1, 1, self._actions_ha[self._opt_horizon:].shape) * 0.01
        else:
            self._actions_ha = np.roll(self._actions_ha, 1, axis=0)

    def _feed_dict(self):
        assert self._input_state is not None
        return {
            self._input_state_ph_s: self._input_state,
            self._dynamics_dual_ph_h: self._dynamics_dual_h,
            self._learner_dual_ph_l: self._learner_dual_l,
            self._actions_ph_ha: self._actions_ha,
            self._states_ph_hs: self._states_hs}

    def _dual_step(self):
        learner_violations, dyn_violations = tf.get_default_session().run(
            self._reified_violations, self._feed_dict())
        self._dynamics_dual_h += self._dual_lr * dyn_violations
        self._learner_dual_l += self._dual_lr * learner_violations

    def act(self, states_ns):
        self._update_to_new_state(states_ns)

        # TODO reporter.report_incremental instead of logging
        log.debug('*' * 10 + ' STEP {:5d} '.format(self._step) + '*' * 10)

        for _ in range(self._dual_steps):
            self._debug_print('before')
            self._states_hs, self._actions_ha = (
                self._primal_optimizer.minimize(
                    self._feed_dict(), self._states_hs, self._actions_ha))
            self._dual_step()
            self._debug_print(' after')

        predicted_reward = tf.get_default_session().run(
            self._reified_reward, self._feed_dict())
        self._input_state = None
        return self._actions_ha[:1], [predicted_reward]

    def _reward(self, reward_fn, states_hs, actions_ha):
        first_reward = reward_fn(
            tf.expand_dims(self._input_state_ph_s, axis=0),
            actions_ha[:1], states_hs[:1], tf.zeros([1]))[0]
        rest_rewards = reward_fn(
            states_hs[:-1], actions_ha[1:],
            states_hs[1:], tf.zeros([self._mpc_horizon - 1]))
        return first_reward + tf.reduce_sum(rest_rewards)

    def _dyn_violations_h(self, dyn_model, states_hs, actions_ha):
        first_predicted_state = dyn_model.predict_tf(
            tf.expand_dims(self._input_state_ph_s, axis=0),
            actions_ha[:1])
        first_dyn_violation = tf.norm(
            first_predicted_state - states_hs[:1], axis=1) ** 2
        rest_dyn_violations = tf.norm(
            dyn_model.predict_tf(states_hs[:-1], actions_ha[1:]) -
            states_hs[1:], axis=1) ** 2
        return tf.concat(
            [first_dyn_violation, rest_dyn_violations], axis=0)

    def _learner_violations_l(self, learner, states_hs, actions_ha):
        # note we already assume opt_horizon > 0
        # get states s_o, ... s_h where o = opt_horizon
        states_o_to_h = states_hs[self._opt_horizon - 1:-1]
        return tf.norm(learner(states_o_to_h) -
                       actions_ha[self._opt_horizon:], axis=1) ** 2

    def _primal(self, reward_fn, dyn_model, learner, states_hs, actions_ha):
        primal = -self._reward(reward_fn, states_hs, actions_ha)
        primal += tf.reduce_sum(
            self._dynamics_dual_ph_h * self._dyn_violations_h(
                dyn_model, states_hs, actions_ha))
        primal += tf.reduce_sum(
            self._learner_dual_ph_l * self._learner_violations_l(
                learner, states_hs, actions_ha))
        return primal

    def _debug_tensors(self, reward_fn, dyn_model, learner, states, actions):
        ac_grad, state_grad = tf.gradients(
            self._primal(reward_fn, dyn_model, learner, states, actions),
            [actions, states])

        dyn_dual = tf.reduce_sum(self._dynamics_dual_ph_h)
        learn_dual = tf.reduce_sum(self._learner_dual_ph_l)
        dyn_violations = tf.reduce_sum(self._dyn_violations_h(
            dyn_model, states, actions))
        learn_violations = tf.reduce_sum(self._learner_violations_l(
            learner, states, actions))

        return [
            self._primal(reward_fn, dyn_model, learner, states, actions),
            self._reward(reward_fn, states, actions),
            tf.norm(ac_grad), tf.norm(state_grad),
            tf.norm(actions), tf.norm(states),
            dyn_violations, learn_violations, dyn_dual, learn_dual]

    def _debug_print(self, prefix):
        dbg_tensors = tf.get_default_session().run(
            self._reified_debug_tensors, self._feed_dict())
        msg = prefix
        msg += ' primal {:.5g} reward {:.5g} ac-grad-norm {:.5g}'
        msg += ' ob-grad-norm {:.5g}'
        msg += ' acs-norm {:.5g} ob-norm {:.5g}'
        log.debug(msg, *dbg_tensors[:-4])
        msg = ' ' * len(prefix)
        msg += ' dyn-violations {:.5g} learn-violations {:.5g}'
        msg += ' dyn-dual {:.5g} learn-dual {:.5g}'
        log.debug(msg, *dbg_tensors[-4:])


def _dummy_learner(env):
    return lambda _: tf.zeros([0, get_ac_dim(env)])
