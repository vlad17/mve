"""
Optimize the planning problem explicitly with colocation.
"""

import numpy as np
import tensorflow as tf

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

        self._actions_ha = tf.get_variable(
            'colocation_acs', shape=[mpc_horizon, get_ac_dim(env)],
            initializer=tf.zeros_initializer)
        # states s_2...s_H+1
        self._states_hs = tf.get_variable(
            'colocation_obs', shape=[mpc_horizon, get_ob_dim(env)],
            initializer=tf.zeros_initializer)
        self._dynamics_dual_ph_h = tf.placeholder(tf.float32, [mpc_horizon])
        self._learner_dual_ph_l = tf.placeholder(
            tf.float32, [mpc_horizon - flags.opt_horizon])
        self._input_state_ph_s = tf.placeholder(
            tf.float32, [get_ob_dim(env)])

        # non-tf stored values for dual variables (to reuse between act calls)
        self._dynamics_dual_h = np.ones(mpc_horizon)
        self._learner_dual_l = np.ones(mpc_horizon - flags.opt_horizon)
        # re-use _actions_ha solutions between act() calls (but shift over 1)
        self._roll_op = _tf_roll_assign(self._actions_ha)
        # placeholder to update state
        self._state_init_ph_s = tf.placeholder(
            tf.float32, [mpc_horizon, get_ob_dim(env)])
        self._assign_state = tf.assign(
            self._states_hs, self._state_init_ph_s)

        opt = tf.train.AdamOptimizer(flags.primal_lr)

        def _body(t, _, primal):
            with tf.control_dependencies([primal]):
                primal_to_optimize = self._primal(
                    reward_fn, dyn_model, learner)
                update_op = opt.minimize(primal_to_optimize, var_list=[
                    self._actions_ha, self._states_hs])
            with tf.control_dependencies([update_op]):
                new_primal = self._primal(reward_fn, dyn_model, learner)
                return [t + 1, primal - new_primal, new_primal]

        def _cond(t, primal_improvement, _):
            return tf.logical_and(
                t < flags.primal_steps,
                primal_improvement > flags.tol)

        loop_vars = [0, flags.tol + 1, self._primal(
            reward_fn, dyn_model, learner)]
        self._primal_optimize_loop = tf.while_loop(
            _cond, _body, loop_vars, back_prop=False)
        self._reified_debug_tensors = self._debug_tensors(
            reward_fn, dyn_model, learner)
        self._first_acs = self._actions_ha[0]
        self._reified_reward = self._reward(reward_fn)
        self._step = None

        learner_violations = self._learner_violations_l(learner)
        dyn_violations = self._dyn_violations_h(dyn_model)
        self._reified_violations = learner_violations, dyn_violations
        self._dual_lr = flags.dual_lr
        self._dual_steps = flags.dual_steps

    def reset(self, n):
        tf.get_default_session().run([self._actions_ha.initializer])
        self._dynamics_dual_h = np.ones_like(self._dynamics_dual_h)
        self._learner_dual_l = np.ones_like(self._learner_dual_l)
        self._step = 0

    def act(self, states_ns):
        assert len(states_ns) == 1, 'batch size {} > 1 disallowed'.format(len(
            states_ns))
        tf.get_default_session().run(self._roll_op)
        feed_dict = {
            self._input_state_ph_s: states_ns[0],
            self._dynamics_dual_ph_h: self._dynamics_dual_h,
            self._learner_dual_ph_l: self._learner_dual_l}
        self._step += 1

        # need to always re-center on current state
        # we also need to always add a random perturb
        # without this perturbation some numerical ill-conditioning
        # shit totally breaks
        next_states = np.repeat(states_ns, self._mpc_horizon, axis=0)
        perturb = np.random.uniform(-1, 1, next_states.shape)
        perturb *= (np.fabs(states_ns[0]) / 100) + 0.01  # TODO: hacky, fix
        next_states += perturb
        tf.get_default_session().run(self._assign_state, {
            self._state_init_ph_s: next_states})

        # TODO reporter.report_incremental instead of logging
        log.debug('*' * 10 + ' STEP {:5d} '.format(self._step) + '*' * 10)

        for _ in range(self._dual_steps):
            before = tf.get_default_session().run(
                self._reified_debug_tensors, feed_dict)
            self._debug_print('before', before)

            # primal step
            steps, diff, _ = tf.get_default_session().run(
                self._primal_optimize_loop, feed_dict)

            log.debug('       ---> primal steps taken {} final improvement {}',
                      steps, diff)

            # dual step
            learner_violations, dyn_violations = tf.get_default_session().run(
                self._reified_violations, feed_dict)
            self._dynamics_dual_h += self._dual_lr * dyn_violations
            self._learner_dual_l += self._dual_lr * learner_violations

            after = tf.get_default_session().run(
                self._reified_debug_tensors, feed_dict)
            self._debug_print(' after', after)

        initial_acs, predicted_reward = tf.get_default_session().run(
            [self._first_acs, self._reified_reward], feed_dict)
        return [initial_acs], [predicted_reward]

    def _reward(self, reward_fn):
        first_reward = reward_fn(
            tf.expand_dims(self._input_state_ph_s, axis=0),
            self._actions_ha[:1], self._states_hs[:1], tf.zeros([1]))[0]
        rest_rewards = reward_fn(
            self._states_hs[:-1], self._actions_ha[1:],
            self._states_hs[1:], tf.zeros([self._mpc_horizon - 1]))
        return first_reward + tf.reduce_sum(rest_rewards)

    def _dyn_violations_h(self, dyn_model):
        first_predicted_state = dyn_model.predict_tf(
            tf.expand_dims(self._input_state_ph_s, axis=0),
            self._actions_ha[:1])
        first_dyn_violation = tf.norm(
            first_predicted_state - self._states_hs[:1], axis=1) ** 2
        rest_dyn_violations = tf.norm(
            dyn_model.predict_tf(self._states_hs[:-1], self._actions_ha[1:]) -
            self._states_hs[1:], axis=1) ** 2
        return tf.concat(
            [first_dyn_violation, rest_dyn_violations], axis=0)

    def _learner_violations_l(self, learner):
        # note we already assume opt_horizon > 0
        # get states s_o, ... s_h where o = opt_horizon
        states_o_to_h = self._states_hs[self._opt_horizon - 1:-1]
        return tf.norm(learner(states_o_to_h) -
                       self._actions_ha[self._opt_horizon:], axis=1) ** 2

    def _primal(self, reward_fn, dyn_model, learner):
        primal = -self._reward(reward_fn)
        primal += tf.reduce_sum(
            self._dynamics_dual_ph_h * self._dyn_violations_h(dyn_model))
        primal += tf.reduce_sum(
            self._learner_dual_ph_l * self._learner_violations_l(learner))
        return primal

    def _primal_grads(self, reward_fn, dyn_model, learner):
        return tf.gradients(
            self._primal(reward_fn, dyn_model, learner),
            [self._actions_ha, self._states_hs])

    def _debug_tensors(self, reward_fn, dyn_model, learner):
        ac_grad, state_grad = self._primal_grads(reward_fn, dyn_model, learner)

        dyn_dual = tf.reduce_sum(self._dynamics_dual_ph_h)
        learn_dual = tf.reduce_sum(self._learner_dual_ph_l)
        dyn_violations = tf.reduce_sum(self._dyn_violations_h(dyn_model))
        learn_violations = tf.reduce_sum(self._learner_violations_l(learner))

        return [
            self._primal(reward_fn, dyn_model, learner),
            self._reward(reward_fn),
            tf.norm(ac_grad), tf.norm(state_grad),
            tf.norm(self._actions_ha), tf.norm(self._states_hs),
            dyn_violations, learn_violations, dyn_dual, learn_dual]

    @staticmethod
    def _debug_print(prefix, dbg_tensors):
        msg = prefix
        msg += ' primal {:.5g} reward {:.5g} ac-grad-norm {:.5g}'
        msg += ' ob-grad-norm {:.5g}'
        msg += ' acs-norm {:.5g} ob-norm {:.5g}'
        log.debug(msg, *dbg_tensors[:-4])
        msg = ' ' * len(prefix)
        msg += ' dyn-violations {:.5g} learn-violations {:.5g}'
        msg += ' dyn-dual {:.5g} learn-dual {:.5g}'
        log.debug(msg, *dbg_tensors[-4:])


def _tf_roll_assign(tfvar):
    rolled = tf.concat([tfvar[-1:], tfvar[1:]], axis=0)
    return tf.assign(tfvar, rolled)


class _DummyLearner:
    def __init__(self, tf_action):
        self.tf_action = tf_action


def _dummy_learner(env):
    return lambda _: tf.zeros([0, get_ac_dim(env)])
