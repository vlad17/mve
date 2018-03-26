"""
SAC training algorithm.
"""

import numpy as np
import tensorflow as tf

from context import flags
import env_info
import reporter
from log import debug
from sample import sample_venv
from tf_reporter import TFReporter
from qvalues import qvals, offline_oracle_q
from utils import as_controller, timeit


class SAC:
    """Implemets training algorithm for soft actor critic."""

    def __init__(self, policy, qfn, vfn, dynamics):
        self._reporter = TFReporter()
        # we use one-sample monte carlo estimators
        # feeding copies of the same state to obs_ph_ns increases
        # the number of samples used to perform the integral

        # TODO consider using different obs samples
        # for the pi and V updates -- TODO also try this on DDPG

        self.obs_ph_ns = tf.placeholder(
            tf.float32, shape=[None, env_info.ob_dim()])
        self.next_obs_ph_ns = tf.placeholder(
            tf.float32, shape=[None, env_info.ob_dim()])
        self.terminals_ph_n = tf.placeholder(
            tf.float32, shape=[None])
        self.rewards_ph_n = tf.placeholder(
            tf.float32, shape=[None])
        self.actions_ph_na = tf.placeholder(
            tf.float32, shape=[None, env_info.ac_dim()])

        temperature = flags().sac.temperature
        current_V_n = vfn.tf_state_value(self.obs_ph_ns)
        self._policy_loss = policy.expectation(
            self.obs_ph_ns,
            lambda acs_na, log_pi_n: (
                temperature * log_pi_n -
                qfn.tf_state_action_value(self.obs_ph_ns, acs_na) +
                current_V_n))
        self._reporter.scalar('policy loss', self._policy_loss)
        policy.tf_report(self._reporter, self.obs_ph_ns)

        # targets
        with tf.variable_scope('targets'):
            next_V_n = vfn.tf_state_value(self.next_obs_ph_ns)
            from utils import trainable_vars
            target_vars = trainable_vars(tf.get_variable_scope().name)

        # note the Q,V here are soft
        # full details https://arxiv.org/abs/1801.01290
        # need to apply TD-k trick to both V and Q -- Q needs rewards, V just
        # needs local Q updates
        current_Q_n = qfn.tf_state_action_value(
            self.obs_ph_ns, self.actions_ph_na)
        self._reporter.stats('buffer Q', current_Q_n)
        self._reporter.stats('buffer V', current_V_n)
        discount = flags().experiment.discount
        if flags().sac.sac_mve:
            h = flags().sac.model_horizon
            debug('using learned-dynamics Q estimator with {} steps as '
                  'target critic', h)
            
            # --- this section is purely for dynamics accuracy recording
            from dynamics_metrics import DynamicsMetrics
            self._dyn_metrics = DynamicsMetrics(
                h, env_info.make_env, flags().dynamics_metrics, discount)
            self._unroll_states_ph_ns = tf.placeholder(
                tf.float32, shape=[None, env_info.ob_dim()])
            n = tf.shape(self._unroll_states_ph_ns)[0]
            with tf.variable_scope('scratch-sac'):
                actions_hna = tf.get_variable(
                    'dynamics_actions', initializer=tf.zeros(
                        [h, n, env_info.ac_dim()]),
                    dtype=tf.float32, trainable=False, collections=[],
                    validate_shape=False)
                states_hns = tf.get_variable(
                    'dynamics_states', initializer=tf.zeros(
                        [h, n, env_info.ob_dim()]),
                    dtype=tf.float32, trainable=False, collections=[],
                    validate_shape=False)
            from ddpg.ddpg import _tf_unroll
            self._unroll_loop = _tf_unroll(
                h, self._unroll_states_ph_ns,
                policy.tf_greedy_action, dynamics,
                actions_hna, states_hns)
            self._initializer = [
                actions_hna.initializer, states_hns.initializer]
            # --- end section for dyn acc diagnostics

            self._qfn_loss, self._vfn_loss = _tf_compute_model_value_expansion(
                self.obs_ph_ns,
                self.actions_ph_na,
                self.rewards_ph_n,
                self.next_obs_ph_ns,
                self.terminals_ph_n,
                policy,
                qfn,
                vfn,
                dynamics)
        else:
            target_Q_n = self.rewards_ph_n + (1. - self.terminals_ph_n) * (
                discount * next_V_n)
            self._reporter.stats('target Q', target_Q_n)
            self._qfn_loss = tf.losses.mean_squared_error(
                tf.stop_gradient(target_Q_n), current_Q_n)

            onpol_act_na, log_prob_acs_n = policy.tf_sample_action_with_log_prob(
                self.obs_ph_ns)
            onpol_qfn_n = qfn.tf_state_action_value(
                self.obs_ph_ns, onpol_act_na)
            target_V_n = onpol_qfn_n - temperature * log_prob_acs_n
            self._reporter.stats('target V', target_V_n)
            self._vfn_loss = tf.losses.mean_squared_error(
                tf.stop_gradient(target_V_n), current_V_n)

        self._reporter.scalar('qfn loss', self._qfn_loss)
        self._reporter.scalar('vfn loss', self._vfn_loss)

        # all this should be refactored into an "optimize together" utility
        vfn_opt = tf.train.AdamOptimizer(learning_rate=flags().sac.value_lr)
        qfn_opt = tf.train.AdamOptimizer(learning_rate=flags().sac.value_lr)
        policy_opt = tf.train.AdamOptimizer(
            learning_rate=flags().sac.policy_lr)
        self._reporter.grads(
            'policy grad', policy_opt, self._policy_loss, policy.variables)
        self._reporter.grads(
            'qfn grad', qfn_opt, self._qfn_loss, qfn.variables)
        self._reporter.grads(
            'vfn grad', vfn_opt, self._vfn_loss, vfn.variables)

        losses = [
            (policy_opt, self._policy_loss, policy.variables),
            (vfn_opt, self._vfn_loss, vfn.variables),
            (qfn_opt, self._qfn_loss, qfn.variables)]
        all_grads = [opt.compute_gradients(loss, var_list=variables) for
                     opt, loss, variables in losses]
        with tf.control_dependencies([
                grad for grads_and_vars in all_grads
                for grad, _ in grads_and_vars]):
            self._optimize = tf.group(*(
                opt.apply_gradients(grads)
                for (opt, _, _), grads in zip(losses, all_grads)))

        # TODO make utility here
        updates = []
        tau = flags().sac.vfn_target_rate
        for current_var, target_var in zip(vfn.variables, target_vars):
            # equivalent lockless target update
            # as in tf.train.ExponentialMovingAverage docs
            with tf.control_dependencies([self._optimize]):
                updates.append(
                    tf.assign_add(
                        target_var, tau * (
                            current_var.read_value() - target_var.read_value())
                    ))
        self._optimize = tf.group(*updates)

        for v in policy.variables + qfn.variables + vfn.variables:
            self._reporter.weights(v.name, v)

        self._policy = policy
        self._vfn = vfn
        self._qfn = qfn
        self._venv = env_info.make_venv(16)

    def _evaluate(self):
        # runs out-of-band trials for less noise performance evaluation
        paths = sample_venv(self._venv, as_controller(self._policy.greedy_act))
        rews = [path.rewards.sum() for path in paths]
        reporter.add_summary_statistics('current policy reward', rews)
        paths = sample_venv(self._venv, as_controller(self._policy.act))
        rews = [path.rewards.sum() for path in paths]
        reporter.add_summary_statistics('exploration policy reward', rews)

    def _sample(self, batch):
        obs, next_obs, rewards, acs, terminals = batch
        feed_dict = {
            self.obs_ph_ns: obs,
            self.next_obs_ph_ns: next_obs,
            self.terminals_ph_n: terminals,
            self.rewards_ph_n: rewards,
            self.actions_ph_na: acs}
        return feed_dict

    def train(self, data, nbatches, batch_size, _):
        """Run nbatches training iterations of SAC"""
        if flags().experiment.should_evaluate():
            if hasattr(self, '_dyn_metrics'):
                self._eval_dynamics(data, 'sac before/')

        batches = data.sample_many(nbatches, batch_size)
        for i, batch in enumerate(batches, 1):
            feed_dict = self._sample(batch)
            tf.get_default_session().run(self._optimize, feed_dict)
            if (i % max(nbatches // 10, 1)) == 0:
                pl, ql, vl = tf.get_default_session().run(
                    [self._policy_loss, self._qfn_loss, self._vfn_loss],
                    feed_dict)
                fmt = '{: ' + str(len(str(nbatches))) + 'd}'
                debug('sac ' + fmt + ' of ' + fmt + ' batches - '
                      'policy loss {:.4g} qfn loss {:.4g} '
                      'vfn loss {:.4g}',
                      i, nbatches, pl, ql, vl)

        if flags().experiment.should_evaluate():
            if data.size:
                batch = self._sample(next(data.sample_many(1, batch_size)))
                self._reporter.report(batch)

            self._evaluate()

            if hasattr(self, '_dyn_metrics'):
                self._eval_dynamics(data, 'sac after/')

    def _eval_dynamics(self, data, prefix):
        # TODO -- abstract common code instead of being this lazy
        from ddpg.ddpg import DDPG
        DDPG._eval_dynamics(self, data, prefix)

def _tf_compute_model_value_expansion(
        obs0_ns,
        acs0_na,
        rew0_n,
        obs1_ns,
        terminals1_n,
        policy,
        qfn,
        vfn,
        dynamics):
    h = flags().sac.model_horizon
    reward_fn = env_info.reward_fn()
    discount = flags().experiment.discount
    temperature = flags().sac.temperature

    # the i-th states along the first axis here (of length h) correspond
    # to the states that occur after the obs1 state (the resulting state
    # from the real-data transitions). Thus at position 0 we have
    # exactly obs1 but at position i we have the predicted state i steps
    # past obs1.
    obs_hns = []
    # the action taken at the i-th step above
    acs_hna = []
    # after playing the corresponding action in the corresponding state
    # we get this reward
    rew_hn = []
    log_probs_hn = []

    # We could use tf.while_loop here, but TF actually handles a
    # static graph much better. Since we know h ahead of time we should use
    # it.

    curr_ob_ns = obs1_ns
    for _ in range(h):
        obs_hns.append(curr_ob_ns)
        ac_na, log_prob_n = policy.tf_sample_action_with_log_prob(curr_ob_ns)
        acs_hna.append(ac_na)
        log_probs_hn.append(log_prob_n)
        next_ob_ns = dynamics.predict_tf(curr_ob_ns, ac_na)
        curr_reward_n = reward_fn(curr_ob_ns, ac_na, next_ob_ns)
        rew_hn.append(curr_reward_n)
        curr_ob_ns = next_ob_ns

    # final_ob_ns should be the final state resulting from playing
    # acs_hna[h-1] on obs_hns[h-1]
    final_ob_ns = curr_ob_ns
    with tf.variable_scope('targets'):
        final_V_n = vfn.tf_state_value(final_ob_ns)

    # like in mve-ddpg, propogate through the network up-front for
    # TensorFlow efficiency (this could've been a python for-loop
    # generating O(h) TF graph nodes)
    n = tf.shape(obs_hns)[1]
    a = env_info.ac_dim()
    s = env_info.ob_dim()
    all_Q_hn = tf.reshape(qfn.tf_state_action_value(
        tf.reshape(obs_hns, [-1, s]),
        tf.reshape(acs_hna, [-1, a])), [h, n])

    # we accumulate error in reverse
    next_V_n = final_V_n
    accum_loss = 0.
    for t in reversed(range(h)):
        target_Q_n = rew_hn[t] + discount * next_V_n
        curr_Q_n = all_Q_hn[t]
        # the dynamics model doesn't predict terminal states
        # so we only need to remove the terminal states from
        # the batch
        weights = 1.0 - terminals1_n
        curr_residual_loss = tf.losses.mean_squared_error(
            target_Q_n, curr_Q_n, weights=weights)
        if not flags().sac.drop_tdk:
            accum_loss += curr_residual_loss
        next_V_n = target_Q_n - log_probs_hn[t] * temperature

    # compute the full-trajectory TD-h error on obs0 now
    target_Q_n = rew0_n + discount * (1 - terminals1_n) * next_V_n
    curr_Q_n = qfn.tf_state_action_value(obs0_ns, acs0_na)
    qloss = accum_loss + tf.losses.mean_squared_error(
        target_Q_n, curr_Q_n)

    # Don't need MVE for V, but we do apply the TD-k trick
    accum_loss = 0.
    if not flags().sac.drop_tdk:
        final_ac_na, final_log_prob_n = policy.tf_sample_action_with_log_prob(
            final_ob_ns)
        final_Q_n = qfn.tf_state_action_value(
            final_ob_ns, final_ac_na)
        weights = 1.0 - terminals1_n
        accum_loss += tf.losses.mean_squared_error(
            final_Q_n - final_log_prob_n * temperature,
            vfn.tf_state_value(final_ob_ns),
            weights=weights)
        all_V_hn = tf.reshape(vfn.tf_state_value(
            tf.reshape(obs_hns, [-1, s])), [h, n])
        all_V_target_hn = all_Q_hn - tf.stack(log_probs_hn) * temperature
        accum_loss += tf.losses.mean_squared_error(
            all_V_target_hn,
            all_V_hn,
            weights=tf.expand_dims(weights, 0)) * h
    onpol_act_na, log_prob_acs_n = policy.tf_sample_action_with_log_prob(
        obs0_ns)
    onpol_qfn_n = qfn.tf_state_action_value(obs0_ns, onpol_act_na)
    vloss = accum_loss + tf.losses.mean_squared_error(
        onpol_qfn_n - temperature * log_prob_acs_n,
        vfn.tf_state_value(obs0_ns))
    return qloss, vloss
