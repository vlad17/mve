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

    def __init__(self, policy, qfn, vfn):
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
            vfn.tf_state_value(self.next_obs_ph_ns)
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
        if flags().sac.q_target_mixture:
            # TODO: don't forget the entropy estimates when unrolling
            raise ValueError('unsupported option --q_target_mixture true')
        else:

            target_Q_n = self.rewards_ph_n + (1. - self.terminals_ph_n) * (
                discount * next_V_n)
            self._reporter.stats('target Q', target_Q_n)
            self._qfn_loss = tf.losses.mean_squared_error(
                tf.stop_gradient(target_Q_n), current_Q_n)
            self._reporter.scalar('qfn loss', self._qfn_loss)

            onpol_act_na, log_prob_acs_n = policy.tf_sample_action_with_log_prob(
                self.obs_ph_ns)
            onpol_qfn_n = qfn.tf_state_action_value(
                self.obs_ph_ns, onpol_act_na)
            target_V_n = onpol_qfn_n - temperature * log_prob_acs_n
            self._reporter.stats('target V', target_V_n)
            self._vfn_loss = tf.losses.mean_squared_error(
                tf.stop_gradient(target_V_n), current_V_n)
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
