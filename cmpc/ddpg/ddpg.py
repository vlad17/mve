"""DDPG training."""

import tensorflow as tf
import numpy as np

from context import flags
import env_info
from log import debug
from multiprocessing_env import make_venv
import reporter
from sample import sample_venv
from tf_reporter import TFReporter
from qvalues import qvals, offline_oracle_q
from utils import scale_from_box, as_controller


def _incremental_report_name(update_iteration, total_updates):
    name = '{:0' + str(len(str(total_updates))) + 'd}-of-'
    name += str(total_updates) + ' trained/'
    name = 'training ddpg/' + name
    return name.format(update_iteration)


class DDPG:  # pylint: disable=too-many-instance-attributes
    """
    Builds the part of the TF graph responsible for training actor
    and critic networks with the DDPG algorithm
    """

    def __init__(self, actor, critic, discount=0.99, scope='ddpg',
                 actor_lr=1e-3, critic_lr=1e-3, decay=0.99, explore_stddev=0.2,
                 nbatches=1):

        self._reporter = TFReporter()
        self._nbatches = nbatches

        self.obs0_ph_ns = tf.placeholder(
            tf.float32, shape=[None, env_info.ob_dim()])
        self.obs1_ph_ns = tf.placeholder(
            tf.float32, shape=[None, env_info.ob_dim()])
        self.terminals1_ph_n = tf.placeholder(
            tf.float32, shape=[None])
        self.rewards_ph_n = tf.placeholder(
            tf.float32, shape=[None])
        self.actions_ph_na = tf.placeholder(
            tf.float32, shape=[None, env_info.ac_dim()])

        # actor maximizes current Q
        critic_at_actor_n = critic.tf_critic(
            self.obs0_ph_ns,
            actor.tf_action(self.obs0_ph_ns))
        self._reporter.stats('actor Q', critic_at_actor_n)
        self._actor_loss = -1 * tf.reduce_mean(critic_at_actor_n)
        self._reporter.scalar('actor loss', self._actor_loss)
        opt = tf.train.AdamOptimizer(
            learning_rate=actor_lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        with tf.variable_scope(scope):
            with tf.variable_scope('opt_actor'):
                self._reporter.grads(
                    'actor grad', opt, self._actor_loss, actor.variables)
                optimize_actor_op = opt.minimize(
                    self._actor_loss, var_list=actor.variables)

        # critic minimizes TD-1 error wrt to target Q and target actor
        current_Q_n = critic.tf_critic(
            self.obs0_ph_ns, self.actions_ph_na)
        self._reporter.stats('critic Q', current_Q_n)
        next_Q_n = critic.tf_target_critic(
            self.obs1_ph_ns, actor.tf_target_action(self.obs1_ph_ns))
        target_Q_n = self.rewards_ph_n + (1. - self.terminals1_ph_n) * (
            discount * next_Q_n)
        self._reporter.stats('target Q', target_Q_n)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self._critic_loss = tf.losses.mean_squared_error(
            target_Q_n,
            current_Q_n) + reg_loss
        self._reporter.scalar('critic loss', self._critic_loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=critic_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        # perform the critic update after the actor (which is dependent on it)
        # then perform both updates
        with tf.variable_scope(scope):
            with tf.variable_scope('opt_critic'):
                self._reporter.grads(
                    'critic grad', opt, self._critic_loss, critic.variables)
                optimize_critic_op = opt.minimize(
                    self._critic_loss, var_list=critic.variables)
        with tf.control_dependencies([optimize_actor_op, optimize_critic_op]):
            update_targets = tf.group(
                actor.tf_target_update(decay),
                critic.tf_target_update(decay))

        for v in actor.variables + critic.variables:
            self._reporter.weights(v.name, v)

        self._copy_targets = tf.group(
            actor.tf_target_update(0),
            critic.tf_target_update(0))

        # Parameter-noise exploration: "train" a noise variable to learn
        # a perturbation stddev for actor network weights that hits the target
        # action stddev. After every gradient step sample how far our current
        # parameter space noise puts our actions from mean.
        with tf.variable_scope(scope):
            init_stddev = explore_stddev / (actor.depth + 1)
            adaptive_noise = tf.get_variable(
                'adaptive_noise', trainable=False, initializer=init_stddev)
            # sum of action noise we've seen in the current training iteration
            # (summed across mini-batches)
            self._observed_noise_sum = tf.get_variable(
                'observed_noise_sum', trainable=False, initializer=0.)
        self._reporter.scalar('adaptive param noise', adaptive_noise)
        self._reporter.scalar(
            'action noise', self._observed_noise_sum / nbatches)
        # This implementation observes the param noise after every gradient
        # step this isn't absolutely necessary but doesn't seem to be a
        # bottleneck.
        #
        # We then adjust parameter noise by the adaption coefficient
        # once every iteration.
        with tf.control_dependencies([optimize_actor_op]):
            re_perturb = actor.tf_perturb_update(adaptive_noise)
            with tf.control_dependencies([re_perturb]):
                mean_ac = scale_from_box(
                    env_info.ac_space(), actor.tf_action(self.obs0_ph_ns))
                perturb_ac = scale_from_box(
                    env_info.ac_space(), actor.tf_perturbed_action(
                        self.obs0_ph_ns))
                batch_observed_noise = tf.sqrt(
                    tf.reduce_mean(tf.square(mean_ac - perturb_ac)))
                save_noise = tf.assign_add(
                    self._observed_noise_sum, batch_observed_noise)

        self._optimize = tf.group(update_targets, save_noise)
        multiplier = tf.cond(
            self._observed_noise_sum < explore_stddev * nbatches,
            lambda: 1.01, lambda: 1 / 1.01)
        self._update_adapative_noise_op = tf.assign(
            adaptive_noise, adaptive_noise * multiplier)
        self._actor = actor
        self._critic = critic
        self._venv = make_venv(
            flags().experiment.make_env, 10)
        self._assess_venv = make_venv(
            flags().experiment.make_env,
            flags().ddpg.oracle_nenvs_with_default())

    def _evaluate(self, reporting_prefix='', hide=True):
        # runs out-of-band trials for less noise performance evaluation
        paths = sample_venv(self._venv, as_controller(self._actor.target_act))
        rews = [path.rewards.sum() for path in paths]
        reporter.add_summary_statistics(
            reporting_prefix + 'target policy reward', rews, hide=hide)
        acs = np.concatenate([path.acs for path in paths])
        obs = np.concatenate([path.obs for path in paths])
        qs = np.concatenate(qvals(paths, flags().experiment.discount))
        model_horizon = flags().ddpg.model_horizon
        target_qs = self._critic.target_critique(obs, acs)
        qs_estimators = [
            (self._critic.critique(obs, acs), 'critic'),
            (target_qs, 'target'),
            (offline_oracle_q(paths, target_qs, model_horizon),
             'oracle-' + str(model_horizon))]
        for est_qs, name in qs_estimators:
            diffs = est_qs - qs
            if np.all(np.isfinite(diffs)):
                reporter.add_summary_statistics(
                    reporting_prefix + 'Q bias/' + name, diffs, hide=hide)
            qmse = np.square(diffs).mean()
            if np.isfinite(qmse):
                reporter.add_summary(
                    reporting_prefix + 'Q MSE/' + name, qmse, hide=hide)

        paths = sample_venv(self._venv, as_controller(self._actor.act))
        rews = [path.rewards.sum() for path in paths]
        reporter.add_summary_statistics(
            reporting_prefix + 'current policy reward', rews, hide=hide)
        return np.mean(rews)

    def initialize_targets(self):
        """
        New targets are initialized randomly, but they should be initially
        set to equal the initialization of the starting networks.
        """
        debug('copying current network to target for DDPG init')
        tf.get_default_session().run(self._copy_targets)

    def _sample(self, batch):
        obs, next_obs, rewards, acs, terminals = batch
        feed_dict = {
            self.obs0_ph_ns: obs,
            self.obs1_ph_ns: next_obs,
            self.terminals1_ph_n: terminals,
            self.rewards_ph_n: rewards,
            self.actions_ph_na: acs}
        return feed_dict

    def train(self, data, batch_size, num_reports):
        """Run nbatches training iterations of DDPG"""
        itrs_when_we_report = {
            1 if i == 0 else i * self._nbatches // num_reports
            for i in range(num_reports)}

        tf.get_default_session().run(self._observed_noise_sum.initializer)
        batches = data.sample_many(self._nbatches, batch_size)
        for itr, batch in enumerate(batches, 1):
            feed_dict = self._sample(batch)
            tf.get_default_session().run(self._optimize, feed_dict)
            if itr in itrs_when_we_report:
                self._incremental_report(feed_dict, itr)

        tf.get_default_session().run(self._update_adapative_noise_op)

        batch = self._sample(next(data.sample_many(1, batch_size)))
        self._reporter.report(batch)
        self._evaluate(reporting_prefix='', hide=False)

    def _incremental_report(self, feed_dict, itr):
        fmt = 'itr {: 6d} critic loss {:7.3f} actor loss {:7.3f}'
        fmt += ' action noise {:.5g} mean rew {:6.1f}'
        critic_loss, actor_loss, noise_sum = tf.get_default_session().run(
            [self._critic_loss, self._actor_loss, self._observed_noise_sum],
            feed_dict)
        prefix = _incremental_report_name(itr, self._nbatches)
        mean_perf = self._evaluate(prefix, hide=True)
        mean_noise = noise_sum / itr
        debug(fmt, itr, critic_loss, actor_loss, mean_noise, mean_perf)
        reporter.add_summary(prefix + 'actor loss', actor_loss, hide=True)
        reporter.add_summary(prefix + 'critic loss', critic_loss, hide=True)
        reporter.add_summary(prefix + 'action noise', mean_noise, hide=True)
