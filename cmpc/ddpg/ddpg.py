"""DDPG training."""

import tensorflow as tf
import numpy as np

from context import flags
import env_info
from log import debug
import reporter
from sample import sample_venv
from tf_reporter import TFReporter
from qvalues import qvals, offline_oracle_q, oracle_q
from utils import scale_from_box, as_controller
from venv.parallel_venv import ParallelVenv


def _tf_seq(a, b_fn):
    # force a temporal dependence on the evaluation of b after a
    # from Haskell :)
    with tf.control_dependencies([a]):
        return b_fn()


def _tf_doif(cond, if_true_fn):
    return tf.cond(
        cond,
        lambda: _tf_seq(if_true_fn(), lambda: tf.constant(0)),
        lambda: 0)


class DDPG:  # pylint: disable=too-many-instance-attributes
    """
    Builds the part of the TF graph responsible for training actor
    and critic networks with the DDPG algorithm
    """

    def __init__(self, actor, critic, discount=0.99, scope='ddpg',
                 actor_lr=1e-3, critic_lr=1e-3, explore_stddev=0.2):

        self._reporter = TFReporter()

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
        if flags().ddpg.mixture_estimator == 'oracle':
            # h-step observations
            h = flags().ddpg.model_horizon
            debug('using oracle Q estimator with {} steps', h)
            self._oracle_venv = ParallelVenv(
                flags().ddpg.oracle_nenvs_with_default())
            self._target_Q_ph_n = tf.placeholder(
                tf.float32, shape=[None])
            target_Q_n = self._target_Q_ph_n

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self._critic_loss = tf.losses.mean_squared_error(
            target_Q_n, current_Q_n) + reg_loss
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
                actor.tf_target_update(flags().ddpg.actor_decay),
                critic.tf_target_update(flags().ddpg.critic_decay))

        for v in actor.variables + critic.variables:
            self._reporter.weights(v.name, v)

        self._copy_targets = tf.group(
            actor.tf_target_update(0),
            critic.tf_target_update(0))

        # Parameter-noise exploration: "train" a noise variable to learn
        # a perturbation stddev for actor network weights that hits the target
        # action stddev. After every gradient step sample how far our current
        # parameter space noise puts our actions from mean.
        with tf.variable_scope(scope), tf.variable_scope('adaption'):
            init_stddev = explore_stddev / (actor.depth + 1)
            adaptive_noise = tf.get_variable(
                'adaptive_noise', trainable=False, initializer=init_stddev)
            # sum of action noise we've seen in the last few updates
            observed_noise_sum = tf.get_variable(
                'observed_noise_sum', trainable=False, initializer=0.)
            # number of observations in sum
            adaption_update_ctr = tf.get_variable(
                'update_ctr', trainable=False, initializer=0)
            self._action_noise = tf.get_variable(
                'prev_mean_action_noise', trainable=False, initializer=0.)
        self._reporter.scalar('adaptive param noise', adaptive_noise)
        self._reporter.scalar('action noise', self._action_noise)
        # This implementation observes the param noise after every gradient
        # step this isn't absolutely necessary but doesn't seem to be a
        # bottleneck.
        #
        # We then adjust parameter noise by the adaption coefficient
        # once every iteration.
        with tf.control_dependencies([optimize_actor_op]):
            re_perturb = actor.tf_perturb_update(adaptive_noise)
            mean_ac = scale_from_box(
                env_info.ac_space(), actor.tf_action(self.obs0_ph_ns))
            with tf.control_dependencies([re_perturb]):
                perturb_ac = scale_from_box(
                    env_info.ac_space(), actor.tf_perturbed_action(
                        self.obs0_ph_ns))
                batch_observed_noise = tf.sqrt(
                    tf.reduce_mean(tf.square(mean_ac - perturb_ac)))
                save_noise = tf.group(
                    tf.assign_add(observed_noise_sum, batch_observed_noise),
                    tf.assign_add(adaption_update_ctr, 1))
        adaption_interval = flags().ddpg.param_noise_adaption_interval
        adaption_rate = 1.01
        with tf.control_dependencies([save_noise]):
            multiplier = tf.cond(
                observed_noise_sum.read_value() <
                explore_stddev * adaption_interval,
                lambda: adaption_rate, lambda: 1 / adaption_rate)
            adapted_noise = adaptive_noise.read_value() * multiplier
            # this is a bit icky, but if it wasn't for TF it'd be easy to read:
            # if we had adaption_interval updates, then adapt the
            # parameter-space noise and clear the running sum and counters.
            # The lambdas are necessary b/c TF relies on op-creation context
            # to create temporal dependencies.
            conditional_update = _tf_doif(
                tf.equal(adaption_update_ctr.read_value(), adaption_interval),
                lambda: _tf_seq(
                    tf.group(
                        tf.assign(adaptive_noise, adapted_noise),
                        tf.assign(self._action_noise,
                                  observed_noise_sum.read_value() /
                                  tf.to_float(
                                      adaption_update_ctr.read_value()))),
                    lambda: tf.group(
                        tf.assign(observed_noise_sum, 0.),
                        tf.assign(adaption_update_ctr, 0))))

        self._optimize = tf.group(update_targets, conditional_update)
        self._actor = actor
        self._critic = critic
        self._venv = ParallelVenv(10)

    def _evaluate(self):
        # runs out-of-band trials for less noise performance evaluation
        paths = sample_venv(self._venv, as_controller(self._actor.target_act))
        rews = [path.rewards.sum() for path in paths]
        reporter.add_summary_statistics('target policy reward', rews)
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
                reporter.add_summary_statistics('Q bias/' + name, diffs)
            qmse = np.square(diffs).mean()
            if np.isfinite(qmse):
                reporter.add_summary('Q MSE/' + name, qmse)

        paths = sample_venv(self._venv, as_controller(self._actor.act))
        rews = [path.rewards.sum() for path in paths]
        reporter.add_summary_statistics('current policy reward', rews)

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
        if hasattr(self, '_target_Q_ph_n'):
            h = flags().ddpg.model_horizon
            h_n = np.full(len(obs), h, dtype=int)
            model_expanded_Q = oracle_q(
                self._critic.target_critique,
                self._actor.target_act,
                obs, acs, self._oracle_venv, h_n)
            feed_dict[self._target_Q_ph_n] = model_expanded_Q
        return feed_dict

    def train(self, data, nbatches, batch_size):
        """Run nbatches training iterations of DDPG"""
        batches = data.sample_many(nbatches, batch_size)
        for i, batch in enumerate(batches, 1):
            feed_dict = self._sample(batch)
            tf.get_default_session().run(self._optimize, feed_dict)
            if (i % max(nbatches // 10, 1)) == 0:
                cl, al, an = tf.get_default_session().run(
                    [self._critic_loss, self._actor_loss, self._action_noise],
                    feed_dict)
                fmt = '{: ' + str(len(str(nbatches))) + 'd}'
                debug('ddpg ' + fmt + ' of ' + fmt + ' batches - '
                      'critic loss {:.4g} actor loss {:.4g} '
                      'action noise {:.4g}',
                      i, nbatches, cl, al, an)

        if data.size:
            batch = self._sample(next(data.sample_many(1, batch_size)))
            self._reporter.report(batch)

        self._evaluate()
