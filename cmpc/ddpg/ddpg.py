"""DDPG training."""

import tensorflow as tf
import numpy as np

from log import debug
import reporter
from utils import get_ob_dim, get_ac_dim, scale_from_box


def _flatgrad_norms(opt, loss, variables):
    grads_and_vars = opt.compute_gradients(loss, var_list=variables)
    grads = [grad for grad, var in grads_and_vars if var is not None]
    flats = [tf.reshape(x, [-1]) for x in grads]
    grad = tf.concat(flats, axis=0)
    l2 = tf.norm(grad, ord=2)
    linf = tf.norm(grad, ord=np.inf)
    return l2, linf


class DDPG:
    """
    Builds the part of the TF graph responsible for training actor
    and critic networks with the DDPG algorithm
    """

    def __init__(
            self, env, actor, critic, discount=0.99, scope='ddpg',
            actor_lr=1e-4, critic_lr=1e-3, decay=0.99, explore_stddev=0.2,
            nbatches=1):

        self._debugs = []
        self._nbatches = nbatches

        self.obs0_ph_ns = tf.placeholder(
            tf.float32, shape=[None, get_ob_dim(env)])
        self.obs1_ph_ns = tf.placeholder(
            tf.float32, shape=[None, get_ob_dim(env)])
        self.terminals1_ph_n = tf.placeholder(
            tf.float32, shape=[None])
        self.rewards_ph_n = tf.placeholder(
            tf.float32, shape=[None])
        self.actions_ph_na = tf.placeholder(
            tf.float32, shape=[None, get_ac_dim(env)])

        # actor maximizes current Q
        normalized_critic_at_actor_n = critic.tf_critic(
            self.obs0_ph_ns,
            actor.tf_action(self.obs0_ph_ns))
        self._debug_stats('actor Q', normalized_critic_at_actor_n)
        self._actor_loss = -1 * tf.reduce_mean(normalized_critic_at_actor_n)
        self._debug_scalar('actor loss', self._actor_loss)
        opt = tf.train.AdamOptimizer(
            learning_rate=actor_lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        with tf.variable_scope(scope):
            with tf.variable_scope('opt_actor'):
                self._debug_grads(
                    'actor grad', opt, self._actor_loss, actor.variables)
                optimize_actor_op = opt.minimize(
                    self._actor_loss, var_list=actor.variables)

        # critic minimizes TD-1 error wrt to target Q and target actor
        current_Q_n = critic.tf_critic(
            self.obs0_ph_ns, self.actions_ph_na)
        self._debug_stats('critic Q', current_Q_n)
        next_Q_n = critic.tf_target_critic(
            self.obs1_ph_ns, actor.tf_target_action(self.obs1_ph_ns))
        target_Q_n = self.rewards_ph_n + (1. - self.terminals1_ph_n) * (
            discount * next_Q_n)
        self._debug_stats('target Q', target_Q_n)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self._critic_loss = tf.losses.mean_squared_error(
            target_Q_n,
            current_Q_n) + reg_loss
        self._debug_scalar('critic loss', self._critic_loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=critic_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        # perform the critic update after the actor (which is dependent on it)
        # then perform both updates
        with tf.control_dependencies([optimize_actor_op]):
            with tf.variable_scope(scope):
                with tf.variable_scope('opt_critic'):
                    self._debug_grads(
                        'critic grad', opt,
                        self._critic_loss, critic.variables)
                    optimize_critic_op = opt.minimize(
                        self._critic_loss, var_list=critic.variables)
            with tf.control_dependencies([optimize_critic_op]):
                update_targets = tf.group(
                    actor.tf_target_update(decay),
                    critic.tf_target_update(decay))

        for v in actor.variables + critic.variables:
            self._debug_weights(v.name, v)

        self._copy_targets = tf.group(
            actor.tf_target_update(0),
            critic.tf_target_update(0))

        # Parameter-noise exploration: "train" a noise variable to learn
        # a perturbation stddev for actor network weights that hits the target
        # action stddev. After every gradient step sample how far our current
        # parameter space noise puts our actions from mean.
        with tf.variable_scope(scope):
            adaptive_noise = tf.get_variable(
                'adaptive_noise', trainable=False, initializer=explore_stddev)
            # sum of action noise we've seen in the current training iteration
            # (summed across mini-batches)
            self._observed_noise_sum = tf.get_variable(
                'observed_noise_sum', trainable=False, initializer=0.)
        self._debug_scalar('adaptive param noise', adaptive_noise)
        self._debug_scalar('action noise', self._observed_noise_sum / nbatches)
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
                    env.action_space, actor.tf_action(self.obs0_ph_ns))
                perturb_ac = scale_from_box(
                    env.action_space, actor.tf_perturbed_action(
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

    def initialize_targets(self):
        """
        New targets are initialized randomly, but they should be initially
        set to equal the initialization of the starting networks.
        """
        tf.get_default_session().run(self._copy_targets)

    def _debug_stats(self, name, tensor):
        flat = tf.reshape(tensor, [-1])
        self._debugs.append(('stats', name, flat))

    def _debug_scalar(self, name, tensor):
        self._debugs.append(('scalar', name, tensor))

    def _debug_grads(self, name, opt, loss, variables):
        grad_l2, grad_linf = _flatgrad_norms(opt, loss, variables)
        self._debug_scalar(name + ' l2', grad_l2)
        self._debug_scalar(name + ' linf', grad_linf)

    def _debug_weights(self, name, weights):
        flat = tf.reshape(weights, [-1])
        ave_magnitude = tf.reduce_mean(tf.abs(flat))
        self._debugs.append(('weight', name, ave_magnitude))

    def train(self, data, batch_size):
        """Run nbatches training iterations of DDPG"""
        nprints = 5
        nbatches = self._nbatches
        period = max(nbatches // nprints, 1)
        feed_dict = None

        tf.get_default_session().run(self._observed_noise_sum.initializer)

        for itr, batch in enumerate(data.sample_many(
                nbatches, batch_size)):
            obs, next_obs, rewards, acs, terminals = batch
            # popart would go here
            feed_dict = {
                self.obs0_ph_ns: obs,
                self.obs1_ph_ns: next_obs,
                self.terminals1_ph_n: terminals,
                self.rewards_ph_n: rewards,
                self.actions_ph_na: acs}

            tf.get_default_session().run(self._optimize, feed_dict)

            if itr == 0 or itr + 1 == batch_size or (itr + 1) % period == 0:
                fmt = 'itr {: 6d} critic loss {:7.3f} actor loss {:7.3f}'
                fmt += ' action noise {:.5g}'
                sess = tf.get_default_session()
                critic_loss, actor_loss, noise_sum = sess.run(
                    [self._critic_loss, self._actor_loss,
                     self._observed_noise_sum], feed_dict)
                debug(fmt, itr + 1, critic_loss, actor_loss,
                      noise_sum / (itr + 1))

        tf.get_default_session().run(self._update_adapative_noise_op)

        self._debug_print(data)

    def _debug_print(self, data):
        feed_dict = {
            self.obs0_ph_ns: data.obs,
            self.obs1_ph_ns: data.next_obs,
            self.terminals1_ph_n: data.terminals,
            self.rewards_ph_n: data.rewards,
            self.actions_ph_na: data.acs}
        debug_types, names, tensors = zip(*self._debugs)
        values = tf.get_default_session().run(tensors, feed_dict)
        for debug_type, name, value in zip(debug_types, names, values):
            if debug_type == 'scalar':
                reporter.add_summary(name, value)
            elif debug_type == 'stats':
                reporter.add_summary_statistics(name, value)
            elif debug_type == 'weight':
                reporter.add_summary(name, value, hide=True)
            else:
                raise ValueError('unknown debug_type {}'.format(debug_type))
