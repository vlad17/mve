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
            nbatches=1, gpu_dataset=None, batch_size=512):

        self._debugs = []

        # Create a version of the graph for debugging
        self._debug_size = tf.placeholder(tf.int32, [])
        debug_ix = gpu_dataset.tf_sample_uniform(self._debug_size)
        obs_ns, next_obs_ns, rewards_n, acs_na, terminals_n = (
            gpu_dataset.get_batch(debug_ix))

        # actor maximizes current Q
        normalized_critic_at_actor_n = critic.tf_critic(
            obs_ns,
            actor.tf_action(obs_ns))
        self._debug_stats('actor Q', normalized_critic_at_actor_n)
        self._actor_loss = -1 * tf.reduce_mean(normalized_critic_at_actor_n)
        self._debug_scalar('actor loss', self._actor_loss)
        opt = tf.train.AdamOptimizer(
            learning_rate=actor_lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self._debug_grads(
            'actor grad', opt, self._actor_loss, actor.variables)

        # critic minimizes TD-1 error wrt to target Q and target actor
        current_Q_n = critic.tf_critic(obs_ns, acs_na)
        self._debug_stats('critic Q', current_Q_n)
        next_Q_n = critic.tf_target_critic(
            next_obs_ns, actor.tf_target_action(next_obs_ns))
        target_Q_n = rewards_n + (1. - terminals_n) * (
            discount * next_Q_n)
        self._debug_stats('target Q', target_Q_n)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self._critic_loss = tf.losses.mean_squared_error(
            target_Q_n,
            current_Q_n) + reg_loss
        self._debug_scalar('critic loss', self._critic_loss)
        self._debug_grads(
            'critic grad', opt, self._critic_loss, critic.variables)

        for v in actor.variables + critic.variables:
            self._debug_weights(v.name, v)

        self._copy_targets = tf.group(
            actor.tf_target_update(0),
            critic.tf_target_update(0))

        # code for actually performing multiple mini-batch steps (note it's
        # different from the debug part of the graph above)
        #
        # parameter-noise exploration: "train" a noise variable to learn
        # a perturbation stddev for actor network weights that hits the target
        # action stddev. After every gradient step sample how far our current
        # parameter space noise puts our actions from mean.
        with tf.variable_scope(scope):
            adaptive_noise = tf.get_variable(
                'adaptive_noise', trainable=False, initializer=explore_stddev)
        self._debug_scalar('adaptive param noise', adaptive_noise)
        # to make adaptive noise approximately equal to the desired level,
        # explore_stddev, we monitor what the noise of a perturbed actor
        # looks like over the course of training with the current
        # adaptive_noise level, and adjust that as necessary
        #
        # This implementation observes the param noise after every gradient
        # step this isn't absolutely necessary but doesn't seem to be a
        # bottleneck.
        #
        # We then adjust parameter noise by the adaption coefficient
        # once every iteration.

        def _train_loop(i, observed_noise_sum):
            batch_ix = gpu_dataset.tf_sample_uniform(batch_size)
            obs_ns, next_obs_ns, rewards_n, acs_na, terminals_n = (
                gpu_dataset.get_batch(batch_ix))
            actor_loss = -1 * tf.reduce_mean(critic.tf_critic(
                obs_ns, actor.tf_action(obs_ns)))
            current_Q_n = tf.squeeze(
                critic.tf_critic(obs_ns, acs_na), axis=1)
            next_Q_n = tf.squeeze(critic.tf_target_critic(
                next_obs_ns, actor.tf_target_action(next_obs_ns)), axis=1)
            target_Q_n = rewards_n + (1. - terminals_n) * (
                discount * next_Q_n)
            reg_loss = sum(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))
            critic_loss = tf.losses.mean_squared_error(
                target_Q_n,
                current_Q_n) + reg_loss
            with tf.variable_scope(scope):
                with tf.variable_scope('opt_actor'):
                    optimize_actor_op = opt.minimize(
                        actor_loss, var_list=actor.variables)
                with tf.variable_scope('opt_critic'):
                    optimize_critic_op = opt.minimize(
                        critic_loss, var_list=critic.variables)
            with tf.control_dependencies([
                    optimize_actor_op, optimize_critic_op]):
                update_targets = tf.group(
                    actor.tf_target_update(decay),
                    critic.tf_target_update(decay))
            with tf.control_dependencies([optimize_actor_op]):
                re_perturb = actor.tf_perturb_update(adaptive_noise)
                with tf.control_dependencies([re_perturb]):
                    mean_ac = scale_from_box(
                        env.action_space, actor.tf_action(obs_ns))
                    perturb_ac = scale_from_box(
                        env.action_space, actor.tf_perturbed_action(obs_ns))
                    batch_observed_noise = tf.sqrt(
                        tf.reduce_mean(tf.square(mean_ac - perturb_ac)))
            with tf.control_dependencies([update_targets]):
                return i + 1, batch_observed_noise + observed_noise_sum

        _, noise_sum = tf.while_loop(lambda t, _: t < nbatches, _train_loop,
                                     [0, 0.], back_prop=False)
        noise = noise_sum / nbatches
        multiplier = tf.cond(noise < explore_stddev,
                             lambda: 1.01, lambda: 1 / 1.01)
        self._optimize = tf.assign(
            adaptive_noise, adaptive_noise * multiplier)

    def initialize_targets(self):
        """
        New targets are initialized randomly, but they should be initially
        set to equal the initialization of the starting networks.
        """
        debug('copying current network to target for DDPG init')
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

    def train(self):
        """Run nbatches training iterations of DDPG"""

        tf.get_default_session().run(self._optimize)
        self._debug_print()

    def _debug_print(self):
        feed_dict = {self._debug_size: 1024}
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
