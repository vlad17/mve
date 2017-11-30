# pylint: skip-file

from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from log import debug
from ddpg.util import reduce_std


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def get_target_updates(vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars), ((vars), (target_vars))
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(
            tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            updates.append(tf.assign(
                perturbed_var, var + tf.random_normal(
                    tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class DDPG(object):
    def __init__(self, actor, critic, observation_shape, action_shape,
                 param_noise=None, action_noise=None,
                 gamma=0.99, tau=0.001, normalize_returns=False,
                 enable_popart=False, normalize_observations=False,
                 batch_size=128, observation_range=(-5., 5.),
                 action_range=(-1., 1.), return_range=(-np.inf, np.inf),
                 adaptive_param_noise=True,
                 adaptive_param_noise_policy_threshold=.1,
                 critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3,
                 clip_norm=None):
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(
            None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(
            None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(
            tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(
            tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(
            None,) + action_shape, name='actions')
        self.critic_target = tf.placeholder(
            tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(
            tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        self.obs_rms = None
        self.ret_rms = None

        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
                                           self.observation_range[0],
                                           self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
                                           self.observation_range[0],
                                           self.observation_range[1])

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(normalized_obs0)
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        self.critic_tf = denormalize(tf.clip_by_value(
            self.normalized_critic_tf, self.return_range[0],
            self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(
            normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(
            self.normalized_critic_with_actor_tf, self.return_range[0],
            self.return_range[1]), self.ret_rms)
        Q_obs1 = denormalize(target_critic(
            normalized_obs1, target_actor(normalized_obs1)), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)
        # target updates before optimizer, which adds vars
        self.setup_target_network_updates()
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(
            self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(
            self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
        self.param_noise_actor = param_noise_actor
        self.perturb_policy_ops = get_perturbed_actor_updates(
            self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(
            self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(
            tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        debug('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list()
                        for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape)
                               for shape in actor_shapes])
        debug('  actor shapes: {}', (actor_shapes))
        debug('  actor params: {}', (actor_nb_params))
        opt = tf.train.AdamOptimizer(
            learning_rate=self.actor_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.actor_optimize = opt.minimize(
            self.actor_loss, var_list=self.actor.trainable_vars)

    def setup_critic_optimizer(self):
        debug('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(
            self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(
            tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [
                var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                debug('  regularizing: {}', (var.name))
            debug('  applying l2 regularization with {}', (self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list()
                         for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape)
                                for shape in critic_shapes])
        debug('  critic shapes: {}', (critic_shapes))
        debug('  critic params: {}', (critic_nb_params))
        opt = tf.train.AdamOptimizer(
            learning_rate=self.critic_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.critic_optimize = opt.minimize(
            self.critic_loss, var_list=self.critic.trainable_vars)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [
                M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [
                b.assign((b * self.old_std + self.old_mean - new_mean)
                         / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean),
                    tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def pi(self, obs, apply_noise=True, compute_Q=True):
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf
        feed_dict = {self.obs0: obs}
        if compute_Q:
            action, q = self.sess.run(
                [actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def update_obs_norm_stats(self, obs0):
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def train(self, batch):
        if self.normalize_returns and self.enable_popart:
            assert False, 'normalizing returns currently unsupported'
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean,
                                                         self.ret_rms.std,
                                                         self.target_Q],
                                                        feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std: np.array([old_std]),
                self.old_mean: np.array([old_mean]),
            })
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get all gradients and perform a synced update.
        ops = [self.actor_optimize, self.actor_loss,
               self.critic_optimize, self.critic_loss]
        _, actor_loss, _, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })

        return critic_loss, actor_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def adapt_param_noise(self, batch):
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = np.mean(distance)
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def tf_n_perturbed(self, states_ns):
        n = tf.shape(states_ns)[0]
        initial_acs = tf.zeros([n, self.actor.nb_actions])
        # create a local var (not initialized) based off the states
        # w/ the corresponding action for each state, computed in a
        # while loop.
        acs_na = tf.get_variable('ddpg_acs_na',
                                 initializer=initial_acs,
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                 dtype=tf.float32,
                                 trainable=False,
                                 validate_shape=False)
        self.acs_initializer = acs_na.initializer

        def body(i):
            updates = []
            for var, perturbed_var in zip(
                    self.actor.vars, self.param_noise_actor.vars):
                if var in self.actor.perturbable_vars:
                    updates.append(tf.assign(
                        perturbed_var, var + tf.random_normal(
                            tf.shape(var), mean=0.,
                            stddev=self.param_noise_stddev)))
            with tf.control_dependencies(updates):
                ac_a = self.param_noise_actor(
                    tf.expand_dims(states_ns[i], axis=0),
                    reuse=True)[0]
            update = tf.scatter_update(
                acs_na, i, ac_a)
            with tf.control_dependencies([update]):
                return i + 1
        update_acs = tf.while_loop(lambda i: i < n, body, loop_vars=[0])
        with tf.control_dependencies([update_acs]):
            return acs_na
