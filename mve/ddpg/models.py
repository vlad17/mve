"""
The various neural networks (policy mean, critics) in DDPG.

The only implementation differences from OpenAI is that we use a
simple MLP for the critic network instead of some weird multi-level
thing.
"""

import tensorflow as tf

import env_info
from memory import scale_acs, unscale_acs
from utils import build_mlp


def _layer_norm(x):
    return tf.contrib.layers.layer_norm(x, center=True, scale=True)


def trainable_vars(parent_scope, child_scope):
    # TODO de-dup with utils.trainable_vars
    """Returns all trainable variables within the nested scope."""
    with tf.variable_scope(parent_scope):
        with tf.variable_scope(child_scope):
            scope = tf.get_variable_scope().name
    collection = tf.GraphKeys.TRAINABLE_VARIABLES
    return tf.get_collection(collection, scope)


def _target_updates(parent_scope, current_scope, target_scope, rate):
    current_vars = trainable_vars(parent_scope, current_scope)
    target_vars = trainable_vars(parent_scope, target_scope)
    assert len(current_vars) == len(target_vars), (current_vars, target_vars)
    # variables should have been created the same way, so they'll
    # be ordered correctly.
    updates = []
    for current_var, target_var in zip(current_vars, target_vars):
        # equivalent lockless target update
        # as in tf.train.ExponentialMovingAverage docs
        updates.append(
            tf.assign_add(
                target_var, rate * (
                    current_var.read_value() - target_var.read_value())))
    return tf.group(*updates)


def _perturb_update(parent_scope, current_scope, perturb_scope, noise):
    current_vars = trainable_vars(parent_scope, current_scope)
    perturb_vars = trainable_vars(parent_scope, perturb_scope)
    assert len(current_vars) == len(perturb_vars), (current_vars, perturb_vars)
    updates = []
    for current_var, perturb_var in zip(current_vars, perturb_vars):
        if 'LayerNorm' in current_var.name:
            perturbation = 0
        else:
            perturbation = noise * tf.random_normal(
                tf.shape(current_var), mean=0., stddev=1.)
        updates.append(
            tf.assign(perturb_var, current_var + perturbation))
    return tf.group(*updates)


class Actor:
    """
    The actor (policy mean) network. Has a variables attribute
    for optimizable variables.

    Also has a "perturbed" version for exploration, which uses
    noise as specified by the noise parameter, an instance of
    AdaptiveNoise.
    """

    def __init__(self, scope='ddpg', depth=2, width=64):
        self.depth = depth
        self._common_mlp_kwargs = {
            'output_size': env_info.ac_dim(),
            'n_layers': depth,
            'size': width,
            'activation': tf.nn.relu,
            'reuse': None,
            'output_activation': tf.nn.sigmoid,
            'activation_norm': _layer_norm}
        self._scope = scope

        self._states_ph_ns = tf.placeholder(
            tf.float32, [None, env_info.ob_dim()])
        # result unimportant, just generate the corresponding graph variables
        self._acs_na = self.tf_action(self._states_ph_ns)
        self._target_acs_na = self.tf_target_action(self._states_ph_ns)
        self._perturb_acs_na = self.tf_perturbed_action(self._states_ph_ns)

        self.variables = trainable_vars(self._scope, 'actor')

    def tf_target_update(self, rate):
        """Create and return an op to do target updates"""
        return _target_updates(
            self._scope, 'actor', 'target_actor', rate)

    def tf_perturb_update(self, noise):
        """
        Create an op that updates the perturbed actor network with a
        perturbed version of the current actor network.
        """
        return _perturb_update(self._scope, 'actor', 'perturbed_actor', noise)

    def act(self, states_ns):
        """
        Return a numpy array with the current (mean) actor's actions.
        """
        return tf.get_default_session().run(self._acs_na, feed_dict={
            self._states_ph_ns: states_ns})

    def target_act(self, states_ns):
        """
        Return a numpy array with the target actor's actions.
        """
        return tf.get_default_session().run(self._target_acs_na, feed_dict={
            self._states_ph_ns: states_ns})

    def perturbed_act(self, states_ns):
        """
        Return a numpy array with the current (perturbed) actor's actions.
        """
        return tf.get_default_session().run(self._perturb_acs_na, feed_dict={
            self._states_ph_ns: states_ns})

    def tf_action(self, states_ns):
        """Return the current actor's actions for the given states in TF."""
        return self._tf_action(states_ns, 'actor')

    def tf_perturbed_action(self, states_ns):
        """Return the current perturbed actor's actions for the given states"""
        return self._tf_action(states_ns, 'perturbed_actor')

    def tf_target_action(self, states_ns):
        """Return the target actor's actions for the given states in TF."""
        return self._tf_action(states_ns, 'target_actor')

    def _tf_action(self, states_ns, child_scope):
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            relative_acs = build_mlp(
                states_ns, scope=child_scope, **self._common_mlp_kwargs)
        return unscale_acs(relative_acs)


class Critic:
    """
    The critic (value) network. Has a variables attribute for
    optimizable variables.
    """

    def __init__(self, scope='ddpg', width=64, depth=2, l2reg=0):
        self._common_mlp_kwargs = {
            'output_size': 1,
            'size': width,
            'n_layers': depth,
            'activation': tf.nn.relu,
            'activation_norm': _layer_norm,
            'output_activation': None,
            'l2reg': l2reg,
            'reuse': None}
        self._scope = scope

        self._states_ph_ns = tf.placeholder(
            tf.float32, [None, env_info.ob_dim()])
        self._acs_ph_na = tf.placeholder(
            tf.float32, [None, env_info.ac_dim()])
        self._q_n = self.tf_critic(self._states_ph_ns, self._acs_ph_na)
        self._target_q_n = self.tf_target_critic(
            self._states_ph_ns, self._acs_ph_na)

        self.variables = trainable_vars(self._scope, 'critic')

    def tf_target_update(self, rate):
        """Create and return an op to do target updates"""
        return _target_updates(
            self._scope, 'critic', 'target_critic', rate)

    def critique(self, states_ns, acs_na):
        """Return current Q-value estimates"""
        return tf.get_default_session().run(self._q_n, feed_dict={
            self._states_ph_ns: states_ns,
            self._acs_ph_na: acs_na})

    def target_critique(self, states_ns, acs_na):
        """Return current Q-value estimates"""
        return tf.get_default_session().run(self._target_q_n, feed_dict={
            self._states_ph_ns: states_ns,
            self._acs_ph_na: acs_na})

    def _tf_critic(self, states_ns, acs_na, scope):
        acs_na = scale_acs(acs_na)
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            inputs = tf.concat([states_ns, acs_na], axis=1)
            out = build_mlp(inputs, scope=scope, **self._common_mlp_kwargs)
        return tf.squeeze(out, axis=1)

    def tf_critic(self, states_ns, acs_na):
        """Return the current critic's Q-valuation."""
        return self._tf_critic(states_ns, acs_na, 'critic')

    def tf_target_critic(self, states_ns, acs_na):
        """Return the target critic's Q-valuation."""
        return self._tf_critic(states_ns, acs_na, 'target_critic')
