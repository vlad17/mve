"""
The various neural networks (policy mean, critics) in DDPG.

Implementation differences from OpenAI, done for simplicity:

* Initializing the final layer with the default Glorot initializer
  instead of the initialization to fixed magnitude 3e-3.
* Weird network structure for critic removed (now simple MLP)
* We don't initialize the target network to equal the current network
"""

import tensorflow as tf

from utils import (build_mlp, get_ac_dim, get_ob_dim,
                   scale_to_box, scale_from_box)


def _layer_norm(x):
    return tf.contrib.layers.layer_norm(x, center=True, scale=True)


def _trainable_vars(parent_scope, child_scope):
    with tf.variable_scope(parent_scope):
        with tf.variable_scope(child_scope):
            scope = tf.get_variable_scope().name
    collection = tf.GraphKeys.TRAINABLE_VARIABLES
    return tf.get_collection(collection, scope)


def _target_updates(parent_scope, current_scope, target_scope, decay):
    current_vars = _trainable_vars(parent_scope, current_scope)
    target_vars = _trainable_vars(parent_scope, target_scope)
    assert len(current_vars) == len(target_vars), (current_vars, target_vars)
    # variables should have been created the same way, so they'll
    # be ordered correctly.
    updates = []
    for current_var, target_var in zip(current_vars, target_vars):
        # equivalent lockless target update
        # as in tf.train.ExponentialMovingAverage docs
        updates.append(
            tf.assign_sub(
                target_var, (1 - decay) * (target_var - current_var)))
    return tf.group(*updates)


def _perturb_update(parent_scope, current_scope, perturb_scope, noise):
    current_vars = _trainable_vars(parent_scope, current_scope)
    perturb_vars = _trainable_vars(parent_scope, perturb_scope)
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

    def __init__(self, env, scope='ddpg', depth=2, width=64):
        self._ac_space = env.action_space
        self._ob_space = env.observation_space
        self._common_mlp_kwargs = {
            'output_size': get_ac_dim(env),
            'n_layers': depth,
            'size': width,
            'activation': tf.nn.relu,
            'reuse': None,
            'output_activation': tf.nn.sigmoid,
            'activation_norm': _layer_norm}
        self._scope = scope

        self._states_ph_ns = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)])
        # result unimportant, just generate the corresponding graph variables
        self.tf_action(self._states_ph_ns)
        self.tf_target_action(self._states_ph_ns)
        self._acs_na = self.tf_perturbed_action(self._states_ph_ns)

        self.variables = _trainable_vars(self._scope, 'actor')

    def tf_target_update(self, decay):
        """Create and return an op to do target updates"""
        return _target_updates(
            self._scope, 'actor', 'target_actor', decay)

    def tf_perturb_update(self, noise):
        """
        Create an op that updates the perturbed actor network with a
        perturbed version of the current actor network.
        """
        return _perturb_update(self._scope, 'actor', 'perturbed_actor', noise)

    def act(self, states_ns):
        """
        Return a numpy array with the current (perturbed) actor's actions.
        """
        return tf.get_default_session().run(self._acs_na, feed_dict={
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
        states_ns = scale_from_box(self._ob_space, states_ns)
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            relative_acs = build_mlp(
                states_ns, scope=child_scope, **self._common_mlp_kwargs)
        return scale_to_box(self._ac_space, relative_acs)


class Critic:
    """
    The critic (value) network. Has a variables attribute for
    optimizable variables.
    """

    def __init__(self, env, scope='ddpg', width=64, depth=2, l2reg=0):
        self._ac_space = env.action_space
        self._ob_space = env.observation_space
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

        # we don't care about the critic results here, but need
        # TensorFlow to generate the corresponding graph variables
        states_ph_ns = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)])
        acs_ph_ns = tf.placeholder(
            tf.float32, [None, get_ac_dim(env)])
        self.tf_critic(states_ph_ns, acs_ph_ns)
        self.tf_target_critic(states_ph_ns, acs_ph_ns)

        self.variables = _trainable_vars(self._scope, 'critic')

    def tf_target_update(self, decay):
        """Create and return an op to do target updates"""
        return _target_updates(
            self._scope, 'critic', 'target_critic', decay)

    def _tf_critic(self, states_ns, acs_na, scope):
        states_ns = scale_from_box(self._ob_space, states_ns)
        acs_na = scale_from_box(self._ac_space, acs_na)
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            inputs = tf.concat([states_ns, acs_na], axis=1)
            out = build_mlp(inputs, scope=scope, **self._common_mlp_kwargs)
        return out

    def tf_critic(self, states_ns, acs_na):
        """Return the current critic's Q-valuation."""
        return self._tf_critic(states_ns, acs_na, 'critic')

    def tf_target_critic(self, states_ns, acs_na):
        """Return the target critic's Q-valuation."""
        return self._tf_critic(states_ns, acs_na, 'target_critic')
