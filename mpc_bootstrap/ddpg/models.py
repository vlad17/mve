"""The various neural networks (policy mean, critics) in DDPG"""

import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    """
    Abstraction for a collection of TF variables that can be trained or
    perturbed.
    """

    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        """All modifiable variables"""
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.name)

    @property
    def trainable_vars(self):
        """All trainable variables"""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.name)

    @property
    def perturbable_vars(self):
        """All perturbable variables"""
        return [var for var in self.trainable_vars
                if 'LayerNorm' not in var.name]


class Actor(Model):
    """The actor (policy mean) network"""

    def __init__(self, nb_actions, name='actor', depth=2, width=64,
                 layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self._depth = depth
        self._width = width

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            for _ in range(self._depth):
                x = tf.layers.dense(x, self._width)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.layers.dense(
                x, self.nb_actions,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    """The critic (value) network"""

    def __init__(self, name='critic', layer_norm=True, width=64, depth=2):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self._width = width
        self._depth = depth

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            for _ in range(self._depth // 2):
                x = tf.layers.dense(x, self._width)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            for _ in range(self._depth - (self._depth // 2)):
                x = tf.layers.dense(x, self._width)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.layers.dense(
                x, 1, kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        """variables produced as output by the model"""
        output_vars = [
            var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
