"""
Parameterized value functions without targets.
"""

import tensorflow as tf

from context import flags
import env_info
from memory import scale_acs
from utils import build_mlp, trainable_vars


class QFunction:
    """State-action-value function."""

    def __init__(self, scope='sac/qfn'):
        self._states_ph_ns = tf.placeholder(
            tf.float32, [None, env_info.ob_dim()])
        self._acs_ph_na = tf.placeholder(
            tf.float32, [None, env_info.ac_dim()])
        self._scope = scope

        self._value_n = self.tf_state_action_value(
            self._states_ph_ns, self._acs_ph_na)
        self.variables = trainable_vars(self._scope)

    def tf_state_action_value(self, obs_ns, acs_na):
        """TF tensor for the Q value at given states, actions"""
        acs_na = scale_acs(acs_na)
        cat = tf.concat([obs_ns, acs_na], axis=1)
        q_n1 = build_mlp(
            cat, scope=self._scope,
            output_size=1,
            n_layers=flags().sac.learner_depth,
            size=flags().sac.learner_width,
            activation=tf.nn.relu,
            reuse=tf.AUTO_REUSE)
        return tf.squeeze(q_n1, axis=1)

    def state_action_value(self, obs_ns, acs_na):
        """Eager numpy version of tf_state_action_value"""
        return tf.get_default_session().run(self._value_n, feed_dict={
            self._states_ph_ns: obs_ns,
            self._acs_ph_na: acs_na})


class VFunction:
    """State-value function."""

    def __init__(self, scope='sac/vfn'):
        self._states_ph_ns = tf.placeholder(
            tf.float32, [None, env_info.ob_dim()])
        self._scope = scope

        self._value_n = self.tf_state_value(self._states_ph_ns)
        self.variables = trainable_vars(self._scope)

    def tf_state_value(self, obs_ns):
        """TF tensor for the V value at given states"""
        v_n1 = build_mlp(
            obs_ns, scope=self._scope,
            output_size=1,
            n_layers=flags().sac.learner_depth,
            size=flags().sac.learner_width,
            activation=tf.nn.relu,
            reuse=tf.AUTO_REUSE)
        return tf.squeeze(v_n1, axis=1)

    def state_value(self, obs_ns):
        """Eager numpy version of tf_state_value"""
        return tf.get_default_session().run(self._value_n, feed_dict={
            self._states_ph_ns: obs_ns})
