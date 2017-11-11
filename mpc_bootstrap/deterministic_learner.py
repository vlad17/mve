"""
Imagine we generate a big set of rollouts using some policy. This gives us a
big set of states labeled with their corresponding actions. A
DeterministicLearner is a more or less a neural network that uses this data to
predict an action for a given state.

A DeterministicLearner learner is, well, deterministic. That is given a
DeterministicLearner `learner`, `learner.tf_action(states_ns,
is_initial=False)` is purely deterministic. Calling `tf_action(states_ns,
is_initial=True)` will return an action that is randomly perturbed from
`tf_action(states_ns, is_initial=False)`.
"""


import tensorflow as tf
import numpy as np

from learner import Learner
from utils import (build_mlp, create_random_tf_policy, get_ac_dim, get_ob_dim)


# TODO: reduce the number of instance attributes here
class DeterministicLearner(Learner):  # pylint: disable=too-many-instance-attributes
    """Only noisy on first action."""

    def __init__(self,
                 env,
                 learning_rate=None,
                 depth=None,
                 width=None,
                 batch_size=None,
                 epochs=None,
                 explore_std=0,  # 0 means use uniform exploration, >0 normal
                 sess=None):
        self.sess = sess
        self.batch_size = batch_size
        self.epochs = epochs
        self.explore_std = explore_std
        self.ac_dim = get_ac_dim(env)
        self.width = width
        self.depth = depth
        self.ac_space = env.action_space

        # create placeholder for training an MPC learner
        # a = action dim
        # s = state dim
        # n = batch size
        self.input_state_ph_ns = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)])
        self.policy_action_na = self._exploit_policy(
            self.input_state_ph_ns, reuse=None)
        self.expert_action_ph_na = tf.placeholder(
            tf.float32, [None, self.ac_dim])
        mse = tf.losses.mean_squared_error(
            self.expert_action_ph_na,
            self.policy_action_na)

        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(mse)

    def _exploit_policy(self, states_ns, reuse=True):
        ac_na = build_mlp(
            states_ns, scope='learner_policy_mean',
            n_layers=self.depth, size=self.width, activation=tf.nn.relu,
            output_activation=tf.sigmoid, reuse=reuse)
        ac_na *= self.ac_space.high - self.ac_space.low
        ac_na += self.ac_space.low
        return ac_na

    def _explore_policy(self, state_ns):
        if self.explore_std == 0:
            random_policy = create_random_tf_policy(self.ac_space)
            return random_policy(state_ns)

        ac_na = self._exploit_policy(state_ns, reuse=True)
        ac_width = self.ac_space.high - self.ac_space.low
        std_a = tf.constant(ac_width * self.explore_std, tf.float32)
        perturb_na = tf.random_normal([tf.shape(state_ns)[0], self.ac_dim])
        perturb_na *= std_a
        ac_na += perturb_na
        ac_na = tf.minimum(ac_na, self.ac_space.high)
        ac_na = tf.maximum(ac_na, self.ac_space.low)
        return ac_na

    def tf_action(self, states_ns, is_initial=True):
        if is_initial:
            return self._explore_policy(states_ns)
        return self._exploit_policy(states_ns, reuse=True)

    def fit(self, data, **kwargs):
        obs = data.stationary_obs()
        if 'use_labelled' in kwargs and kwargs['use_labelled']:
            acs = data.labelled_acs()
        else:
            acs = data.stationary_acs()

        nexamples = len(obs)
        assert nexamples == len(acs), (nexamples, len(acs))
        per_epoch = max(nexamples // self.batch_size, 1)
        batches = np.random.randint(nexamples, size=(
            self.epochs * per_epoch, self.batch_size))
        for batch_idx in batches:
            input_states_sample = obs[batch_idx]
            label_actions_sample = acs[batch_idx]
            self.sess.run(self.update_op, feed_dict={
                self.input_state_ph_ns: input_states_sample,
                self.expert_action_ph_na: label_actions_sample})

    def act(self, states_ns):
        acs = self.sess.run(self.policy_action_na, feed_dict={
            self.input_state_ph_ns: states_ns})
        rws = np.zeros(len(states_ns))
        return acs, rws
