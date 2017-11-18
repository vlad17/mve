"""
Like a DeterministicLearner, a StochasticLearner takes in a big set of rollouts
and learns to predict an action given a state. Unlike, the deterministic
learner, a stochastic learner acts stocahstically.
"""


import tensorflow as tf
import numpy as np

from learner import Learner
from utils import (build_mlp, create_random_tf_policy,
                   get_ac_dim, get_ob_dim, log_statistics)
import logz


class StochasticLearner(Learner):  # pylint: disable=too-many-instance-attributes
    """Noisy everywhere"""

    def __init__(self,
                 env,
                 learning_rate=None,
                 depth=None,
                 width=None,
                 batch_size=None,
                 epochs=None,
                 no_extra_explore=False,
                 sess=None):
        self.sess = sess
        self.batch_size = batch_size
        self.epochs = epochs
        self.ac_dim = get_ac_dim(env)
        self.width = width
        self.depth = depth
        self.ac_space = env.action_space
        self.extra_explore = not no_extra_explore

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
        nll = tf.negative(
            tf.reduce_mean(self._pdf(self.input_state_ph_ns,
                                     self.expert_action_ph_na)))
        # TODO: consider an l2_loss on policy actions?
        policy_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='learner_policy')
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(
            nll, var_list=policy_vars)
        self._nll = nll

    def _pdf(self, states_ns, expert_acs_na):
        ac_na = build_mlp(
            states_ns, scope='learner_policy',
            n_layers=self.depth, size=self.width, activation=tf.nn.relu,
            output_activation=tf.sigmoid, reuse=True)
        expert_acs_na -= self.ac_space.low
        expert_acs_na /= self.ac_space.high - self.ac_space.low
        with tf.variable_scope('learner_policy', reuse=True):
            logstd_a = tf.get_variable('logstd', [self.ac_dim],
                                       initializer=tf.constant_initializer(0))
        norm = tf.contrib.distributions.MultivariateNormalDiag(
            tf.zeros([self.ac_dim]), tf.exp(logstd_a))
        return norm.log_prob(expert_acs_na - ac_na)

    def _exploit_policy(self, states_ns, reuse=True):
        ac_na = build_mlp(
            states_ns, scope='learner_policy',
            n_layers=self.depth, size=self.width, activation=tf.nn.relu,
            output_activation=tf.sigmoid, reuse=reuse)
        with tf.variable_scope('learner_policy', reuse=reuse):
            logstd_a = tf.get_variable('logstd', [self.ac_dim],
                                       initializer=tf.constant_initializer(0))
        perturb_na = tf.random_normal([tf.shape(states_ns)[0], self.ac_dim])
        ac_na += perturb_na * tf.exp(logstd_a)
        ac_na = tf.clip_by_value(ac_na, 0, 1)
        ac_na *= self.ac_space.high - self.ac_space.low
        ac_na += self.ac_space.low
        return ac_na

    def _explore_policy(self, state_ns):
        random_policy = create_random_tf_policy(self.ac_space)
        return random_policy(state_ns)

    def tf_action(self, states_ns, is_initial=True):
        if is_initial and self.extra_explore:
            return self._explore_policy(states_ns)
        return self._exploit_policy(states_ns, reuse=True)

    def fit(self, data):
        obs = data.obs
        acs = data.acs
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

    def log(self, most_recent_rollouts):
        with tf.variable_scope('learner_policy', reuse=True):
            logstd_a = tf.get_variable('logstd')
        std_a = np.exp(self.sess.run(logstd_a))
        log_statistics('learner-policy-std', std_a)
        most_recent = most_recent_rollouts
        nll = self.sess.run(self._nll, feed_dict={
            self.input_state_ph_ns: most_recent.obs,
            self.expert_action_ph_na: most_recent.acs})
        logz.log_tabular('avg-nll', nll)
