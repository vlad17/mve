"""Learn the (stationary) dynamics from transitions"""

import tensorflow as tf
import numpy as np

from flags import Flags, ArgSpec
from tfnode import TFNode
from utils import build_mlp, get_ob_dim, get_ac_dim


class DynamicsFlags(Flags):
    """
    We use a neural network to model an environment's dynamics. These flags
    define the architecture and learning policy of neural network.
    """

    @staticmethod
    def _generate_arguments():
        yield ArgSpec(
            name='dyn_depth',
            type=int,
            default=8,
            help='dynamics NN depth',)
        yield ArgSpec(
            name='dyn_width',
            type=int,
            default=128,
            help='dynamics NN width',)
        yield ArgSpec(
            name='dyn_learning_rate',
            type=float,
            default=1e-3,
            help='dynamics NN learning rate',)
        yield ArgSpec(
            name='dyn_batch_size',
            type=int,
            default=512,
            help='dynamics NN batch size',)
        yield ArgSpec(
            name='restore_dynamics',
            default=None,
            type=str,
            help='restore dynamics from the given path')
        yield ArgSpec(
            name='dynamics_nbatches',
            type=int,
            default=4000,
            help='number of mini-batches to train dynamics on every round',)

    def __init__(self):
        super().__init__('dynamics', 'learned dynamics',
                         list(DynamicsFlags._generate_arguments()))


class _Statistics:
    def __init__(self, data):
        if data.size == 0:
            self.mean_ob = np.zeros(data.obs.shape[1])
            self.std_ob = np.ones(data.obs.shape[1])
            self.mean_delta = np.zeros(data.obs.shape[1])
            self.std_delta = np.ones(data.obs.shape[1])
            self.mean_ac = np.zeros(data.acs.shape[1])
            self.std_ac = np.ones(data.acs.shape[1])
        else:
            self.mean_ob = np.mean(data.obs, axis=0)
            self.std_ob = np.std(data.obs, axis=0)
            diffs = data.next_obs - data.obs
            self.mean_delta = np.mean(diffs, axis=0)
            self.std_delta = np.std(diffs, axis=0)
            self.mean_ac = np.mean(data.acs, axis=0)
            self.std_ac = np.std(data.acs, axis=0)


class _AssignableStatistic:
    # Need stateful statistics so we don't have to feed
    # stats through a feed_dict every time we want to use the
    # dynamics.
    def __init__(self, suffix, initial_mean, initial_std):
        self._mean_var = tf.get_variable(
            name='mean_' + suffix, trainable=False,
            initializer=initial_mean.astype('float32'))
        self._std_var = tf.get_variable(
            name='std_' + suffix, trainable=False,
            initializer=initial_std.astype('float32'))
        self._mean_ph = tf.placeholder(tf.float32, initial_mean.shape)
        self._std_ph = tf.placeholder(tf.float32, initial_std.shape)
        self._assign_both = tf.group(
            tf.assign(self._mean_var, self._mean_ph),
            tf.assign(self._std_var, self._std_ph))

    def update_statistics(self, mean, std):
        """
        Update the stateful statistics using the default session.
        """
        tf.get_default_session().run(
            self._assign_both, feed_dict={
                self._mean_ph: mean,
                self._std_ph: std})

    def tf_normalize(self, x):
        """Normalize a value according to these statistics"""
        return (x - self._mean_var) / (self._std_var + 1e-6)

    def tf_denormalize(self, x):
        """Denormalize a value according to these statistics"""
        return x * (self._std_var + 1e-6) + self._mean_var


class _DeltaNormalizer:
    def __init__(self, data, eps=1e-6):
        self._eps = eps
        stats = _Statistics(data)
        self._ob_tf_stats = _AssignableStatistic(
            'ob', stats.mean_ob, stats.std_ob)
        self._delta_tf_stats = _AssignableStatistic(
            'delta', stats.mean_delta, stats.std_delta)
        self._ac_tf_stats = _AssignableStatistic(
            'ac', stats.mean_ac, stats.std_ac)

    def norm_obs(self, obs):
        """normalize observations"""
        return self._ob_tf_stats.tf_normalize(obs)

    def norm_acs(self, acs):
        """normalize actions"""
        return self._ac_tf_stats.tf_normalize(acs)

    def norm_delta(self, deltas):
        """normalize deltas"""
        return self._delta_tf_stats.tf_normalize(deltas)

    def denorm_delta(self, deltas):
        """denormalize deltas"""
        return self._delta_tf_stats.tf_denormalize(deltas)

    def update_stats(self, data):
        """update the stateful normalization statistics"""
        stats = _Statistics(data)
        self._ob_tf_stats.update_statistics(stats.mean_ob, stats.std_ob)
        self._delta_tf_stats.update_statistics(
            stats.mean_delta, stats.std_delta)
        self._ac_tf_stats.update_statistics(stats.mean_ac, stats.std_ac)


class NNDynamicsModel(TFNode):
    """Stationary neural-network-based dynamics model."""

    def __init__(self, env, norm_data, dyn_flags):
        ob_dim, ac_dim = get_ob_dim(env), get_ac_dim(env)
        self._nbatches = dyn_flags.dynamics_nbatches
        self._batch_size = dyn_flags.dyn_batch_size

        self._input_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'dynamics_input_state')
        self._input_action_ph_na = tf.placeholder(
            tf.float32, [None, ac_dim], 'dynamics_input_action')
        self._next_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'true_next_state_diff')

        with tf.variable_scope('dynamics', reuse=False):
            self._norm = _DeltaNormalizer(norm_data)
        self._mlp_kwargs = {
            'output_size': ob_dim,
            'scope': 'dynamics',
            'reuse': tf.AUTO_REUSE,
            'n_layers': dyn_flags.dyn_depth,
            'size': dyn_flags.dyn_width,
            'activation': tf.nn.relu,
            'output_activation': None}
        _, deltas_ns = self._predict_tf(
            self._input_state_ph_ns, self._input_action_ph_na)
        true_deltas_ns = self._norm.norm_delta(
            self._next_state_ph_ns - self._input_state_ph_ns)
        train_mse = tf.losses.mean_squared_error(
            true_deltas_ns, deltas_ns)
        self._update_op = tf.train.AdamOptimizer(
            dyn_flags.dyn_learning_rate).minimize(train_mse)

        super().__init__('dynamics', dyn_flags.restore_dynamics)

    def fit(self, data):
        """Fit the dynamics to the given dataset of transitions"""
        self._norm.update_stats(data)

        # I actually tried out the tf.contrib.train.Dataset API, and
        # it was *slower* than this feed_dict method. I figure that the
        # data throughput per batch is small enough (since the data is so
        # simple) that up-front shuffling as performed here is probably
        # saving us more time.
        for batch in data.sample_many(self._nbatches, self._batch_size):
            obs, next_obs, _, acs, _ = batch
            self._update_op.run(feed_dict={
                self._input_state_ph_ns: obs,
                self._input_action_ph_na: acs,
                self._next_state_ph_ns: next_obs,
            })

    def _predict_tf(self, states, actions):
        state_action_pair = tf.concat([
            self._norm.norm_obs(states),
            self._norm.norm_acs(actions)], axis=1)
        standard_predicted_state_diff_ns = build_mlp(
            state_action_pair, **self._mlp_kwargs)
        predicted_state_diff_ns = self._norm.denorm_delta(
            standard_predicted_state_diff_ns)
        return states + predicted_state_diff_ns, \
            standard_predicted_state_diff_ns

    def predict_tf(self, states, actions):
        """
        Predict the next state given the current state and action.

        Assumes input TF tensors are batched states and actions.
        Outputs corresponding predicted next state as a TF tensor.
        """
        return self._predict_tf(states, actions)[0]
