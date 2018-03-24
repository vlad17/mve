"""Learn the (stationary) dynamics from transitions"""

import distutils.util

import tensorflow as tf
import numpy as np

from context import flags
import env_info
from flags import Flags, ArgSpec
import reporter
from tfnode import TFNode
from utils import build_mlp, get_ob_dim, get_ac_dim, AssignableStatistic


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
            name='decoupled_dynamics',
            type=int,
            default=0,
            help='predict dynamics dimensions independently. '
            '0 = share weights. k = k-way associative weight sharing over '
            'output dimension')
        yield ArgSpec(
            name='restore_dynamics',
            default=None,
            type=str,
            help='restore dynamics from the given path')
        yield ArgSpec(
            name='dynamics_batches_per_timestep',
            type=float,
            default=4,
            help='number of mini-batches to train dynamics per '
            'new sample observed')
        yield ArgSpec(
            name='dyn_min_buf_size',
            default=1,
            type=int,
            help='Minimum number of frames in replay buffer before '
                 'training')
        yield ArgSpec(
            name='dyn_bn',
            default=False,
            type=distutils.util.strtobool,
            help='Add batch norm to the dynamics net')
        yield ArgSpec(
            name='dyn_l2_reg',
            type=float,
            default=0.,
            help='dynamics net l2 regularization')
        yield ArgSpec(
            name='dyn_dropout',
            type=float,
            default=0.,
            help='if nonzero, the dropout probability after every layer')

    def __init__(self):
        super().__init__('dynamics', 'learned dynamics',
                         list(DynamicsFlags._generate_arguments()))

    def nbatches(self, timesteps):
        """The number training batches, given this many timesteps."""
        nbatches = self.dynamics_batches_per_timestep * timesteps
        nbatches = max(int(nbatches), 1)
        return nbatches


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


class _DeltaNormalizer:
    def __init__(self, data, eps=1e-6):
        self._eps = eps
        stats = _Statistics(data)
        self._ob_tf_stats = AssignableStatistic(
            'ob', stats.mean_ob, stats.std_ob)
        self._delta_tf_stats = AssignableStatistic(
            'delta', stats.mean_delta, stats.std_delta)
        self._ac_tf_stats = AssignableStatistic(
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

    def log_stats(self):
        """report normalization statistics"""
        stats = [
            (self._ob_tf_stats, 'observations'),
            (self._ac_tf_stats, 'actions'),
            (self._delta_tf_stats, 'deltas')]

        prefix = 'dynamics statistics/'
        for stat, name in stats:
            reporter.add_summary_statistics(
                prefix + name + '/mean magnitude',
                np.absolute(stat.mean()),
                hide=True)
            reporter.add_summary_statistics(
                prefix + name + '/std',
                stat.std(),
                hide=True)


class NNDynamicsModel(TFNode):
    """Stationary neural-network-based dynamics model."""

    def __init__(self, env, norm_data, dyn_flags):
        ob_dim, ac_dim = get_ob_dim(env), get_ac_dim(env)

        self._input_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'dynamics_input_state')
        self._input_action_ph_na = tf.placeholder(
            tf.float32, [None, ac_dim], 'dynamics_input_action')
        self._next_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'true_next_state_diff')

        with tf.variable_scope('dynamics'):
            self._norm = _DeltaNormalizer(norm_data)
        # only used if dyn_bn is set to true
        self._dyn_training = tf.placeholder_with_default(
            False, [], 'dynamics_dyn_training_mode')
        self._mlp_kwargs = {
            'output_size': ob_dim,
            'scope': 'dynamics',
            'reuse': tf.AUTO_REUSE,
            'n_layers': dyn_flags.dyn_depth,
            'size': dyn_flags.dyn_width,
            'activation': tf.nn.relu,
            'l2reg': dyn_flags.dyn_l2_reg,
            'activation_norm': self._activation_norm}

        pred_states_ns, deltas_ns = self._predict_tf(
            self._input_state_ph_ns, self._input_action_ph_na)
        true_deltas_ns = self._norm.norm_delta(
            self._next_state_ph_ns - self._input_state_ph_ns)
        train_mse = tf.losses.mean_squared_error(
            true_deltas_ns, deltas_ns)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self._dyn_loss = reg_loss + train_mse
        self._one_step_pred_error = tf.losses.mean_squared_error(
            self._next_state_ph_ns, pred_states_ns)
        with tf.control_dependencies(update_ops):
            self._update_op = tf.train.AdamOptimizer(
                dyn_flags.dyn_learning_rate).minimize(self._dyn_loss)

        super().__init__('dynamics', dyn_flags.restore_dynamics)

    def fit(self, data, timesteps):
        """
        Fit the dynamics to the given dataset of transitions, given that
        we saw a certain number of new timesteps
        """
        if not data.size or not timesteps:
            return
        if data.size < flags().dynamics.dyn_min_buf_size:
            return
        self._norm.update_stats(data)
        self._norm.log_stats()

        if flags().experiment.should_evaluate():
            self._evaluate(data, 'dynamics/before training/')

        # I actually tried out the tf.contrib.train.Dataset API, and
        # it was *slower* than this feed_dict method. I figure that the
        # data throughput per batch is small enough (since the data is so
        # simple) that up-front shuffling as performed here is probably
        # saving us more time.
        nbatches = flags().dynamics.nbatches(timesteps)
        batch_size = flags().dynamics.dyn_batch_size
        for batch in data.sample_many(nbatches, batch_size):
            obs, next_obs, _, acs, _ = batch
            self._update_op.run(feed_dict={
                self._input_state_ph_ns: obs,
                self._input_action_ph_na: acs,
                self._next_state_ph_ns: next_obs,
                self._dyn_training: True
            })

        if flags().experiment.should_evaluate():
            self._evaluate(data, 'dynamics/before training/')

    def _predict_tf(self, states, actions):
        state_action_pair = tf.concat([
            self._norm.norm_obs(states),
            self._norm.norm_acs(actions)], axis=1)
        if flags().dynamics.decoupled_dynamics > 0:
            next_state = []
            skip = flags().dynamics.decoupled_dynamics
            for i in range(0, env_info.ob_dim(), skip):
                with tf.variable_scope('dyn{}-{}'.format(i, i + skip)):
                    mlp_kwargs = self._mlp_kwargs.copy()
                    mlp_kwargs['output_size'] = min(
                        env_info.ob_dim(), i + skip) - i
                    next_dim_nx = build_mlp(
                        state_action_pair, **mlp_kwargs)
                next_state.append(next_dim_nx)
            standard_predicted_state_diff_ns = tf.concat(next_state, axis=1)
        else:
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

    def _activation_norm(self, inputs):
        outputs = inputs
        if flags().dynamics.dyn_bn:
            outputs = tf.layers.batch_normalization(
                outputs, training=self._dyn_training)
        if flags().dynamics.dyn_dropout:
            outputs = tf.nn.dropout(
                outputs,
                tf.maximum(
                    1 - tf.to_float(self._dyn_training),
                    flags().dynamics.dyn_dropout))
        return outputs

    def _evaluate(self, data, prefix):
        if not data.size:
            return
        batch_size = flags().dynamics.dyn_batch_size * 10
        batch = next(data.sample_many(1, batch_size))
        obs, next_obs, _, acs, _ = batch
        dyn_loss, one_step_error = tf.get_default_session().run(
            [self._dyn_loss, self._one_step_pred_error],
            feed_dict={
                self._input_state_ph_ns: obs,
                self._input_action_ph_na: acs,
                self._next_state_ph_ns: next_obs,
            })
        reporter.add_summary(prefix + 'one-step mse', one_step_error)
        reporter.add_summary(prefix + 'loss', dyn_loss)
