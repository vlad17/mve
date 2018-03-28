"""Learn the (stationary) dynamics from transitions"""

import distutils.util

import tensorflow as tf

from context import flags
from flags import Flags, ArgSpec
import env_info
import reporter
from tfnode import TFNode
from utils import build_mlp


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


class NNDynamicsModel(TFNode):
    """Stationary neural-network-based dynamics model."""

    def __init__(self, norm):
        dyn_flags = flags().dynamics
        ob_dim, ac_dim = env_info.ob_dim(), env_info.ac_dim()

        self._norm = norm
        self._input_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'dynamics_input_state')
        self._input_action_ph_na = tf.placeholder(
            tf.float32, [None, ac_dim], 'dynamics_input_action')
        self._next_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'true_next_state_diff')

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

    def evaluate(self, data, prefix='dynamics'):
        """report dynamics metrics"""
        if not data.size:
            return
        prefix = prefix + ('/' if prefix else '')
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
