"""Learn the (stationary) dynamics from transitions"""

import distutils.util
import itertools

import numpy as np
import tensorflow as tf

from context import flags
from flags import Flags, ArgSpec
import env_info
from log import debug
from memory import DummyNormalizer
import reporter
from tfnode import TFNode
from utils import build_mlp, trainable_vars


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
            name='dyn_target_rate',
            type=float,
            default=1e-3,
            help='dynamics NN target update rate',)
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
            name='disable_normalization',
            default=False,
            type=distutils.util.strtobool,
            help='Disable normalization')
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
        yield ArgSpec(
            name='dynamics_early_stop',
            type=float,
            default=0,
            help='if nonzero, use that proportion of the data to use for '
            'validation, which in turn informs early stopping')

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

        if flags().dynamics.disable_normalization:
            self._norm = DummyNormalizer()
        else:
            self._norm = norm
        self._input_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'dynamics_input_state')
        self._input_action_ph_na = tf.placeholder(
            tf.float32, [None, ac_dim], 'dynamics_input_action')
        self._next_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'true_next_state_diff')

        self._validation_mask_ph_n = tf.placeholder(
            tf.bool, [None], 'validation')

        # only used if dyn_bn is set to true
        self._dyn_training = tf.placeholder_with_default(
            False, [], 'dynamics_dyn_training_mode')
        self._mlp_kwargs = {
            'output_size': ob_dim,
            'reuse': tf.AUTO_REUSE,
            'n_layers': dyn_flags.dyn_depth,
            'size': dyn_flags.dyn_width,
            'activation': tf.nn.relu,
            'l2reg': dyn_flags.dyn_l2_reg,
            'activation_norm': self._activation_norm}

        pred_states_ns, deltas_ns = self._predict_tf(
            self._input_state_ph_ns, self._input_action_ph_na)

        val_pred_states_ns, val_deltas_ns = self._predict_tf(
            self._input_state_ph_ns, self._input_action_ph_na, 'target_dynamics')

        true_deltas_ns = self._norm.norm_delta(
            self._next_state_ph_ns - self._input_state_ph_ns)

        self._train_loss = tf.losses.mean_squared_error(
            true_deltas_ns, deltas_ns, weights=tf.expand_dims(
                1 - tf.to_float(self._validation_mask_ph_n), 1))
        self._validation_loss = tf.losses.mean_squared_error(
            true_deltas_ns, val_deltas_ns, weights=tf.expand_dims(
                tf.to_float(self._validation_mask_ph_n), 1))
        self._train_one_step_pred_error = tf.losses.mean_squared_error(
            self._next_state_ph_ns, pred_states_ns, weights=tf.expand_dims(
                1 - tf.to_float(self._validation_mask_ph_n), 1))
        self._validation_one_step_pred_error = tf.losses.mean_squared_error(
            self._next_state_ph_ns, val_pred_states_ns, weights=tf.expand_dims(
                tf.to_float(self._validation_mask_ph_n), 1))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        with tf.control_dependencies(update_ops):
            self._update_op = tf.train.AdamOptimizer(
                dyn_flags.dyn_learning_rate).minimize(
                    self._train_loss + reg_loss)

        updates = []
        target_vars = trainable_vars('target_dynamics')
        current_vars = trainable_vars('dynamics')

        with tf.control_dependencies([self._update_op]):
            for current_var, target_var in zip(current_vars, target_vars):
                updates.append(
                    tf.assign_add(
                        target_var, dyn_flags.dyn_target_rate * (
                            current_var.read_value() - target_var.read_value())))

        self._update_op = tf.group(*updates)

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

        if flags().dynamics.dynamics_early_stop > 0:
            val_loss = np.inf
            for i in itertools.count(1):
                new_val_loss = self._get_losses(data)[1]
                if new_val_loss >= val_loss:
                    debug('dyn new val loss {:.4g} >= old one {:.4g}, '
                          'stopping dynamics training',
                          new_val_loss, val_loss)
                    return
                else:
                    debug('dyn new val loss {:.4g} >= old one {:.4g}, '
                          'continuing dynamics training (epoch {})',
                          new_val_loss, val_loss, i)
                val_loss = new_val_loss
                self._do_training_epoch(data, timesteps)
        else:
            self._do_training_epoch(data, timesteps)

    def _do_training_epoch(self, data, timesteps):
        # I actually tried out the tf.contrib.train.Dataset API, and
        # it was *slower* than this feed_dict method. I figure that the
        # data throughput per batch is small enough (since the data is so
        # simple) that up-front shuffling as performed here is probably
        # saving us more time.
        nbatches = flags().dynamics.nbatches(timesteps)
        batch_size = flags().dynamics.dyn_batch_size
        for i, batch in enumerate(data.sample_many(nbatches, batch_size), 1):
            obs, next_obs, _, acs, _, masks = batch
            self._update_op.run(feed_dict={
                self._input_state_ph_ns: obs,
                self._input_action_ph_na: acs,
                self._next_state_ph_ns: next_obs,
                self._dyn_training: True,
                self._validation_mask_ph_n: masks[:, 0]
            })
            if (i % max(nbatches // 3, 1)) == 0:
                train_loss, val_loss, train_mse, val_mse = self._get_losses(
                    data)
                fmt = '{: ' + str(len(str(nbatches))) + 'd}'
                debug('dynamics ' + fmt + ' of ' + fmt + ' batches - '
                      'dyn train loss {:.4g} val loss {:.4g} '
                      'train mse {:.4g} val mse {:.4g}',
                      i, nbatches, train_loss, val_loss, train_mse, val_mse)

    def _predict_tf(self, states, actions, scope='dynamics'):
        state_action_pair = tf.concat([
            self._norm.norm_obs(states),
            self._norm.norm_acs(actions)], axis=1)
        standard_predicted_state_diff_ns = build_mlp(
            state_action_pair, scope=scope, **self._mlp_kwargs)
        predicted_state_diff_ns = self._norm.denorm_delta(
            standard_predicted_state_diff_ns)
        return states + predicted_state_diff_ns, \
            standard_predicted_state_diff_ns

    def predict_tf(self, states, actions):
        """
        Predict the current dynamics' next state given the 
        current state and action.

        Assumes input TF tensors are batched states and actions.
        Outputs corresponding predicted next state as a TF tensor.
        """
        return self._predict_tf(states, actions, 'dynamics')[0]


    def target_predict_tf(self, states, actions):
        """
        Predict the current dynamics' next state given the 
        current state and action.
        """
        return self._predict_tf(states, actions, 'target_dynamics')[0]


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

    def _get_losses(self, data):
        losses = [
            self._train_loss,
            self._validation_loss,
            self._train_one_step_pred_error,
            self._validation_one_step_pred_error]
        batch_size = flags().dynamics.dyn_batch_size * 32
        batch = next(data.sample_many(1, batch_size))
        obs, next_obs, _, acs, _, mask = batch
        return tf.get_default_session().run(
            losses,
            feed_dict={
                self._input_state_ph_ns: obs,
                self._input_action_ph_na: acs,
                self._next_state_ph_ns: next_obs,
                self._validation_mask_ph_n: mask[:, 0]
            })

    def evaluate(self, data, prefix='dynamics'):
        """report dynamics metrics"""
        if not data.size:
            return
        prefix = prefix + ('/' if prefix else '')
        train_loss, val_loss, train_mse, val_mse = self._get_losses(data)
        reporter.add_summary(prefix + 'train mse', train_mse)
        reporter.add_summary(prefix + 'train loss', train_loss)
        reporter.add_summary(prefix + 'val mse', val_mse)
        reporter.add_summary(prefix + 'val loss', val_loss)
