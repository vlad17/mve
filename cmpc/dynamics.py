"""Learn the (stationary) dynamics from transitions"""

import tensorflow as tf
import numpy as np

from flags import Flags
from utils import build_mlp, get_ob_dim, get_ac_dim


class DynamicsFlags(Flags):
    """
    We use a neural network to model an environment's dynamics. These flags
    define the architecture and learning policy of neural network.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        dynamics_nn = parser.add_argument_group('dynamics')
        dynamics_nn.add_argument(
            '--dyn_depth',
            type=int,
            default=2,
            help='dynamics NN depth',
        )
        dynamics_nn.add_argument(
            '--dyn_width',
            type=int,
            default=500,
            help='dynamics NN width',
        )
        dynamics_nn.add_argument(
            '--dyn_learning_rate',
            type=float,
            default=1e-3,
            help='dynamics NN learning rate',
        )
        dynamics_nn.add_argument(
            '--dyn_epochs',
            type=int,
            default=60,
            help='dynamics NN epochs',
        )
        dynamics_nn.add_argument(
            '--dyn_batch_size',
            type=int,
            default=512,
            help='dynamics NN batch size',
        )

    @staticmethod
    def name():
        return 'dynamics'

    def __init__(self, args):
        self.dyn_depth = args.dyn_depth
        self.dyn_width = args.dyn_width
        self.dyn_learning_rate = args.dyn_learning_rate
        self.dyn_epochs = args.dyn_epochs
        self.dyn_batch_size = args.dyn_batch_size


class _DeltaNormalizer:
    def __init__(self, data, eps=1e-6):
        self.mean_ob = np.mean(data.obs, axis=0)
        self.std_ob = np.std(data.obs, axis=0)
        diffs = data.next_obs - data.obs
        self.mean_delta = np.mean(diffs, axis=0)
        self.std_delta = np.std(diffs, axis=0)
        self.mean_ac = np.mean(data.acs, axis=0)
        self.std_ac = np.std(data.acs, axis=0)
        self.eps = eps

    def norm_obs(self, obs):
        """normalize observations"""
        return (obs - self.mean_ob) / (self.std_ob + self.eps)

    def norm_acs(self, acs):
        """normalize actions"""
        return (acs - self.mean_ac) / (self.std_ac + self.eps)

    def norm_delta(self, deltas):
        """normalize deltas"""
        return (deltas - self.mean_delta) / (self.std_delta + self.eps)

    def denorm_delta(self, deltas):
        """denormalize deltas"""
        return deltas * self.std_delta + self.mean_delta


class NNDynamicsModel:  # pylint: disable=too-many-instance-attributes
    """Stationary neural-network-based dynamics model."""

    def __init__(self, env, norm_data, dyn_flags):
        ob_dim, ac_dim = get_ob_dim(env), get_ac_dim(env)
        self.epochs = dyn_flags.dyn_epochs
        self.batch_size = dyn_flags.dyn_batch_size

        self.input_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], "dynamics_input_state")
        self.input_action_ph_na = tf.placeholder(
            tf.float32, [None, ac_dim], "dynamics_input_action")
        self.next_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], "true_next_state_diff")

        self.norm = _DeltaNormalizer(norm_data)

        self.mlp_kwargs = {
            'output_size': ob_dim,
            'scope': 'dynamics',
            'reuse': None,
            'n_layers': dyn_flags.dyn_depth,
            'size': dyn_flags.dyn_width,
            'activation': tf.nn.relu,
            'output_activation': None}

        self.predicted_next_state_ns, deltas_ns = self._predict_tf(
            self.input_state_ph_ns, self.input_action_ph_na, reuse=None)
        self.mse = tf.losses.mean_squared_error(
            self.next_state_ph_ns,
            self.predicted_next_state_ns)
        true_deltas_ns = self.norm.norm_delta(
            self.next_state_ph_ns - self.input_state_ph_ns)
        train_mse = tf.losses.mean_squared_error(
            true_deltas_ns, deltas_ns)
        self.update_op = tf.train.AdamOptimizer(
            dyn_flags.dyn_learning_rate).minimize(train_mse)

    def fit(self, data):
        """Fit the dynamics to the given dataset of transitions"""
        # I actually tried out the tf.contrib.train.Dataset API, and
        # it was *slower* than this feed_dict method. I figure that the
        # data throughput per batch is small enough (since the data is so
        # simple) that up-front shuffling as performed here is probably
        # saving us more time.
        nbatches = data.batches_per_epoch(self.batch_size) * self.epochs
        for batch in data.sample_many(nbatches, self.batch_size):
            obs, next_obs, _, acs, _ = batch
            self.update_op.run(feed_dict={
                self.input_state_ph_ns: obs,
                self.input_action_ph_na: acs,
                self.next_state_ph_ns: next_obs,
            })

    def _predict_tf(self, states, actions, reuse=None):
        state_action_pair = tf.concat([
            self.norm.norm_obs(states),
            self.norm.norm_acs(actions)], axis=1)
        kwargs = self.mlp_kwargs.copy()
        kwargs['reuse'] = reuse
        standard_predicted_state_diff_ns = build_mlp(
            state_action_pair, **kwargs)
        predicted_state_diff_ns = self.norm.denorm_delta(
            standard_predicted_state_diff_ns)
        return states + predicted_state_diff_ns, \
            standard_predicted_state_diff_ns

    def predict_tf(self, states, actions):
        """Same as predict, but generates a tensor for the prediction"""
        return self._predict_tf(states, actions, reuse=True)[0]

    def predict(self, states, actions):
        """Return an array for the predicted next state"""
        next_states = self.predicted_next_state_ns.eval(feed_dict={
            self.input_state_ph_ns: states,
            self.input_action_ph_na: actions,
        })
        return next_states

    def dataset_mse(self, data):
        """Return the MSE of predictions for the given dataset"""
        mse = self.mse.eval(feed_dict={
            self.input_state_ph_ns: data.obs,
            self.input_action_ph_na: data.acs,
            self.next_state_ph_ns: data.next_obs,
        })
        return mse
