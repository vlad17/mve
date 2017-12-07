"""Learn the (stationary) dynamics from transitions"""

import tensorflow as tf
import numpy as np

from flags import Flags
import reporter
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
        dynamics_nn.add_argument(
            '--renormalize',
            default=False,
            action='store_true',
            help='re-calculate dynamics normalization statistics after every '
            'iteration'
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
        self.renormalize = args.renormalize


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
        self._mean_var = tf.Variable(
            initial_value=initial_mean, dtype=tf.float32,
            name='mean_' + suffix, trainable=False)
        self._std_var = tf.Variable(
            initial_value=initial_std, dtype=tf.float32,
            name='std_' + suffix, trainable=False)
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


class NNDynamicsModel: # pylint: disable=too-many-instance-attributes
    """Stationary neural-network-based dynamics model."""

    def __init__(self, env, norm_data, dyn_flags, mpc_horizon):
        ob_dim, ac_dim = get_ob_dim(env), get_ac_dim(env)
        self._epochs = dyn_flags.dyn_epochs
        self._batch_size = dyn_flags.dyn_batch_size

        self._input_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'dynamics_input_state')
        self._input_action_ph_na = tf.placeholder(
            tf.float32, [None, ac_dim], 'dynamics_input_action')
        self._next_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'true_next_state_diff')

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
        self._renormalize = dyn_flags.renormalize
        self._metrics = _DynamicsMetrics(self, mpc_horizon, env)

    def fit(self, data):
        """Fit the dynamics to the given dataset of transitions"""
        if self._renormalize:
            self._norm.update_stats(data)

        # I actually tried out the tf.contrib.train.Dataset API, and
        # it was *slower* than this feed_dict method. I figure that the
        # data throughput per batch is small enough (since the data is so
        # simple) that up-front shuffling as performed here is probably
        # saving us more time.
        nbatches = data.batches_per_epoch(self._batch_size) * self._epochs
        for batch in data.sample_many(nbatches, self._batch_size):
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

    def log(self, data):
        """
        Given data from recent episodes, this reports various statistics about
        dynamics prediction accuracy.
        """
        self._metrics.log(data)

class _DynamicsMetrics:

    def __init__(self, dynamics, horizon, env):
        num_intervals = 5
        if horizon <= num_intervals:
            self._prediction_steps = range(1, num_intervals + 1)
        else:
            # 5-way split, always including 1 and horizon.
            self._prediction_steps = [1] + [
                i * (horizon - 1) // (num_intervals - 1) + 1
                for i in range(1, num_intervals)]

        # n = batch size, h = prediction horizon (one of prediction_steps)
        # a = action dim, s = state dim
        self._ob_ph_ns = tf.placeholder(tf.float32, [None, get_ob_dim(env)])
        self._ac_ph_hna = tf.placeholder(
            tf.float32, [None, None, get_ac_dim(env)])
        self._h_step_ob_ph_ns = tf.placeholder(
            tf.float32, self._ob_ph_ns.shape)

        h_step_state_ns = tf.foldl(dynamics.predict_tf, self._ac_ph_hna,
                                   initializer=self._ob_ph_ns, back_prop=False)
        self._absolute_mse = tf.losses.mean_squared_error(
            self._h_step_ob_ph_ns,
            h_step_state_ns,
            loss_collection=None)
        mean_guess = tf.reduce_mean(
            self._h_step_ob_ph_ns, axis=0, keep_dims=True)
        n = tf.shape(self._h_step_ob_ph_ns)[0]
        normalizer = tf.losses.mean_squared_error(
            self._h_step_ob_ph_ns,
            tf.tile(mean_guess, [n, 1]),
            loss_collection=None)
        self._absolute_mse = self._absolute_mse
        self._smse = self._absolute_mse / normalizer

    def log(self, data):
        """
        Report H-step absolute and standardized dynamics accuracy.
        """
        acs, obs = data.episode_acs_obs()
        for h_step in self._prediction_steps:
            starting_obs = [ob[:-h_step] for ob in obs]
            starting_obs = np.concatenate(starting_obs)
            hacs = [_wrap_diagonally(ac, h_step) for ac in acs]
            hacs = np.concatenate(hacs, axis=1)
            ending_obs = [ob[h_step:] for ob in obs]
            ending_obs = np.concatenate(ending_obs)

            absolute_mse, smse = tf.get_default_session().run(
                [self._absolute_mse, self._smse], feed_dict={
                    self._ob_ph_ns: starting_obs,
                    self._ac_ph_hna: hacs,
                    self._h_step_ob_ph_ns: ending_obs})

            fmt = len(str(max(self._prediction_steps)))
            fmt = '{:' + str(fmt) + 'd}'
            absolute_str = 'absolute ' + fmt + '-step dynamics mse'
            reporter.add_summary(absolute_str.format(h_step), absolute_mse)
            smse_str = 'standardized ' + fmt + '-step dynamics mse'
            reporter.add_summary(smse_str.format(h_step), smse)

def _wrap_diagonally(actions_na, horizon):
    # "wrap" an n-by-a array into a horizon-(n-horizon)-a array res,
    # satisfying the following property for all i, 0 <= i < n - horizon,
    # all j, 0 <= j < a, and all k, 0 <= k <= horizon:
    #
    # res[k][i][j] = actions_na[i + k][j]
    n, a = actions_na.shape
    res = np.zeros((horizon, n - horizon, a))
    for k in range(horizon):
        end = n - horizon + k
        res[k] = actions_na[k:end]
    return res
