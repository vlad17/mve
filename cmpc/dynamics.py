"""Learn the (stationary) dynamics from transitions"""

import tensorflow as tf
import numpy as np

from flags import Flags, ArgSpec
from multiprocessing_env import make_venv
import log
import reporter
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
            default=2,
            help='dynamics NN depth',)
        yield ArgSpec(
            name='dyn_width',
            type=int,
            default=500,
            help='dynamics NN width',)
        yield ArgSpec(
            name='dyn_learning_rate',
            type=float,
            default=1e-3,
            help='dynamics NN learning rate',)
        yield ArgSpec(
            name='dyn_epochs',
            type=int,
            default=60,
            help='dynamics NN epochs',)
        yield ArgSpec(
            name='dyn_batch_size',
            type=int,
            default=512,
            help='dynamics NN batch size',)
        yield ArgSpec(
            name='sample_percent',
            default=0.1,
            type=float,
            help='sub-sample previous states by this ratio when evaluating '
            'expensive dynamics metrics')
        yield ArgSpec(
            name='restore_dynamics',
            default=None,
            type=str,
            help='restore dynamics from the given path')

    def __init__(self):
        super().__init__('dynamics', 'learned dynamics',
                         list(DynamicsFlags._generate_arguments()))

    @property
    def subsample(self):  # pylint: disable=missing-docstring
        return self.sample_percent


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

    def __init__(self, env, norm_data, dyn_flags, mpc_horizon, make_env):
        ob_dim, ac_dim = get_ob_dim(env), get_ac_dim(env)
        self._epochs = dyn_flags.dyn_epochs
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
        self._metrics = _DynamicsMetrics(
            self, mpc_horizon, env, make_env, dyn_flags.subsample)

        super().__init__('dynamics', dyn_flags.restore_dynamics)

    def fit(self, data):
        """Fit the dynamics to the given dataset of transitions"""
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

    def __init__(self, dynamics, horizon, env, make_env, subsample):
        self._make_env = make_env
        self._subsample = subsample
        self._prediction_steps = [1, horizon if horizon > 1 else 2]

        # n = batch size, h = prediction horizon (one of prediction_steps)
        # a = action dim, s = state dim
        self._ob_ph_ns = tf.placeholder(tf.float32, [None, get_ob_dim(env)])
        self._ac_ph_hna = tf.placeholder(
            tf.float32, [None, None, get_ac_dim(env)])
        self._h_step_ob_ph_ns = tf.placeholder(
            tf.float32, self._ob_ph_ns.shape)

        h_step_state_ns = tf.foldl(dynamics.predict_tf, self._ac_ph_hna,
                                   initializer=self._ob_ph_ns, back_prop=False)
        self._mse = tf.losses.mean_squared_error(
            self._h_step_ob_ph_ns,
            h_step_state_ns,
            loss_collection=None)

    def log(self, data):
        """
        Report H-step absolute and standardized dynamics accuracy.
        """
        self._log_open_loop_prediction(data)
        self._log_closed_loop_prediction(data)

    def _log_open_loop_prediction(self, data):
        if data.planned_acs.size == 0:
            return

        n = len(data.obs)
        nsamples = max(int(n * self._subsample), 1)
        sample = np.random.randint(0, n, size=nsamples)

        prefix = 'dynamics/open loop/'
        for h_step in self._prediction_steps:
            acs_nha = data.planned_acs[sample, :h_step, :]
            acs_hna = np.swapaxes(acs_nha, 0, 1)
            obs_ns = data.obs[sample]
            ending_obs, mask = self._eval_open_loop(obs_ns, acs_hna)
            if np.sum(mask) == 0:
                log.debug('all open loop evaluations terminated early -- '
                          'skipping {} step eval', h_step)
                continue
            acs_hna = acs_hna[:, mask, :]
            obs_ns = obs_ns[mask, :]
            ending_obs = ending_obs[mask, :]

            mse = tf.get_default_session().run(
                self._mse, feed_dict={
                    self._ob_ph_ns: obs_ns,
                    self._ac_ph_hna: acs_hna,
                    self._h_step_ob_ph_ns: ending_obs})

            self._print_mse_prefix(prefix, mse, h_step)

    def _print_mse_prefix(self, prefix, mse, h_step):
        fmt = len(str(max(self._prediction_steps)))
        fmt = '{:' + str(fmt) + 'd}'
        fmt_str = prefix + fmt + '-step mse'
        print_str = fmt_str.format(h_step)
        reporter.add_summary(print_str, mse)

    def _log_closed_loop_prediction(self, data):
        prefix = 'dynamics/closed loop/'
        acs, obs = data.episode_acs_obs()
        for h_step in self._prediction_steps:
            starting_obs = [ob[:-h_step] for ob in obs]
            if any(ob.shape[0] == 0 for ob in starting_obs):
                log.debug('skipping closed-loop {}-step dynamics logging '
                          '(episode too short)', h_step)
                continue
            starting_obs = np.concatenate(starting_obs)
            hacs = [_wrap_diagonally(ac, h_step) for ac in acs]
            hacs = np.concatenate(hacs, axis=1)
            ending_obs = [ob[h_step:] for ob in obs]
            ending_obs = np.concatenate(ending_obs)

            mse = tf.get_default_session().run(
                self._mse, feed_dict={
                    self._ob_ph_ns: starting_obs,
                    self._ac_ph_hna: hacs,
                    self._h_step_ob_ph_ns: ending_obs})

            self._print_mse_prefix(prefix, mse, h_step)

    def _eval_open_loop(self, states_ns, acs_hna):
        venv = make_venv(self._make_env, acs_hna.shape[1])
        venv.set_state_from_obs(states_ns)
        for acs_na in acs_hna:
            states_ns, _, done_n, _ = venv.step(acs_na)
            done_n = np.asarray(done_n)
            for i in np.flatnonzero(done_n):
                venv.mask(i)
        ndone = np.sum(done_n)
        if ndone > 0:
            log.debug('WARNING: {} early termination(s) during open-loop'
                      ' eval', ndone)
        venv.close()
        return np.asarray(states_ns), ~done_n


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
