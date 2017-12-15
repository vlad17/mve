"""Code for evaluating a dynamics model"""

import numpy as np
import tensorflow as tf

from flags import Flags, ArgSpec
from multiprocessing_env import make_venv
import log
import reporter
from utils import get_ob_dim, get_ac_dim


class DynamicsMetricsFlags(Flags):
    """Specification for the dynamics metrics"""

    def __init__(self):
        args = [
            ArgSpec(
                name='sample_percent',
                default=0.1,
                type=float,
                help='sub-sample previous states by this ratio when evaluating'
                ' expensive dynamics metrics')]
        super().__init__('dynamics_metrics', 'dynamics metrics', args)


class DynamicsMetrics:
    """
    The dynamics metrics evaluate how accurate a planning controller was
    in its predictions.

    We offer two evaluations, closed-loop and open-loop.

    Closed-loop evaluation is easy. If we just had a rollout with states
    s_1, ..., s_T and actions taken a_1, ..., a_T then we can evaluate
    closed-loop dynamics MSE by taking any s_i with i <= T - H if H
    is the planning horizon for the controller that was used to make the
    rollout. Then, we take our dynamics model M and generate predicted
    states: p_i, ..., p_{i+H} where p_i = s_i and p_{j+1} = M(p_j, a_j).

    Then an observation for our h-step prediction MSE is (p_{i+h}-s_{i+h})^2.
    We average over all i and possibly several episodes.

    Since actions a_i are taken with respect to the closed-loop controller,
    the closed-loop evaluation is not exactly what we want (though still
    useful for debugging). Our desired evaluation is actually what the planner
    thought it'd be doing. At each step i, the open-loop planner that the
    controller uses had an H-step plan of actions a_i^0, a_i^1, ..., a_i^H.
    It gets to execute a_i = a_i^0, but not the rest with MPC.
    However, the open-loop planner took a_i^1 because of how it perceived
    the future would look like based on the dynamics model after taking
    actions a_i^j.

    Now for a sampled i in [T] we evaluate p_i^1 = s_i and
    p_i^{j+1} = M(p_i^j, a_i^j), then compute the corresponding prediction MSE.
    """

    def __init__(self, dynamics, horizon, env, make_env, flags):
        self._make_env = make_env
        self._subsample = flags.sample_percent
        self._horizon = horizon

        # n = batch size, h = horizon
        # a = action dim, s = state dim
        self._ob_ph_ns = tf.placeholder(tf.float32, [None, get_ob_dim(env)])
        self._ac_ph_hna = tf.placeholder(
            tf.float32, [horizon, None, get_ac_dim(env)])
        self._future_ob_ph_hns = tf.placeholder(
            tf.float32, [horizon, None, get_ob_dim(env)])

        state_hns = tf.scan(dynamics.predict_tf, self._ac_ph_hna,
                            initializer=self._ob_ph_ns, back_prop=False)
        diffsq_hns = tf.square(state_hns - self._future_ob_ph_hns)
        se_hn = tf.reduce_sum(diffsq_hns, axis=2)
        self._mse_h = tf.reduce_mean(se_hn, axis=1)

    def log(self, data):
        """
        Report H-step absolute and standardized dynamics accuracy.
        """
        self._log_open_loop_prediction(data)
        self._log_closed_loop_prediction(data)

    def _log_open_loop_prediction(self, data):
        if data.planned_acs.size == 0:
            return

        prefix = 'dynamics/open loop/'
        n = len(data.obs)
        nsamples = max(int(n * self._subsample), 1)
        sample = np.random.randint(0, n, size=nsamples)
        acs_nha = data.planned_acs[sample, :self._horizon, :]
        acs_hna = np.swapaxes(acs_nha, 0, 1)
        obs_ns = data.obs[sample]
        obs_hns, mask = self._eval_open_loop(obs_ns, acs_hna)
        if np.sum(mask) == 0:
            log.debug('all open loop evaluations terminated early -- '
                      'skipping open-loop dynamics eval')
            return
        acs_hna = acs_hna[:, mask, :]
        obs_ns = obs_ns[mask, :]
        obs_hns = obs_hns[:, mask, :]

        mse_h = tf.get_default_session().run(
            self._mse_h, feed_dict={
                self._ob_ph_ns: obs_ns,
                self._ac_ph_hna: acs_hna,
                self._future_ob_ph_hns: obs_hns})
        self._print_mse_prefix(prefix, mse_h)

    def _print_mse_prefix(self, prefix, mse_h):
        fmt = len(str(self._horizon))
        fmt = '{:' + str(fmt) + 'd}'
        fmt_str = prefix + fmt + '-step mse'
        assert len(mse_h) == self._horizon, (len(mse_h), self._horizon)
        for h, mse in enumerate(mse_h, 1):
            print_str = fmt_str.format(h)
            hide = h not in [1, self._horizon]
            reporter.add_summary(print_str, mse, hide)

    def _log_closed_loop_prediction(self, data):
        prefix = 'dynamics/closed loop/'
        acs, obs = data.episode_acs_obs()
        mask = [ob.shape[0] > self._horizon for ob in obs]
        if not all(mask):
            log.debug('WARNING: {} early terminations during closed-loop'
                      ' eval', len(mask) - sum(mask))
        if not any(mask):
            log.debug('skipping closed-loop {}-step dynamics logging '
                      '(all episodes too short)', self._horizon)
            return
        acs = [ac for include, ac in zip(mask, acs) if include]
        obs = [ob for include, ob in zip(mask, obs) if include]
        h1obs = [_wrap_diagonally(ob, self._horizon + 1) for ob in obs]
        h1obs = np.concatenate(h1obs, axis=1)
        hacs = [_wrap_diagonally(ac, self._horizon) for ac in acs]
        hacs = [hac[:, :-1, :] for hac in hacs]
        hacs = np.concatenate(hacs, axis=1)

        # could subsample here, but not necessary (fast enough)

        mse_h = tf.get_default_session().run(
            self._mse_h, feed_dict={
                self._ob_ph_ns: h1obs[0],
                self._ac_ph_hna: hacs,
                self._future_ob_ph_hns: h1obs[1:]})
        self._print_mse_prefix(prefix, mse_h)

    def _eval_open_loop(self, states_ns, acs_hna):
        venv = make_venv(self._make_env, acs_hna.shape[1])
        venv.set_state_from_obs(states_ns)
        states_hns = np.empty(acs_hna.shape[:1] + states_ns.shape)
        for i, acs_na in enumerate(acs_hna):
            states_ns, _, done_n, _ = venv.step(acs_na)
            states_hns[i] = np.asarray(states_ns)
            done_n = np.asarray(done_n)
            for j in np.flatnonzero(done_n):
                venv.mask(j)
        ndone = np.sum(done_n)
        if ndone > 0:
            log.debug('WARNING: {} early terminations during open-loop'
                      ' eval', ndone)
        venv.close()
        return states_hns, ~done_n


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
