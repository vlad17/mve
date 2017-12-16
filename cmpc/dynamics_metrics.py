"""Code for evaluating a dynamics model"""

import numpy as np

from flags import Flags, ArgSpec
from multiprocessing_env import make_venv
import log
import reporter
from utils import rate_limit


class DynamicsMetricsFlags(Flags):
    """Specification for the dynamics metrics"""

    def __init__(self):
        args = [
            ArgSpec(
                name='evaluation_envs',
                default=1000,
                type=int,
                help='number of environments to use for evaluating dynamics ')]
        super().__init__('dynamics_metrics', 'dynamics metrics', args)


class DynamicsMetrics:
    """
    The dynamics metrics evaluate how accurate a planning controller was
    in its predictions.

    We offer two evaluations, closed-loop and open-loop.

    Closed-loop evaluation is easy. Suppose we just had a rollout with states
    s_1, ..., s_T, and the controller, for every i in [T], predicted the
    next states s_i^j for j in [H] where H is the planning horizon.
    Then an observation for our h-step prediction MSE is (s_i^h-s_{i+h})^2.
    We average over all i and possibly several episodes.

    Since actions in the rollout are taken with respect to the closed-loop
    controller, which re-plans at every moment, the closed-loop evaluation
    tells us how far the agent's actual trajectories are from the
    planner's trajectory.

    An evaluation of the predictive power of the dynamics tests
    how close the dynamics are to what the open-loop planner
    thought it'd be doing. At each step i, the open-loop planner that the
    controller uses had an H-step plan of actions a_i^0, a_i^1, ..., a_i^H.
    It gets to execute a_i = a_i^0, but not the rest with MPC.
    However, the open-loop planner took a_i^0 because of how it perceived
    the future would look like based on the dynamics model after taking
    actions a_i^j.

    For i in [T], using the true environment M, we can evaluate the states
    that *would have occurred* had the planner executed its actions. Thus,
    starting at t_i^0 = s_i, we get the "true planned states"
    t_i^{j+1} = M(t_i^j, a_i^j). Then compute the corresponding h-step
    prediction MSE between t_i^j and s_i^j.
    """

    def __init__(self, planning_horizon, make_env, flags):
        self._venv = make_venv(make_env, flags.evaluation_envs)
        self._num_envs = flags.evaluation_envs
        self._horizon = planning_horizon

    @staticmethod
    def _mse_h(true_obs_nhs, pred_obs_nhs):
        # n = batch size, h = horizon, s = state dim
        return np.square(true_obs_nhs - pred_obs_nhs).sum(axis=2).mean(axis=0)

    def log(self, data):
        """
        Report H-step absolute and standardized dynamics accuracy.
        """
        self._log_open_loop_prediction(data)
        self._log_closed_loop_prediction(data)

    def _log_open_loop_prediction(self, data):
        if data.planned_acs.size == 0:
            return

        acs_nha = data.planned_acs
        assert acs_nha.shape[1] == self._horizon, (
            acs_nha.shape, self._horizon)
        obs_ns = data.obs
        obs_nhs, mask = self._eval_open_loop(obs_ns, acs_nha)
        if np.sum(mask) == 0:
            log.debug('all open loop evaluations terminated early -- '
                      'skipping open-loop dynamics eval')
            return
        acs_nha = acs_nha[mask]
        obs_nhs = obs_nhs[mask]
        obs_ns = obs_ns[mask]
        planned_obs_nhs = data.planned_obs[mask]

        try:
            mse_h = self._mse_h(obs_nhs, planned_obs_nhs)
        except FloatingPointError as e:
            log.debug('error during open-loop dynamics evaluation: {}', e)
            return
        prefix = 'dynamics/open loop/'
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
        obs, planned_obs = data.episode_obs_planned_obs()
        mask = [ob.shape[0] > self._horizon for ob in obs]
        if not all(mask):
            log.debug('WARNING: {} early terminations during closed-loop'
                      ' eval', len(mask) - sum(mask))
        if not any(mask):
            log.debug('skipping closed-loop {}-step dynamics logging '
                      '(all episodes too short)', self._horizon)
            return
        obs = [ob for include, ob in zip(mask, obs) if include]
        planned_obs = [pob for include, pob in zip(mask, planned_obs)
                       if include]
        # n = episode length - horizon - 1 since
        # predictions start for second state
        obs_hns = np.concatenate(
            [_wrap_diagonally(ob, self._horizon)[:, 1:] for ob in obs],
            axis=1)
        obs_nhs = np.swapaxes(obs_hns, 0, 1)
        planned_obs_nhs = np.concatenate(
            [pob[:-self._horizon - 1] for pob in planned_obs],
            axis=0)

        prefix = 'dynamics/closed loop/'
        mse_h = self._mse_h(obs_nhs, planned_obs_nhs)
        self._print_mse_prefix(prefix, mse_h)

    def _eval_open_loop(self, states_ns, acs_nha):
        states_nhs, mask_n = rate_limit(
            self._num_envs, self._eval_open_loop_limited,
            states_ns, acs_nha)
        ndone = np.sum(~mask_n)
        if ndone > 0:
            log.debug('WARNING: {} early terminations during open-loop'
                      ' eval', ndone)
        return states_nhs, mask_n

    def _eval_open_loop_limited(self, states_ns, acs_nha):
        self._venv.set_state_from_obs(states_ns)
        acs_hna = np.swapaxes(acs_nha, 0, 1)
        states_hns, done_n = self._venv.multi_step(acs_hna)
        states_nhs = np.swapaxes(states_hns, 0, 1)
        return states_nhs, ~done_n

    def close(self):
        """Close the internal simulation environment"""
        self._venv.close()


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
