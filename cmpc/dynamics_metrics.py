"""Code for evaluating a dynamics model"""

import numpy as np

import env_info
from flags import Flags, ArgSpec
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
    Then an observation for our h-step prediction (dimension-normalized) MSE
    is (s_i^h-s_{i+h})^2/D where s is in R^d.
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

    Finally, we can also measure, to some extent, the nature of the error
    between t_i^j and s_i^j by evaluating the H-step reward according
    to the planner (i.e., from trajectory s_i^0 to s_i^H) versus
    what would have really happened (from t_i^0 to t_i^H). This
    gives a measure of the planner over-optimism or pessimism.
    """

    def __init__(self, planning_horizon, make_env, flags, discount):
        self._venv = env_info.make_venv(flags.evaluation_envs)
        self._venv.reset()
        self._env = make_env()
        self._num_envs = flags.evaluation_envs
        self._horizon = planning_horizon
        self._discount = discount

    @staticmethod
    def _mse_h(true_obs_nhs, pred_obs_nhs):
        # n = batch size, h = horizon, s = state dim
        return np.square(true_obs_nhs - pred_obs_nhs).mean(axis=2).mean(axis=0)

    def log(self, data, prefix=''):
        """
        Report H-step standardized dynamics accuracy.
        """
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

        mse_h = self._mse_h(obs_nhs, planned_obs_nhs)
        prefix += 'dynamics/open loop/'
        self._print_mse_prefix(prefix, mse_h)

        obs_n1s = obs_ns[:, np.newaxis, :]
        prev_obs_nhs = np.concatenate([obs_n1s, obs_nhs[:, :-1]], axis=1)
        prev_planned_obs_nhs = np.concatenate(
            [obs_n1s, planned_obs_nhs[:, :-1]], axis=1)
        rew_n = self._rewards_from_transitions(prev_obs_nhs, acs_nha, obs_nhs)
        planned_rew_n = self._rewards_from_transitions(
            prev_planned_obs_nhs, acs_nha, planned_obs_nhs)
        rew_bias = (rew_n - planned_rew_n).mean()
        rew_mse = np.square(rew_n - planned_rew_n).mean()
        reporter.add_summary('reward bias', rew_bias)
        reporter.add_summary('reward mse', rew_mse)

    def _rewards_from_transitions(self, prev_states_nhs, acs_nha, states_nhs):
        # define N = n * h
        prev_states_Ns = self._merge_axes(prev_states_nhs)
        acs_Ns = self._merge_axes(acs_nha)
        states_Ns = self._merge_axes(states_nhs)
        rew_N = self._env.np_reward(prev_states_Ns, acs_Ns, states_Ns)
        rew_nh = self._split_axes(rew_N)
        rew_n = np.zeros(rew_nh.shape[0])
        for i in reversed(range(rew_nh.shape[1])):
            rew_n = rew_nh[:, i] + rew_n * self._discount
        return rew_n

    def _merge_axes(self, arr):
        assert arr.ndim == 3, arr.ndim
        a, b, c = arr.shape
        assert b == self._horizon, (b, self._horizon)
        return arr.reshape(a * b, c)

    def _split_axes(self, arr):
        assert arr.ndim == 1, arr.ndim
        a = arr.shape[0]
        assert a % self._horizon == 0, (a, self._horizon)
        b, c = a // self._horizon, self._horizon
        return arr.reshape(b, c)

    def _print_mse_prefix(self, prefix, mse_h):
        fmt = len(str(self._horizon))
        fmt = '{:' + str(fmt) + 'd}'
        fmt_str = prefix + fmt + '-step mse'
        assert len(mse_h) == self._horizon, (len(mse_h), self._horizon)
        for h, mse in enumerate(mse_h, 1):
            print_str = fmt_str.format(h)
            hide = h not in [1, self._horizon]
            reporter.add_summary(print_str, mse, hide)

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
        self._venv.set_state_from_ob(states_ns)
        acs_hna = np.swapaxes(acs_nha, 0, 1)
        states_hns, _, done_hn = self._venv.multi_step(acs_hna)
        done_n = done_hn.sum(axis=0)
        states_nhs = np.swapaxes(states_hns, 0, 1)
        return states_nhs, ~done_n

    def close(self):
        """Close the internal simulation environment"""
        self._venv.close()
        self._env.close()
