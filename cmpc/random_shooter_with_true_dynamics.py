"""
Optimize the planning problem by randomly sampling trajectories from true
dynamics.
"""

import numpy as np

from context import flags
from controller import Controller
from itertools import chain
import env_info
from venv.parallel_venv import ParallelVenv
from utils import scale_to_box, rate_limit
from random_shooter import random_shooter_log_reward


flatten = lambda x: list(chain.from_iterable(x))


class RandomShooterWithTrueDynamics(Controller):
    """
    Random-shooter based optimization of the planning objective
    looking mpc_horizon steps ahead.

    See RandomShooterFlags for details.
    """

    def __init__(self):
        mpc_horizon = flags().mpc.mpc_horizon
        learner = flags().random_shooter.make_learner()
        assert mpc_horizon > 0, mpc_horizon
        self._sims_per_state = flags().random_shooter.simulated_paths
        self._n_envs = flags().random_shooter.rs_n_envs
        self._rollout_envs = ParallelVenv(self._n_envs)

        self._mpc_horizon = mpc_horizon

        self._learner = learner
        self._learner_test_env = ParallelVenv(10)

    def _act(self, states_ns):
        """Play forward using mpc_horizon steps in the actual OpenAI gym
        environment, trying simulated_paths number of random actions for each
        state.

        n - number of states
        a - action dimension
        h - horizon
        p - number of simulated paths per state
        s - obs dim
        e - n * p
        """
        # initialize nstates environments environments
        nstates = len(states_ns)
        p, h = self._sims_per_state, self._mpc_horizon

        # initialize states
        self._rollout_envs.set_state_from_obs(np.repeat(states_ns, p, axis=0))

        # initialize random actions
        ac_space = env_info.ac_space()
        ac_dim = ac_space.low.shape
        ac_ea = np.random.random((p * nstates,) + ac_dim)
        ac_ea = scale_to_box(ac_space, ac_ea)
        ac_hea = np.stack([ac_ea] * self._mpc_horizon)

        # play forward random actions
        obs_hes, rewards_he, _ = self._rollout_envs.multi_step(ac_hea)

        # collect our observations, rewards, and actions
        obs_hpns = obs_hes.reshape(h, p, nstates, -1)
        obs_nphs = np.transpose(obs_hpns, (2, 1, 0, 3))
        rewards_pn = np.sum(rewards_he, axis=0).reshape(p, nstates)
        ac_hpna = ac_hea.reshape(h, p, nstates, -1)
        ac_npha = np.transpose(ac_hpna, (2, 1, 0, 3))

        # find best paths for each state, by total reward across horizon
        best_idx = np.argmax(rewards_pn, axis=0)
        best_obs_nhs = np.stack([phs[i] for i, phs in zip(best_idx, obs_nphs)])
        best_ac_nha = np.stack([pha[i] for i, pha in zip(best_idx, ac_npha)])
        best_ac_hna = np.swapaxes(best_ac_nha, 0, 1)
        return best_ac_hna[0], best_ac_nha, best_obs_nhs

    def act(self, states_ns):
        """Rate-limit the `act` call."""
        limit = self._n_envs // self._sims_per_state
        return rate_limit(limit, self._act, states_ns)

    def planning_horizon(self):
        return self._mpc_horizon

    def fit(self, data, timesteps):
        self._learner.fit(data, timesteps)

    def log(self, most_recent):
        random_shooter_log_reward(self._learner, self._learner_test_env)
