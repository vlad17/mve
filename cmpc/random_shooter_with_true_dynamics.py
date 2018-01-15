"""
Optimize the planning problem by randomly sampling trajectories from true
dynamics.
"""

import numpy as np

from context import flags
from controller import Controller
import env_info
from utils import scale_to_box, rate_limit, discounted_rewards
from random_shooter import random_shooter_log_reward


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
        self._rollout_envs = env_info.make_venv(self._n_envs)

        self._mpc_horizon = mpc_horizon

        self._learner = learner
        self._learner_test_env = env_info.make_venv(10)

    def act(self, states_ns):
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
        init_states_es = np.repeat(states_ns, p, axis=0)

        # initialize random actions
        ac_space = env_info.ac_space()
        ac_dim = ac_space.low.shape
        ac_ea = np.random.random((p * nstates,) + ac_dim)
        ac_ea = scale_to_box(ac_space, ac_ea)
        ac_hea = np.stack([ac_ea] * self._mpc_horizon)
        ac_eha = np.swapaxes(ac_hea, 0, 1)

        # play forward random actions
        obs_ehs, rewards_eh, _ = rate_limit(
            self._n_envs, self._set_state_multi_step, init_states_es, ac_eha)
        obs_hes = np.swapaxes(obs_ehs, 0, 1)
        rewards_he = np.swapaxes(rewards_eh, 0, 1)

        # collect our observations, rewards, and actions
        obs_hpns = obs_hes.reshape(h, p, nstates, -1)
        obs_nphs = np.transpose(obs_hpns, (2, 1, 0, 3))
        rewards_pn = discounted_rewards(rewards_he).reshape(p, nstates)
        ac_hpna = ac_hea.reshape(h, p, nstates, -1)
        ac_npha = np.transpose(ac_hpna, (2, 1, 0, 3))

        # find best paths for each state, by total reward across horizon
        best_idx = np.argmax(rewards_pn, axis=0)
        best_obs_nhs = np.stack([phs[i] for i, phs in zip(best_idx, obs_nphs)])
        best_ac_nha = np.stack([pha[i] for i, pha in zip(best_idx, ac_npha)])
        best_ac_hna = np.swapaxes(best_ac_nha, 0, 1)
        return best_ac_hna[0], best_ac_nha, best_obs_nhs

    def _set_state_multi_step(self, init_states_es, ac_eha):
        """Initialize states and run multistep."""
        self._rollout_envs.set_state_from_ob(init_states_es)
        ac_hea = np.swapaxes(ac_eha, 0, 1)
        obs_hes, rewards_he, done_he = self._rollout_envs.multi_step(ac_hea)
        obs_ehs = np.swapaxes(obs_hes, 0, 1)
        rewards_eh = np.swapaxes(rewards_he, 0, 1)
        done_eh = np.swapaxes(done_he, 0, 1)
        return obs_ehs, rewards_eh, done_eh

    def planning_horizon(self):
        return self._mpc_horizon

    def fit(self, data, timesteps):
        self._learner.fit(data, timesteps)

    def log(self, most_recent):
        random_shooter_log_reward(self._learner, self._learner_test_env)
