"""
Optimize the planning problem by randomly sampling trajectories from true
dynamics.
"""

import numpy as np

from context import flags
from controller import Controller
import env_info
from multiprocessing_env import make_venv
from utils import rate_limit, scale_to_box, random_shooter_log_reward


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
        self._n_states = 0
        self._rollout_envs = None

        self._mpc_horizon = mpc_horizon

        self._learner = learner
        self._learner_test_env = make_venv(
            flags().experiment.make_env, 10)

    def _act(self, states):
        """Play forward using mpc_horizon steps in the actual OpenAI gym
        environment, trying simulated_paths number of random actions for each
        state.

        n - number of states
        a - action dimension
        h - horizon
        p - number of simulated paths per state
        s - obs dim
        """
        # initialize nstates environments environments
        nstates = len(states)
        if nstates > self._n_states:
            # TODO rollout?
            self._rollout_envs = make_venv(flags().experiment.make_env, nstates)

        # play random actions p times for each state
        obs_phns, rewards_pn, ac_phna = [], [], []
        self._rollout_envs.set_state_from_obs(states)
        for _ in range(self._sims_per_state):
            self._rollout_envs.reset()

            # initialize random actions
            ac_space = env_info.ac_space()
            ac_dim = ac_space.low.shape
            ac_na = np.random.random((nstates,) + ac_dim)
            ac_na = scale_to_box(ac_space, ac_na)
            ac_hna = np.stack([ac_na] * self._mpc_horizon)

            # play forward random actions
            obs_hns, rewards_hn, _ = self._rollout_envs.multi_step(ac_hna)

            # collect our observations, rewards, and actions
            obs_phns.append(obs_hns)
            rewards_pn.append(np.sum(rewards_hn, axis=0))
            ac_phna.append(ac_hna)

        obs_nphs = np.transpose(np.stack(obs_phns), (2, 0, 1, 3))
        rewards_pn = np.stack(rewards_pn)
        ac_npha = np.transpose(np.stack(ac_phna), (2, 0, 1, 3))

        # find best paths for each state, by total reward across horizon
        best_idx = np.argmax(rewards_pn, axis=0)
        best_obs_nhs = np.stack([phs[i] for i, phs in zip(best_idx, obs_nphs)])
        best_ac_nha = np.stack([pha[i] for i, pha in zip(best_idx, ac_npha)])
        best_ac_hna = np.swapaxes(best_ac_nha, 0, 1)
        return best_ac_hna[0], best_ac_nha, best_obs_nhs

    def act(self, states_ns):
        return rate_limit(500, self._act, states_ns)

    def planning_horizon(self):
        return self._mpc_horizon

    def fit(self, data, timesteps):
        self._learner.fit(data, timesteps)

    def log(self, most_recent):
        random_shooter_log_reward(self, most_recent)