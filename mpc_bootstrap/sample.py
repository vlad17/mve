"""
Utilities for generating rolluts from a policy.
"""

import numpy as np

from dataset import Path


def sample_venv(venv, policy, max_horizon=1000):
    """
    Given a n-way vectorized environment `venv`, generate n paths/rollouts with
    maximum horizon `horizon` using policy `policy`.

    Parameters
    ----------
    venv: multiprocessing_env.MultiprocessingEnv
    policy: policy.Policy
    horizon: int
    """
    obs_n = venv.reset()
    paths = [Path(venv, obs, max_horizon) for obs in obs_n]
    active_n = np.ones(len(obs_n), dtype=bool)
    # a = number of active environments (not done)
    for _ in range(max_horizon):
        obs_a = np.asarray(obs_n)[active_n]
        acs_a, predicted_rewards_a = policy.act(obs_a)
        # list conversion required below b/c acs_n is 1D array of objects
        # but acs_a is a 2D matrix of floats
        acs_n = np.full(len(obs_n), None)
        acs_n[np.nonzero(active_n)] = list(acs_a)
        obs_n, reward_n, done_n, _ = venv.step(acs_n)
        for i, pred_rew in zip(np.flatnonzero(active_n), predicted_rewards_a):
            done_n[i] |= paths[i].next(
                obs_n[i], reward_n[i], done_n[i], acs_n[i], pred_rew)
            if done_n[i]:
                active_n[i] = False
                venv.mask(i)
    return paths
