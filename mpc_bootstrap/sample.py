"""
Utilities for generating rolluts from a policy.
"""

from utils import Path

def sample_venv(venv, policy, horizon=1000):
    """
    Given a n-way vectorized environment `venv`, generate n paths/rollouts with
    horizon `horizon` using policy `policy`. We currently assume that none of
    the n environments in `venv` are "done" before `horizon` is reached.

    Parameters
    ----------
    venv: multiprocessing_env.MultiprocessingEnv
    policy: policy.Policy
    horizon: int
    """
    obs_n = venv.reset()
    policy.reset(len(obs_n))
    paths = [Path(venv, obs, horizon) for obs in obs_n]
    for t in range(horizon):
        acs_n, predicted_rewards_n = policy.act(obs_n)
        obs_n, reward_n, done_n, _ = venv.step(acs_n)
        for p, path in enumerate(paths):
            done_n[p] |= path.next(obs_n[p], reward_n[p], acs_n[p],
                                   predicted_rewards_n[p])
            if done_n[p]:
                assert t + 1 == horizon, (t + 1, horizon)
    return paths
