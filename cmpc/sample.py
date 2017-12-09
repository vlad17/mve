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
    paths = [Path(venv, obs, max_horizon, policy.planning_horizon())
             for obs in obs_n]
    active_n = np.ones(len(obs_n), dtype=bool)
    policy.reset(len(active_n))

    # a = number of active environments (not done)
    for _ in range(max_horizon):
        # If there are no active environments, we're done!
        if np.sum(active_n) == 0:
            break

        # If an environment is inactive, we can still ask the actor
        # to give us an action for it (the venv and actor will give garbage
        # but valid-size outputs).
        acs_n, predicted_rewards_n, planned_acs_n = policy.act(obs_n)
        obs_n, reward_n, done_n, _ = venv.step(acs_n)
        for i in np.flatnonzero(active_n):
            done_n[i] |= paths[i].next(
                obs_n[i], reward_n[i], done_n[i], acs_n[i],
                predicted_rewards_n[i] if predicted_rewards_n is not None
                else None,
                planned_acs_n[i] if planned_acs_n is not None else None)
            if done_n[i]:
                active_n[i] = False
                venv.mask(i)
    return paths


def sample(env, policy, max_horizon=1000, render=False):
    """
    Given a single environment env, perform a rollout up to max_horizon
    steps, possibly rendering, with the given policy.
    """
    ob = env.reset()
    path = Path(env, ob, max_horizon, policy.planning_horizon())
    policy.reset(1)

    for _ in range(max_horizon):
        if render:
            env.render()
        ac, pred_rew, plan = policy.act(ob[np.newaxis, ...])
        ob, reward, done, _ = env.step(ac[0])
        path.next(ob, reward, done, ac[0],
                  None if pred_rew is None else pred_rew[0],
                  None if plan is None else plan[0])
        if done:
            break
    return path
