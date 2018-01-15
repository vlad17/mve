"""
Utilities for generating rolluts from a controller.
"""

import numpy as np

from context import flags
from dataset import Path


def sample_venv(venv, controller):
    """
    Given a n-way vectorized environment `venv`, generate n paths/rollouts with
    maximum horizon `horizon` using controller `controller`.
    """
    max_horizon = flags().experiment.horizon
    obs_n = venv.reset()
    paths = [Path(venv, obs, max_horizon, controller.planning_horizon())
             for obs in obs_n]
    active_n = np.ones(len(obs_n), dtype=bool)
    controller.reset(len(active_n))

    # a = number of active environments (not done)
    for _ in range(max_horizon):
        # If there are no active environments, we're done!
        if np.sum(active_n) == 0:
            break

        # If an environment is inactive, we can still ask the actor
        # to give us an action for it (the venv and actor will give garbage
        # but valid-size outputs).
        acs_n, planned_acs_n, planned_obs_n = (
            controller.act(obs_n))
        obs_n, reward_n, done_n, _ = venv.step(acs_n)
        for i in np.flatnonzero(active_n):
            done_n[i] |= paths[i].next(
                obs_n[i], reward_n[i], done_n[i], acs_n[i],
                None if planned_acs_n is None else planned_acs_n[i],
                None if planned_obs_n is None else planned_obs_n[i])
            if done_n[i]:
                active_n[i] = False
    return paths


def sample(env, controller, render=False):
    """
    Given a single environment env, perform a rollout up to max_horizon
    steps, possibly rendering, with the given controller.
    """
    max_horizon = flags().experiment.horizon
    ob = env.reset()
    path = Path(env, ob, max_horizon, controller.planning_horizon())
    controller.reset(1)

    for _ in range(max_horizon):
        if render:
            env.render()
        ac, plan_ac, plan_ob = controller.act(ob[np.newaxis, ...])
        ob, reward, done, _ = env.step(ac[0])
        path.next(ob, reward, done, ac[0],
                  None if plan_ac is None else plan_ac[0],
                  None if plan_ob is None else plan_ob[0])
        if done:
            break
    return path
