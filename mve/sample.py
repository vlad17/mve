"""
Utilities for generating rolluts from a controller.
"""

import numpy as np

from context import flags
from flags import Flags, ArgSpec
from dataset import Path
import sys


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

    Is guaranteed to complete an episode before returning a path.
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


class Sampler:
    """Stateful sampler for shorter periods between parameter updates.

    Is NOT guaranteed to complete any episode or finish at most one episode. It
    may finish any from 0 to 10 episodes.

    As a result, when using the sampler, recognize calling `sample` once does
    not correspond to executing a single episode and refactor your game loop
    accordingly.
    """

    def __init__(self, env):
        self.env = env
        # TODO: instead of making a ddpg-specific flag for update_every here,
        # make update_every an argument passed to __init__ for this object
        # and then let the caller figure out how to get update_every into
        # this class.
        self.update_every = flags().sampler.sample_interval
        self.ob = env.reset()
        self.max_horizon = flags().experiment.horizon
        self.current_timestep = 0

    def sample(self, controller, data):
        """
        Given a single environment env, perform a rollout up to update_every
        steps, possibly rendering, with the given controller.
        """
        controller.reset(1)
        sample_reward = n_episodes = 0

        for _ in range(self.update_every):
            ac, plan_ac, plan_ob = controller.act(self.ob[np.newaxis, ...])
            try:
                next_ob, reward, done, _ = self.env.step(ac[0])
            except:
                print('ACTION', ac[0], file=sys.stderr)
                raise
            self.current_timestep += 1
            data.next(self.ob, next_ob, reward, done, ac[0],
                      None if plan_ac is None else plan_ac[0],
                      None if plan_ob is None else plan_ob[0])
            sample_reward += reward
            self.ob = next_ob
            if self.current_timestep == self.max_horizon:
                # non-terminal done: cut off the episode but don't report as
                # terminal (doing so would make the environment non-Markov)
                # This is actually very important for SAC.
                done = True
            if done:
                self.ob = self.env.reset()
                n_episodes += 1
                self.current_timestep = 0
        return n_episodes

    def nsteps(self):
        """Number of steps taken each sample."""
        return self.update_every


class SamplerFlags(Flags):
    """Sampling settings"""

    def __init__(self):
        arguments = [
            ArgSpec(
                name='sample_interval',
                type=int,
                default=100,
                help='how many timesteps to collect at a time (all parameter '
                'and statistics updates occur in between collection '
                'intervals)')
        ]
        super().__init__('sampler', 'sampler', arguments)
