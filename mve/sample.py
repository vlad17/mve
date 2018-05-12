"""
Utilities for generating rolluts from a vectorized act method
"""

import distutils.util

import numpy as np

from context import flags
from flags import Flags, ArgSpec
from memory import Path
import sys


def sample_venv(venv, act):
    """
    Given a n-way vectorized environment `venv`, generate n paths/rollouts with
    maximum horizon `horizon` using controller `controller`.
    """
    max_horizon = flags().experiment.horizon
    obs_n = venv.reset()
    paths = [Path(venv, obs, max_horizon) for obs in obs_n]
    active_n = np.ones(len(obs_n), dtype=bool)

    # a = number of active environments (not done)
    for _ in range(max_horizon):
        # If there are no active environments, we're done!
        if np.sum(active_n) == 0:
            break

        # If an environment is inactive, we can still ask the actor
        # to give us an action for it (the venv and actor will give garbage
        # but valid-size outputs).
        acs_n = act(obs_n)
        obs_n, reward_n, done_n, _ = venv.step(acs_n)
        for i in np.flatnonzero(active_n):
            done_n[i] |= paths[i].next(
                obs_n[i], reward_n[i], done_n[i], acs_n[i])
            if done_n[i]:
                active_n[i] = False
    return paths


def sample(env, act, render=False):
    """
    Given a single environment env, perform a rollout up to max_horizon
    steps, possibly rendering, with the given controller.

    Is guaranteed to complete an episode before returning a path.
    """
    max_horizon = flags().experiment.horizon
    ob = env.reset()
    path = Path(env, ob, max_horizon)

    for _ in range(max_horizon):
        if render:
            env.render()
        ac = act(ob[np.newaxis, ...])
        ob, reward, done, _ = env.step(ac[0])
        path.next(ob, reward, done, ac[0])
        if done:
            break
    return path


class Sampler:
    """Stateful sampler for shorter periods between parameter updates.

    Is NOT guaranteed to complete any episode or finish at most one episode.

    As a result, when using the sampler, recognize calling `sample` once does
    not correspond to executing a single episode and refactor your game loop
    accordingly.
    """

    def __init__(self, env):
        self.env = env
        self.update_every = flags().sampler.sample_interval
        self.ob = env.reset()
        self.max_horizon = flags().experiment.horizon
        self.current_timestep = 0

    def sample(self, act, data):
        """
        Given a single environment env, perform a rollout up to update_every
        steps, possibly rendering, with the given controller.
        """
        sample_reward = n_episodes = 0

        for _ in range(self.update_every):
            ac = act(self.ob[np.newaxis, ...])
            try:
                next_ob, reward, done, _ = self.env.step(ac[0])
            except:
                print('ACTION', ac[0], file=sys.stderr)
                raise
            self.current_timestep += 1
            if self.current_timestep == self.max_horizon and \
               not flags().sampler.markovian_termination:
                # matches original gym behavior
                # markovian termination is important for SAC
                done = True
            data.next(self.ob, next_ob, reward, done, ac[0])
            sample_reward += reward
            self.ob = next_ob
            if self.current_timestep == self.max_horizon:
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
                'intervals)'),
            ArgSpec(
                name='markovian_termination',
                default=False,
                type=distutils.util.strtobool,
                # The ICML submission did not have this feature, so the DDPG
                # parameters were tuned to this set as "false".
                help='By default, environments are done when they are played '
                'for longer than their timestep limit, but this is not a '
                'Markovian termination condition. If this flag is activated, '
                'environments still reset after the timecap but the done '
                'indicator is not activated in that case.'),
        ]
        super().__init__('sampler', 'sampler', arguments)
