"""
Contains metadata relating to an environment. These are useful constants
that need to be passed around everywhere but are instead maintained in
this single metadata class.
"""

import os
import weakref

from contextlib import contextmanager, closing
from gym.core import RewardWrapper
import hashlib
import numpy as np

from context import context
import envs
from utils import get_ob_dim, get_ac_dim


def reward_fn():
    """The specified environment's reward function."""
    return context().env_info.env.tf_reward


def ob_dim():
    """The specified environment's observation dimension."""
    return get_ob_dim(context().env_info.env)


def ac_dim():
    """The specified environment's action dimension."""
    return get_ac_dim(context().env_info.env)


def ac_space():
    """The specified environment's action space."""
    return context().env_info.env.action_space


def ob_space():
    """The specified environment's observation space."""
    return context().env_info.env.observation_space


def _env_class():
    env_name = context().flags.experiment.env_name
    if env_name == 'hc':
        return envs.FullyObservableHalfCheetah
    elif env_name == 'ant':
        return envs.FullyObservableAnt
    elif env_name == 'walker2d':
        return envs.FullyObservableWalker2d
    elif env_name == 'hc2':
        import gym2
        return gym2.FullyObservableHalfCheetah
    elif env_name == 'pusher':
        return envs.FullyObservablePusher
    elif env_name == 'hopper':
        return envs.FullyObservableHopper
    elif env_name == 'swimmer':
        return envs.FullyObservableSwimmer
    elif env_name == 'acrobot':
        import envs.acrobot as acrobot  # see note in envs.__init__
        os.environ['OMP_NUM_THREADS'] = str(
            context().flags.experiment.env_parallelism)
        return acrobot.ContinuousAcrobot
    elif env_name == 'humanoid':
        return envs.FullyObservableHumanoid
    else:
        raise ValueError('env {} unsupported'.format(env_name))


def _next_seeds(n):
    # deterministically generate seeds for envs
    # not perfect due to correlation between generators,
    # but we can't use urandom here to have replicable experiments
    # https://stats.stackexchange.com/questions/233061
    mt_state_size = 624
    seeds = []
    for _ in range(n):
        state = np.random.randint(2 ** 32, size=mt_state_size)
        digest = hashlib.sha224(state.tobytes()).digest()
        seed = np.frombuffer(digest, dtype=np.uint32)[0]
        seeds.append(int(seed))
        if seeds[-1] is None:
            seeds[-1] = int(state.sum())
    return seeds


def make_env():
    """
    Generates an unvectorized env from a standard string.
    Creates the experiment-flag specified name by default.
    """
    env = _env_class()()
    env.seed(_next_seeds(1)[0])
    context().env_info.add_env(env)
    return _RewardScalingWrapper(env)


def make_venv(n):
    """
    Same as make_env, but returns a vectorized version of the env.
    """
    if context().flags.experiment.env_name == 'hc2':
        import gym2
        venv = gym2.VectorMJCEnv(
            n, _env_class(),
            max_threads=context().flags.experiment.env_parallelism)
    elif context().flags.experiment.env_name == 'acrobot':
        import envs.acrobot as acrobot  # see note in envs.__init__
        os.environ['OMP_NUM_THREADS'] = str(
            context().flags.experiment.env_parallelism)
        venv = acrobot.VectorizedContinuousAcrobot(n)
    else:
        venv = envs.ParallelGymVenv(
            n, _env_class(),
            parallelism=context().flags.experiment.env_parallelism)
    venv.seed(_next_seeds(n))
    context().env_info.add_env(venv)
    return _RewardScalingWrapper(venv)


class _EnvInfo:
    def __init__(self, env):
        self.env = env
        self._all_envs = []

    def close(self):
        """Close all generated envs"""
        for env in self._all_envs:
            strong_env = env()
            if strong_env is not None:
                strong_env.close()

    def add_env(self, env):
        """Add an env to the garbage-collection list"""
        self._all_envs.append(weakref.ref(env))


@contextmanager
def create():
    """
    Create an environment information instance within a given context.
    """
    assert context().env_info is None, 'env info already exists'
    with closing(_env_class()()) as env:
        env.seed(_next_seeds(1)[0])
        with closing(_EnvInfo(env)) as env_info:
            context().env_info = env_info
            yield
    context().env_info = None


class _RewardScalingWrapper(RewardWrapper):
    """Wrapper that scales rewards based on environments."""

    def __init__(self, *args, **kwargs):
        super(_RewardScalingWrapper, self).__init__(*args, **kwargs)
        self.reward_scale = _get_reward_scale()

    def _reward(self, reward):
        return reward * self.reward_scale

    def __getattr__(self, item):
        return getattr(self.env, item)


def _get_reward_scale():
    """Get reward scaling based on current environment and reward scaling."""
    env_name = context().flags.experiment.env_name
    reward_scaling = context().flags.experiment.reward_scaling
    if reward_scaling <= 0:
        return _get_reward_scale_by_env(env_name)
    return reward_scaling


def _get_reward_scale_by_env(env_name):
    """Return default reward scaling for each environment."""
    if env_name == 'hc':
        return 0.05
    elif env_name == 'ant':
        return 0.3
    elif env_name == 'walker':
        return 0.05
    return 1.0
