"""
Contains metadata relating to an environment. These are useful constants
that need to be passed around everywhere but are instead maintained in
this single metadata class.
"""

from contextlib import contextmanager, closing

from context import context
import envs
from utils import get_ob_dim, get_ac_dim


def reward_fn():
    """The specified environment's reward function."""
    return context().env_info.tf_reward


def ob_dim():
    """The specified environment's observation dimension."""
    return get_ob_dim(context().env_info)


def ac_dim():
    """The specified environment's action dimension."""
    return get_ac_dim(context().env_info)


def ac_space():
    """The specified environment's action space."""
    return context().env_info.action_space


def ob_space():
    """The specified environment's observation space."""
    return context().env_info.observation_space

def _env_class():
    env_name = context().flags.experiment.env_name
    if env_name == 'hc':
        return envs.FullyObservableHalfCheetah
    elif env_name == 'ant':
        return envs.FullyObservableAnt
    elif env_name == 'walker2d':
        return envs.FullyObservableWalker2d
    else:
        raise ValueError('env {} unsupported'.format(env_name))

def make_env():
    """
    Generates an unvectorized env from a standard string.
    Creates the experiment-flag specified name by default.
    """
    return _env_class()()

def make_venv(n):
    """
    Same as make_env, but returns a vectorized version of the env.
    """
    return envs.ParallelGymVenv(n, _env_class())


@contextmanager
def create():
    """
    Create an environment information instance within a given context.
    """
    assert context().env_info is None, 'env info already exists'
    with closing(make_env()) as env:
        context().env_info = env
        yield
    context().env_info = None
