"""
Contains metadata relating to an environment. These are useful constants
that need to be passed around everywhere but are instead maintained in
this single metadata class.
"""

from contextlib import contextmanager, closing

from context import context
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


@contextmanager
def create(make_env):
    """
    Create an environment information instance within a given context.
    """
    assert context().env_info is None, 'env info already exists'
    with closing(make_env()) as env:
        context().env_info = env
        yield
    context().env_info = None
