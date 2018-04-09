"""
Rescales actions to and from [-1, 1]
"""

import numpy as np


def _finite_env(space):
    return all(np.isfinite(space.low)) and all(np.isfinite(space.high))


def scale_from_unit(space, relative):
    """
    Given a hyper-rectangle specified by a gym box space, scale
    relative coordinates between -1 and 1 to the box's coordinates,
    such that the relative vector of all zeros has the smallest
    coordinates (the "bottom left" corner) and vice-versa for ones.

    If environment is infinite, no scaling is performed.
    """
    if not _finite_env(space):
        return relative
    relative += 1
    relative /= 2
    relative *= (space.high - space.low)
    relative += space.low
    return relative


def scale_to_unit(space, absolute):
    """
    Given a hyper-rectangle specified by a gym box space, scale
    exact coordinates from within that space to
    relative coordinates between -1 and 1.

    If environment is infinite, no scaling is performed.
    """
    if not _finite_env(space):
        return absolute
    absolute -= space.low
    absolute /= (space.high - space.low)
    return absolute * 2 - 1
