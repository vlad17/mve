"""
Classic Acrobot task.

This, like FullyObservableHalfCheetah, is adapted from gym. However, the
observed values are not modified in this scenario.

The differences from the gym implementation of Acrobot (at its commit
4c460ba6c8959dd8e0a03b13a1ca817da6d4074f) are mainly the following:

* numba acceleration for computation
* allow for a continuous range of control
"""

import math
from gym import core, spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
from scipy import integrate
from numba import jit, prange
import tensorflow as tf
from .fully_observable import FullyObservable

from .vector_env import VectorEnv

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

# The above was then copied from openai.gym


class VectorizedContinuousAcrobot(VectorEnv, FullyObservable):
    """
    2-link pendulum as described in Sutton and Barto (vectorized
    implementation)

    Acrobot is a 2-link pendulum with only the second joint actuated.

    Intitially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.

    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities.
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].

    thetaDot corresponds to the angular velocity of the corresponding joint.

    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.

    The action is a deterministic choice from a "preference space" [0,1]^3.
    If the i-th entry of the action is the maximum argument, then
    the torque [-1, 0, 1][i] is applied. This could be made into a probibalisitc
    choice by softmaxing the vector but then the dynamics wouldn't be determinsitic.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    dt = .2

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self, n):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high
        self.n = n
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(
            low=np.zeros(3),
            high=np.ones(3))
        self.state = np.empty((n, 4))  # 4 non-control latent state dims
        # only do one step of rk4 integration
        # could in theory make this [0, self.dt / N, 2 * self.dt / N, ...]
        self._integration_ts = np.array([0, self.dt])
        self._work_ks = np.empty((n, 4, 5))  # Runge-Kutta workspace coeffs
        # need 4 vectors (axis 0)
        # with a dim for each state (4 dims + one action/control dimension)
        self._seed_uncorr(n)

    def tf_reward(self, state, action, next_state):
        """
        Given tensors(with the 0th dimension as the batch dimension) for a
        transition during a rollout, this returns the corresponding reward
        as a rank - 1 tensor(vector).
        """
        theta0 = tf.atan2(next_state[:, 1], next_state[:, 0])
        theta1 = tf.atan2(next_state[:, 3], next_state[:, 2])
        height = -tf.cos(theta0) - tf.cos(theta0 + theta1)
        return -1. * tf.to_float(height <= 1.)

    def np_reward(self, state, action, next_state):
        """
        Numpy analogoue for tf_reward.
        """
        theta0 = np.arctan2(next_state[:, 1], next_state[:, 0])
        theta1 = np.arctan2(next_state[:, 3], next_state[:, 2])
        height = -tf.cos(theta0) - tf.cos(theta0 + theta1)
        return -1 * (height <= 1.)

    def _seed(self, seed=None):
        seeds = seed
        self.np_random, seeds = zip(*[seeding.np_random(x) for x in seeds])
        return seeds

    def _reset(self):
        self.state = np.asarray([
            np_random.uniform(low=-0.1, high=0.1, size=(4,))
            for np_random in self.np_random])
        return self._get_ob()

    def set_state_from_ob(self, ob):
        s0 = np.arctan2(ob[:, 1], ob[:, 0])
        s1 = np.arctan2(ob[:, 3], ob[:, 2])
        self.state = np.array([s0, s1, ob[:, 4], ob[:, 5]]).T

        # @profile
    def _step(self, a):
        s = self.state
        preferences = np.asarray(a)
        torque = np.array([-1., 0., 1.])[preferences.argmax(axis=1)].reshape(-1, 1)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        ns = np.concatenate([s, torque], axis=1)

        self._rk4(ns)
        ns = ns[:, :4]

        ns[:, 0] = wrap(ns[:, 0], -pi, pi)
        ns[:, 1] = wrap(ns[:, 1], -pi, pi)
        ns[:, 2] = bound(ns[:, 2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[:, 3] = bound(ns[:, 3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._height() > 1
        reward = -1 * (1 - terminal)
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([
            cos(s[:, 0]), np.sin(s[:, 0]),
            cos(s[:, 1]), sin(s[:, 1]),
            s[:, 2], s[:, 3]]).T

    def _height(self):
        s = self.state
        return -np.cos(s[:, 0]) - np.cos(s[:, 1] + s[:, 0])

    def close(self):
        if self.viewer:
            self.viewer.close()

            #@profile

    def _rk4(self, y0):
        """
        Runge-Kutta, 4th-order. only returns endpoint
        """
        return rk4_acro(y0, self._integration_ts, self._work_ks)


class ContinuousAcrobot(core.Env, FullyObservable):

    def __init__(self):
        self._venv = VectorizedContinuousAcrobot(1)
        self.observation_space = self._venv.observation_space
        self.action_space = self._venv.action_space

    def _seed(self, seed=None):
        return self._venv.seed([seed])[0]

    def _reset(self):
        return self._venv.reset()[0]

    def _step(self, action):
        ob, rew, done, _ = self._venv.step(np.asarray(action)[np.newaxis, ...])
        return ob[0], rew[0], done[0], {}

    def _close(self):
        self._venv.close()

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        if mode == 'human':
            return None
        s = self._venv.state[0]

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

        if s is None:
            return None

        LINK_LENGTH_1 = 1.
        LINK_LENGTH_2 = 1.

        p1 = [-LINK_LENGTH_1 *
              np.cos(s[0]), LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, :: -1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th) in zip(xys, thetas):
            l, r, t, b = 0, 1, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=True)


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    # TODO being lazy here, use appropriate numpy mod
    while np.any(x > M):
        x[x > M] = x[x > M] - diff
    while np.any(x < m):
        x[x < m] = x[x < m] + diff
    return x


def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return np.minimum(np.maximum(x, m), M)


@jit('void(double[:], double[:])', nopython=True, nogil=True)
def _dsdt(s_augmented, out):
    LINK_LENGTH_1 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    m1 = LINK_MASS_1
    m2 = LINK_MASS_2
    l1 = LINK_LENGTH_1
    lc1 = LINK_COM_POS_1
    lc2 = LINK_COM_POS_2
    I1 = LINK_MOI
    I2 = LINK_MOI
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[: -1]
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]
    d1 = m1 * lc1 ** 2 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
        + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
    ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
        / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    out[0] = dtheta1
    out[1] = dtheta2
    out[2] = ddtheta1
    out[3] = ddtheta2
    out[4] = 0.


@jit('void(double[:, :], double[:], double[:,:,:])', nopython=True, nogil=True)
def rk4_acro(y0, ts, all_work_ks):
    """
    RK4 integration procedure that returns the last point at the end of the ts.

    Specialized to acrobot derivative.

    the shape of ys is iters x d where d is the ODE dim and iters are the various
    instances of the same problem we're solving.

    Work_ks should be an array for scratch work of dimension iters x 4 x d

    returns the endpoint in-place by editing the initial condition
    """
    # make multiple work_ks; parallelize numba?
    for k in range(len(y0)):
        y = y0[k]
        derivs = _dsdt
        work_ks = all_work_ks[k]
        for i in range(len(ts) - 1):
            thist = ts[i]
            dt = ts[i + 1] - thist
            dt2 = dt / 2.0

            derivs(y, work_ks[0])
            derivs(y + dt2 * work_ks[0], work_ks[1])
            derivs(y + dt2 * work_ks[1], work_ks[2])
            derivs(y + dt * work_ks[2], work_ks[3])
            y += dt / 6.0 * (work_ks[0] + 2 * work_ks[1]
                             + 2 * work_ks[2] + work_ks[3])
