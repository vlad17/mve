"""
A learner which uses DDPG: an off-policy RL algorithm based on
policy-gradients.
"""


import numpy as np
import tensorflow as tf

from learner import Learner
from ddpg.main import mkagent, train


class DDPGLearner(Learner):
    """
    Use a DDPG agent to learn.
    """

    def __init__(self, env, ddpg_flags):
        self._agent = mkagent(env, ddpg_flags)
        self._initted = False
        self._env = env
        self._flags = ddpg_flags

    def _init(self):
        if not self._initted:
            self._agent.initialize(tf.get_default_session())
            self._initted = True

    def tf_action(self, states_ns):
        acs = self._agent.actor(states_ns, reuse=tf.AUTO_REUSE)
        # TODO: the underlying ddpg implementation assumes that the
        # actions are symmetric... Need to fix this assumption in
        # lots of places.
        acs *= self._env.action_space.high
        return acs

    def act(self, states_ns):
        self._init()
        rws = np.zeros(len(states_ns))
        acs = self._agent.pi(states_ns, apply_noise=False, compute_Q=False)[0]
        acs *= self._env.action_space.high
        return acs, rws

    def fit(self, data):
        """Fit the learner to the specified labels."""
        self._init()
        train(self._env, self._agent, data, nb_iterations=self._flags.nbatches)
