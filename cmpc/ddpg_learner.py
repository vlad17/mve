"""
A learner which uses DDPG: an off-policy RL algorithm based on
policy-gradients.
"""

import tensorflow as tf

from flags import ArgSpec
from learner import Learner
from ddpg.main import mkagent, train


class DDPGLearner(Learner):
    """
    Use a DDPG agent to learn.
    """

    FLAGS = [
        ArgSpec(
            name='actor_lr',
            type=float,
            default=1e-4, help='actor network learning rate'),
        ArgSpec(
            name='critic_lr',
            type=float,
            default=1e-4, help='critic network learning rate'),
        ArgSpec(
            name='critic_l2_reg',
            type=float,
            default=1e-2,
            help='DDPG critic regularization constant'),
        ArgSpec(
            name='learner_depth',
            type=int,
            default=2,
            help='depth for both actor and critic networks'),
        ArgSpec(
            name='learner_width',
            type=int,
            default=64,
            help='width for both actor and critic networks'),
        ArgSpec(
            name='learner_nbatches',
            default=None,
            type=int,
            help='number of minibatches to train on per iteration'),
        ArgSpec(
            name='learner_batch_size',
            default=512,
            type=int,
            help='number of minibatches to train on per iteration')]

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
        acs = self._agent.pi(states_ns, apply_noise=False, compute_Q=False)[0]
        acs *= self._env.action_space.high
        return acs

    def fit(self, data):
        """Fit the learner to the specified labels."""
        self._init()
        train(self._env, self._agent, data,
              nb_iterations=self._flags.learner_nbatches)
