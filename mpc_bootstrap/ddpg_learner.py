"""
A learner which uses DDPG: an off-policy RL algorithm based on
policy-gradients.
"""


import numpy as np
import tensorflow as tf

from learner import Learner
from utils import create_random_tf_policy, get_ac_dim
from ddpg.main import mkagent, train


class DDPGLearner(Learner):  # pylint: disable=too-many-instance-attributes
    """
    Use a DDPG agent to learn. Not compatible with dagger.
    """

    def __init__(self, env, data, sess, ddpg_flags):
        self._agent = mkagent(env, data, ddpg_flags)
        self._initted = False
        self._sess = sess
        self._env = env
        self._epochs = ddpg_flags.con_epochs
        self._batch_size = ddpg_flags.con_batch_size
        self._param_noise = ddpg_flags.param_noise_exploration
        self._action_noise = ddpg_flags.action_noise_exploration
        self._param_noise_act = ddpg_flags.param_noise_exploitation

    def _init(self):
        if not self._initted:
            self._agent.initialize(self._sess)
            self._initted = True

    def tf_action(self, states_ns, is_initial=False):
        acs = self._agent.actor(states_ns, reuse=True)
        acs *= self._env.action_space.high
        if is_initial:
            if self._action_noise > 0:
                perturb = tf.random_normal(tf.shape(acs)) * self._action_noise
                acs += perturb
                acs = tf.minimum(acs, self._env.action_space.high)
                acs = tf.maximum(acs, self._env.action_space.low)
                return acs
            if self._param_noise:
                return self._agent.tf_n_perturbed(states_ns)
            random_policy = create_random_tf_policy(self._env.action_space)
            return random_policy(states_ns)
        return acs

    def act(self, states_ns):
        self._init()
        rws = np.zeros(len(states_ns))
        if self._param_noise_act:
            acs = np.empty((len(states_ns), get_ac_dim(self._env)))
            for i, state in enumerate(states_ns):
                ac = self._agent.pi([state], apply_noise=True,
                                    compute_Q=False)[0][0]
                acs[i] = ac
        else:
            acs = self._agent.pi(states_ns, apply_noise=False,
                                 compute_Q=False)[0]
        acs *= self._env.action_space.high
        return acs, rws

    def fit(self, data):
        """Fit the learner to the specified labels."""
        self._init()
        # TODO: consider training for 0.5 epochs or even a fixed amount each
        # iter
        nexamples = self._epochs * len(data.stationary_obs())
        nbatch = max(nexamples // self._batch_size, 1)
        train(self._env, self._agent, nb_iterations=nbatch)

    def reset(self, nstates):
        self._init()
        # applies random noise to the actor for the next set of
        # nstates rollouts
        self._agent.reset()
