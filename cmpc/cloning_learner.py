"""
Imagine we generate a big set of rollouts using some policy. This gives us a
big set of states labeled with their corresponding actions. A
CloningLearner is a more or less a neural network that uses this data to
predict an action for a given state, framing this table as a supervised
machine learning problem.
"""


import tensorflow as tf
import numpy as np

from learner import Learner
from utils import (build_mlp, get_ac_dim, get_ob_dim)
from learner_flags import LearnerFlags


class CloningLearnerFlags(LearnerFlags):
    """
    Specify the hyperparameters of the behavior-cloning neural network.
    """

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Adds flags to an argparse parser."""
        if argument_group is None:
            argument_group = parser.add_argument_group('cloning learner')
        argument_group.add_argument(
            '--learner_depth',
            type=int,
            default=5,
            help='learned controller NN depth',
        )
        argument_group.add_argument(
            '--learner_width',
            type=int,
            default=32,
            help='learned controller NN width',
        )
        argument_group.add_argument(
            '--learner_lr',
            type=float,
            default=1e-3,
            help='learned controller NN learning rate',
        )
        argument_group.add_argument(
            '--learner_nbatches',
            type=int,
            default=4000,
            help='learned controller training minibatches',
        )
        argument_group.add_argument(
            '--learner_batch_size',
            type=int,
            default=512,
            help='learned controller batch size',
        )
        return argument_group

    @staticmethod
    def name():
        return 'cloner'

    def __init__(self, args):
        self.depth = args.learner_depth
        self.width = args.learner_width
        self.lr = args.learner_lr
        self.nbatches = args.learner_nbatches
        self.batch_size = args.learner_batch_size

    def make_learner(self, env):
        return CloningLearner(env, self)


class CloningLearner(Learner):
    """
    Mimics the controller by optimizing for L2 loss on the predicted actions.

    Creates a behavior-cloned policy, essentially.
    """

    def __init__(self, env, flags):
        self._flags = flags
        self._ac_dim = get_ac_dim(env)
        self._ac_space = env.action_space

        # placeholders for behavior-cloning training
        # a = action dim
        # s = state dim
        # n = batch size
        self._input_state_ph_ns = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)])
        self._policy_action_na = self.tf_action(self._input_state_ph_ns)
        self._expert_action_ph_na = tf.placeholder(
            tf.float32, [None, self._ac_dim])
        mse = tf.losses.mean_squared_error(
            self._expert_action_ph_na,
            self._policy_action_na)

        self._update_op = tf.train.AdamOptimizer(
            self._flags.lr).minimize(mse)

    def tf_action(self, states_ns):
        ac_na = build_mlp(
            states_ns, scope='cloning_learner',
            n_layers=self._flags.depth, size=self._flags.width,
            activation=tf.nn.relu,
            output_activation=tf.sigmoid, reuse=tf.AUTO_REUSE)
        ac_na *= self._ac_space.high - self._ac_space.low
        ac_na += self._ac_space.low
        return ac_na

    def fit(self, data):
        for batch in data.sample_many(
                self._flags.nbatches, self._flags.batch_size):
            obs, _, _, acs, _ = batch
            tf.get_default_session().run(
                self._update_op, feed_dict={
                    self._input_state_ph_ns: obs,
                    self._expert_action_ph_na: acs})

    def act(self, states_ns):
        acs = tf.get_default_session().run(self._policy_action_na, feed_dict={
            self._input_state_ph_ns: states_ns})
        rws = np.zeros(len(states_ns))
        return acs, rws
