"""
Imagine we generate a big set of rollouts using some policy. This gives us a
big set of states labeled with their corresponding actions. A
CloningLearner is a more or less a neural network that uses this data to
predict an action for a given state, framing this table as a supervised
machine learning problem.
"""


import tensorflow as tf

from context import flags
import env_info
from flags import ArgSpec, Flags
from learner import Learner
from utils import build_mlp, scale_to_box


class CloningLearnerFlags(Flags):
    """Specification for cloning learner"""

    def __init__(self):
        arguments = [
            ArgSpec(
                name='cloning_learner_depth',
                type=int,
                default=5,
                help='learned controller NN depth'),
            ArgSpec(
                name='cloning_learner_width',
                type=int,
                default=32,
                help='learned controller NN width'),
            ArgSpec(
                name='cloning_learner_lr',
                type=float,
                default=1e-3,
                help='learned controller NN learning rate'),
            ArgSpec(
                name='cloning_learner_batches_per_timestep',
                type=float,
                default=4,
                help='number of mini-batches to train dynamics per '
                'new sample observed'),
            ArgSpec(
                name='cloning_learner_batch_size',
                type=int,
                default=512,
                help='learned controller batch size')]
        super().__init__('cloning', 'cloning learner', arguments)

    def nbatches(self, timesteps):
        """The number training batches, given this many timesteps."""
        nbatches = self.cloning_learner_batches_per_timestep * timesteps
        nbatches = max(int(nbatches), 1)
        return nbatches


class CloningLearner(Learner):
    """
    Mimics the controller by optimizing for L2 loss on the predicted actions.

    Creates a behavior-cloned policy, essentially.
    """

    def __init__(self):
        # placeholders for behavior-cloning training
        # a = action dim
        # s = state dim
        # n = batch size
        self._input_state_ph_ns = tf.placeholder(
            tf.float32, [None, env_info.ob_dim()])
        self._policy_action_na = self.tf_action(self._input_state_ph_ns)
        self._expert_action_ph_na = tf.placeholder(
            tf.float32, [None, env_info.ac_dim()])
        mse = tf.losses.mean_squared_error(
            self._expert_action_ph_na,
            self._policy_action_na)

        self._update_op = tf.train.AdamOptimizer(
            flags().cloning.cloning_learner_lr).minimize(mse)

    def tf_action(self, states_ns):
        ac_na = build_mlp(
            states_ns, scope='cloning_learner',
            n_layers=flags().cloning.cloning_learner_depth,
            size=flags().cloning.cloning_learner_width,
            activation=tf.nn.relu,
            output_activation=tf.sigmoid, reuse=tf.AUTO_REUSE)
        return scale_to_box(env_info.ac_space(), ac_na)

    def fit(self, data, timesteps):
        nbatches = flags().cloning.nbatches(timesteps)
        batch_size = flags().cloning.cloning_learner_batch_size
        for batch in data.sample_many(nbatches, batch_size):
            obs, _, _, acs, _ = batch
            tf.get_default_session().run(
                self._update_op, feed_dict={
                    self._input_state_ph_ns: obs,
                    self._expert_action_ph_na: acs})

    def act(self, states_ns):
        return tf.get_default_session().run(self._policy_action_na, feed_dict={
            self._input_state_ph_ns: states_ns})
