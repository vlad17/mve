"""
A learner which uses SAC
"""
import distutils.util

from agent import Agent
from context import flags
from flags import Flags, ArgSpec
from learner import Learner
from tfnode import TFNode
from sac.sac import SAC
from sac.stochastic_policy import SquashedGaussianPolicy
from sac.value import QFunction, VFunction


class SACFlags(Flags):
    """SAC settings"""

    def __init__(self):
        arguments = [
            ArgSpec(
                name='policy_lr',
                type=float,
                default=1e-3, help='policy network learning rate'),
            ArgSpec(
                name='value_lr',
                type=float,
                default=1e-3, help='qfn,vfn networks learning rate'),
            ArgSpec(
                name='imaginary_buffer',
                type=float,
                default=0.0,
                help='Use imaginary data in a supplementary buffer, '
                'for TD-1 training. This flag specifies the ratio of '
                'imaginary to real data that should be collected (rollouts '
                'are of length H). The same ratio specifies the training '
                'rate on imaginary to real data.'),
            ArgSpec(
                name='learner_depth',
                type=int,
                default=2,
                help='depth for both policy, qfn, vfn networks'),
            ArgSpec(
                name='temperature',
                type=float,
                default=1.0,
                help='maximum entropy temperature setting'),
            ArgSpec(
                name='learner_width',
                type=int,
                default=64,
                help='width for both policy, qfn, vfn networks'),
            ArgSpec(
                name='learner_batches_per_timestep',
                default=4,
                type=float,
                help='number of mini-batches to train dynamics per '
                'new sample observed'),
            ArgSpec(
                name='learner_batch_size',
                default=2056,
                type=int,
                help='number of minibatches to train on per iteration'),
            ArgSpec(
                name='sac_mve',
                default=False,
                type=distutils.util.strtobool,
                help='Use the mixture estimator instead of the target values'),
            ArgSpec(
                name='model_horizon',
                default=1,
                type=int,
                help='how many steps to expand Q estimates dynamics'),
            ArgSpec(
                name='restore_sac',
                default=None,
                type=str,
                help='restore sac from the given path'),
            ArgSpec(
                name='sac_min_buf_size',
                default=1,
                type=int,
                help='Minimum number of frames in replay buffer before '
                     'training'),
            ArgSpec(
                name='drop_tdk',
                default=False,
                type=distutils.util.strtobool,
                help='By default model-based value expansion corrects for '
                'off-distribution error with the TD-k trick. This disables '
                'use of the trick for diagnostics training'),
            ArgSpec(
                name='vfn_target_rate',
                type=float,
                default=0.01,
                help='target update rate for value function'),
        ]
        super().__init__('sac', 'SAC', arguments)

    def nbatches(self, timesteps):
        """The number training batches, given this many timesteps."""
        nbatches = self.learner_batches_per_timestep * timesteps
        nbatches = max(int(nbatches), 1)
        return nbatches

    def expect_dynamics(self, dyn):
        """
        Check if a dynamics model was provided in learned value mixture
        estimation.
        """
        if self.sac_mve or self.imaginary_buffer > 0:
            assert dyn is not None, 'expecting a dynamics model'
        else:
            assert dyn is None, 'should not be getting a dynamics model'


class SACLearner(Learner, TFNode):
    """
    Use a SAC agent to learn. The learner's tf_action
    gives its mean policy, but when the learner acts it samples
    from a maxent policy.
    """

    def __init__(self, dynamics=None):
        flags().sac.expect_dynamics(dynamics)
        self._batch_size = flags().sac.learner_batch_size

        self.policy = SquashedGaussianPolicy()
        self.qfn = QFunction()
        self.vfn = VFunction()
        self._sac = SAC(self.policy, self.qfn, self.vfn, dynamics)
        TFNode.__init__(self, 'sac', flags().sac.restore_sac)

    def agent(self):
        return Agent.wrap(
            self.policy.act,
            self.policy.greedy_act)

    def train(self, data, timesteps):
        if data.size < flags().sac.sac_min_buf_size:
            return
        nbatches = flags().sac.nbatches(timesteps)
        self._sac.train(data, nbatches, self._batch_size, timesteps)

    def evaluate(self, data):
        self._sac.evaluate(data)
