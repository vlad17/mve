"""
Specify the sampling distribution for a random shooter.

The random shooter might be unconstrained or constrained by changing
the support of its sample space. See RandomShooterFlags.
"""

from context import flags
from flags import Flags, ArgSpec
from random_shooter import RandomShooter
from random_shooter_with_true_dynamics import RandomShooterWithTrueDynamics
from cloning_learner import CloningLearner
from ddpg_learner import DDPGLearner
from zero_learner import ZeroLearner


class RandomShooterFlags(Flags):
    """
    Specifies the random shooter sampling distribution.

    A random shooter weakly optimizes over the MPC horizon, where
    this optimization can be done over a constrained state space by
    restricting the support of the sampling distribution.

    These sampling distributions can be learned. In particular,
    for sampling actions a_1, ..., a_H (corresponding to timesteps
    1 to H, the MPC horizon), we consider distributions that
    are uniform on the entire action space for the first several
    actions (those whose index falls below the "optimization horizon")
    and the rest of the actions are sampled according to a learned
    deterministic function.

    In other words, the state-conditional action distribution for
    the actions a_i for a given set of states s_1, ..., s_H are:

        a_i ~ Uniform(action space) for i < opt_horizon
        a_j ~ DiracDelta(learner(s_j)) for opt_horizon <= j < mpc_horizon

    Where the learner is a deterministic mapping from states to actions.

    Note that the learner doesn't affect the MPC if the
    optimization horizon is equal to the MPC horizon, which is the
    default setting.
    """

    def __init__(self):
        arguments = [
            ArgSpec(
                name='opt_horizon',
                type=int,
                default=None,
                help='Requires using MPC with a learner, can be at most '
                'mpc_horizon. Plan for next mpc_horizon steps but only '
                'optimize over first opt_horizon actions'),
            ArgSpec(
                name='simulated_paths',
                type=int,
                default=1000,
                help='number of simulated MPC rollouts'),
            ArgSpec(
                name='rs_learner',
                type=str,
                default='zero',
                help='learner type, one of {zero, ddpg, cloning}'),
            ArgSpec(
                name='true_dynamics',
                action='store_true',
                help='Indicates whether or not to use true dynamics, the OpenAI'
                ' gym, as opposed to a learned model'
            )]
        super().__init__('random_shooter', 'random shooter', arguments)

    def make(self, dyn_model):
        """Return a random shooter based on the given dynamics model"""
        if self.true_dynamics:
            return RandomShooterWithTrueDynamics()
        return RandomShooter(dyn_model)

    def make_learner(self):
        """Return a learner based on the current flags from the context."""
        if self.rs_learner == 'zero':
            return ZeroLearner()
        elif self.rs_learner == 'ddpg':
            return DDPGLearner()
        elif self.rs_learner == 'cloning':
            return CloningLearner()
        else:
            raise ValueError('random shooter learner {} unrecognized'.format(
                self.rs_learner))

    def opt_horizon_with_default(self):
        """
        If the opt_horizon was not set, use the mpc_horizon as a default.
        Otherwise, return the usual optimization horizon.
        """
        if self.opt_horizon is None:
            return flags().mpc.mpc_horizon
        return self.opt_horizon
