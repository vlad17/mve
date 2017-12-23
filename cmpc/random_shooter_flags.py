"""
Specify the sampling distribution for a random shooter.

The random shooter might be unconstrained or constrained by changing
the support of its sample space. See RandomShooterFlags.
"""

from mpc_flags import MPCFlags, ArgSpec
from random_shooter import RandomShooter
from cloning_learner import CloningLearner
from ddpg_learner import DDPGLearner
from zero_learner import ZeroLearner


class RandomShooterFlags(MPCFlags):
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

    We consider several forms of "learners":

    * rs - no learner (opt_horizon == mpc_horizon)
    * rs_cloning - train a learner to mimic past actions by the controller
    * rs_zero - the learner will always perform the action 0
    * rs_ddpg - the learner is a policy trained by DDPG
    """

    def __init__(self, rs_learner):
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
                help='number of simulated MPC rollouts')]

        pretty_name = RandomShooterFlags._pretty_name_from_rs(rs_learner)
        extra_args = RandomShooterFlags._learner_arguments(rs_learner)
        super().__init__(rs_learner, pretty_name, arguments + extra_args)
        self._rs_learner = rs_learner

    def make_mpc(self, env, dyn_model, all_flags):
        discount = all_flags.experiment.discount
        mpc_horizon = all_flags.mpc.mpc_horizon
        rsclass = RandomShooterFlags._get_learner_class(self._rs_learner)
        learner = rsclass(env, self)
        return RandomShooter(
            env, dyn_model, discount, learner, mpc_horizon, self)

    @staticmethod
    def _get_learner_class(rs_learner):
        def _no_learner(_env, _flags):
            return None
        _no_learner.FLAGS = []
        mapping = {
            'rs': _no_learner,
            'rs_cloning': CloningLearner,
            'rs_zero': ZeroLearner,
            'rs_ddpg': DDPGLearner}
        return mapping[rs_learner]

    @staticmethod
    def _learner_arguments(rs_learner):
        rsclass = RandomShooterFlags._get_learner_class(rs_learner)
        return rsclass.FLAGS

    @staticmethod
    def _pretty_name_from_rs(rs_learner):
        mapping = {
            'rs': 'unconstrained random shooter',
            'rs_cloning':
                'random shooter with a behavior-cloning planning constraint',
            'rs_zero': 'random shooter with a null-action planning constraint',
            'rs_ddpg': 'random shooter with a DDPG action planning constraint'}
        return mapping[rs_learner]

    @staticmethod
    def all_subflags():
        """Return RandomShooterFlags corresponding to all possible subflags."""
        names = ['rs', 'rs_cloning', 'rs_zero', 'rs_ddpg']
        return [RandomShooterFlags(name) for name in names]
