"""
Specify the properties of constrained colocation-based planning.
"""

from mpc_flags import MPCFlags, ArgSpec
from colocation import Colocation


class ColocationFlags(MPCFlags):
    """
    Specifies various optimizer properties for the colocation-based
    optimization.
    """

    @staticmethod
    def _generate_arguments():
        yield ArgSpec(
            name='coloc_opt_horizon',
            type=int,
            default=None,
            help='Requires using MPC with a learner, can be at most '
            'mpc_horizon. Plan for next mpc_horizon steps but only '
            'optimize over first opt_horizon actions')
        yield ArgSpec(
            name='coloc_primal_steps',
            type=int,
            default=500,
            help='maximum primal steps')
        yield ArgSpec(
            name='coloc_dual_steps',
            type=int,
            default=1,
            help='maximum outer-loop dual steps')
        yield ArgSpec(
            name='coloc_primal_tol',
            type=float,
            default=1e-2, help='primal solving tolerance')
        yield ArgSpec(
            name='coloc_primal_lr',
            type=float,
            default=1e-3, help='primal solving lr')
        yield ArgSpec(
            name='coloc_dual_lr',
            type=float,
            default=1e-2, help='dual solving lr')

    def __init__(self):
        super().__init__('colocation', 'colocation-based planning',
                         list(ColocationFlags._generate_arguments()))

    def make_mpc(self, env, dyn_model, all_flags):
        discount = all_flags.experiment.discount
        mpc_horizon = all_flags.mpc.mpc_horizon
        return Colocation(
            env, dyn_model, discount, mpc_horizon, self)
