"""
Specify the properties of constrained colocation-based planning.
"""

from context import flags
from colocation import Colocation
from flags import Flags, ArgSpec


class ColocationFlags(Flags):
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

    @staticmethod
    def make(dyn_model):
        """Generate the colocation instance"""
        return Colocation(dyn_model)

    def coloc_opt_horizon_with_default(self):
        """
        Return the colocation optimization horizon, which is the
        mpc_horizon by default.
        """
        mpc_horizon = flags().mpc.mpc_horizon
        if self.coloc_opt_horizon is None:
            opt_horizon = mpc_horizon
        else:
            opt_horizon = self.coloc_opt_horizon
        assert opt_horizon <= mpc_horizon, (
            opt_horizon, mpc_horizon)
        assert mpc_horizon > 0, mpc_horizon
        assert opt_horizon > 0, opt_horizon
        return opt_horizon
