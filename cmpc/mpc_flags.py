"""MPC flags shared between all palnners"""

from flags import Flags, ArgSpec
from context import flags


class MPCFlags(Flags):
    """
    Top-level MPC flags, used regardless of the planner for managing
    options common to all MPC experiments.
    """

    def __init__(self):
        arguments = [
            ArgSpec(
                name='onpol_iters',
                type=int,
                default=1,
                help='number of outermost on policy aggregation iterations'),
            ArgSpec(
                name='onpol_paths',
                type=int,
                default=1,
                help='number of rollouts per on policy iteration',),
            ArgSpec(
                name='mpc_horizon',
                type=int,
                default=15,
                help='horizon of simulated MPC rollouts'),
            ArgSpec(
                name='mpc_optimizer',
                type=str,
                default='random_shooter',
                help='optimizer for MPC, one of {random_shooter, colocation}')]
        pretty_name = 'common model-predictive control settings'
        super().__init__('mpc', pretty_name, arguments)

    def make_mpc(self, dyn_model):
        """
        Generate an MPC instance according to the specification provided
        by the MPC flags instance based on the given dynamics model.
        """
        if self.mpc_optimizer == 'random_shooter':
            return flags().random_shooter.make(dyn_model)
        elif self.mpc_optimizer == 'colocation':
            return flags().colocation.make(dyn_model)
        else:
            raise ValueError('mpc optimizer {} unrecognized'.format(
                self.mpc_optimizer))
