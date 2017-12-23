"""MPC flags shared between all palnners"""

from flags import Flags, ArgSpec


class SharedMPCFlags(Flags):
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
                help='horizon of simulated MPC rollouts')]
        pretty_name = 'common model-predictive control settings'
        super().__init__('mpc', pretty_name, arguments)


class MPCFlags(Flags):
    """
    Base class for specifying flags for model-predictive planners.
    """

    def make_mpc(self, env, dyn_model, all_flags):
        """
        Generate an MPC instance according to the specification provided
        by the MPC flags instance.
        """
        raise NotImplementedError
