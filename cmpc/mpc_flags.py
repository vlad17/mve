"""MPC flags specify details for the planner"""

from flags import Flags

from random_shooter import RandomShooter


class MpcFlags(Flags):
    """
    Top-level MPC flags, used regardless of the planner for managing
    options common to all MPC experiments.
    """

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Adds flags to an argparse parser."""
        if argument_group is None:
            argument_group = parser.add_argument_group('mpc')
        argument_group.add_argument(
            '--onpol_iters',
            type=int,
            default=1,
            help='number of outermost on policy aggregation iterations',
        )
        argument_group.add_argument(
            '--onpol_paths',
            type=int,
            default=10,
            help='number of rollouts per on policy iteration',
        )
        argument_group.add_argument(
            '--mpc_horizon',
            type=int,
            default=15,
            help='horizon of simulated MPC rollouts',
        )
        argument_group.add_argument(
            '--planner',
            type=str,
            default='shooter',
            help='possibly-constrained planning optimization algorithm')

    @staticmethod
    def name():
        return "mpc"

    def __init__(self, args):
        self.onpol_iters = args.onpol_iters
        self.onpol_paths = args.onpol_paths
        self.horizon = args.mpc_horizon
        self.planner = args.planner

    def make_mpc(self, env, dyn_model, reward_fn, learner, all_flags):
        """
        Generate an MPC instance according to the specification provided
        by the MPC flags
        """

        if self.planner == 'shooter':
            return RandomShooter(
                env, dyn_model, reward_fn, learner, self.horizon,
                all_flags.shooter)
        else:
            raise ValueError('planner {} not known'.format(self.planner))


class RandomShooterFlags(Flags):
    """Flags specific to random-shooter based CMPC"""

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Adds flags to an argparse parser."""
        if argument_group is None:
            argument_group = parser.add_argument_group(
                'random shooter', 'options for --planner shooter')
        argument_group.add_argument(
            '--opt_horizon',
            type=int,
            default=None,
            help='Requires using MPC with a learner, can be at most '
            'mpc_horizon. Plan for next mpc_horizon steps but only '
            'optimize over first opt_horizon actions')
        argument_group.add_argument(
            '--simulated_paths',
            type=int,
            default=1000,
            help='number of simulated MPC rollouts'
        )

    @staticmethod
    def name():
        return "shooter"

    def __init__(self, args):
        self.opt_horizon = args.opt_horizon
        self.simulated_paths = args.simulated_paths
