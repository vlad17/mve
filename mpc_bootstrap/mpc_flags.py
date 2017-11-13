"""MPC flags."""

from flags import Flags


class MpcFlags(Flags):  # pylint: disable=too-many-instance-attributes
    """MPC flags."""

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
            '--horizon',
            type=int,
            default=1000,
            help='real rollout horizon',
        )
        argument_group.add_argument(
            '--mpc_simulated_paths',
            type=int,
            default=1000,
            help='number of simulated MPC rollouts'
        )
        argument_group.add_argument(
            '--mpc_horizon',
            type=int,
            default=15,
            help='horizon of simulated MPC rollouts',
        )

    @staticmethod
    def name():
        return "mpc"

    def __init__(self, args):
        self.onpol_iters = args.onpol_iters
        self.onpol_paths = args.onpol_paths
        self.horizon = args.horizon
        self.mpc_simulated_paths = args.mpc_simulated_paths
        self.mpc_horizon = args.mpc_horizon
