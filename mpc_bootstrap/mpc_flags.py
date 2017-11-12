"""MPC flags."""

from flags import Flags
from mpc import MPC

class MpcFlags(Flags):  # pylint: disable=too-many-instance-attributes
    """MPC flags."""

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Adds flags to an argparse parser."""
        if argument_group is None:
            argument_group = parser.add_argument_group('mpc')
        argument_group.add_argument(
            '--random_paths',
            type=int,
            default=10,
            help='number of purely random paths (to warm up dynamics)',
        )
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
        self.random_paths = args.random_paths
        self.onpol_iters = args.onpol_iters
        self.onpol_paths = args.onpol_paths
        self.horizon = args.horizon
        self.mpc_simulated_paths = args.mpc_simulated_paths
        self.mpc_horizon = args.mpc_horizon

    def make_controller(self, env, venv, sess, dyn_model):
        """Make an MPC controller."""
        return MPC(env=venv,
                   dyn_model=dyn_model,
                   horizon=self.mpc_horizon,
                   reward_fn=env.tf_reward,
                   num_simulated_paths=self.mpc_simulated_paths,
                   sess=sess,
                   learner=None)
