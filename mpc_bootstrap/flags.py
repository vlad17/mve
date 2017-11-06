"""
Ever wonder what it looks like when a grenade goes off in a hyperparameter
factory? Tranining a (potentially bootstrapped) on-policy MPC controller
involves a _lot_ of hyperparameters that we'd like to toggle from the command
line. This file helps do that without going crazy:

    import flags
    all_flags = flags.get_all_flags()
"""

import argparse
import collections


AllFlags = collections.namedtuple(
    "AllFlags", ["exp", "alg", "dyn", "mpc", "con"])


def flags_to_json(all_flags):
    """Returns a JSON representation of all the flags in all_flags."""
    params = {}
    for flag in all_flags:
        if flag is not None:
            params.update(vars(flag))
    return params


class Flags(object):
    """A group of logically related flags."""

    def __str__(self):
        xs = vars(self)
        return "\n".join(["--{} {}".format(x, v) for (x, v) in xs.items()])

    def __repr__(self):
        return str(self)


class ExperimentFlags(Flags):
    """Generic experiment flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        experiment = parser.add_argument_group('Experiment')
        experiment.add_argument(
            '--exp_name',
            type=str,
            default='mb_mpc',
            help='The name of the experiment',
        )
        experiment.add_argument(
            '--time',
            action='store_true',
            default=False,
            help="Print profiling information",
        )
        experiment.add_argument(
            '--seed',
            type=int,
            default=[3],
            nargs='+',
            help='Seeds for random number generators'
        )
        experiment.add_argument(
            '--verbose',
            action='store_true',
            help='Print debugging statements'
        )

    def __init__(self, args):
        self.exp_name = args.exp_name
        self.time = args.time
        self.seed = args.seed
        self.verbose = args.verbose


# pylint: disable=too-many-instance-attributes
class AlgorithmFlags(Flags):
    """Generic algorithm flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        algorithm = parser.add_argument_group('Algorithm')
        algorithm.add_argument(
            '--env_name',
            type=str,
            default='hc-hard',
            help="Environment to use",
        )
        algorithm.add_argument(
            '--onpol_iters',
            type=int,
            default=1,
            help='Number of outermost on policy iterations',
        )
        algorithm.add_argument(
            '--onpol_paths',
            type=int,
            default=10,
            help='Number of rollouts per on policy iteration',
        )
        algorithm.add_argument(
            '--random_paths',
            type=int,
            default=10,
            help='Number of purely random paths',
        )
        algorithm.add_argument(
            '--horizon',
            type=int,
            default=1000,
            help='Rollout horizon',
        )
        algorithm.add_argument(
            '--hard_cost',
            action='store_true',
            default=False,
            help='Use harder, less cherry-picked cost function',
        )
        algorithm.add_argument(
            '--frame_skip',
            type=int,
            default=1,
        )

    def __init__(self, args):
        self.agent = args.agent
        self.env_name = args.env_name
        self.onpol_iters = args.onpol_iters
        self.onpol_paths = args.onpol_paths
        self.random_paths = args.random_paths
        self.horizon = args.horizon
        self.hard_cost = args.hard_cost
        self.frame_skip = args.frame_skip


class DynamicsFlags(Flags):
    """
    We use a neural network to model an environment's dynamics. These flags
    define the architecture and learning policy of neural network.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        dynamics_nn = parser.add_argument_group('Dynamics NN')
        dynamics_nn.add_argument(
            '--dyn_depth',
            type=int,
            default=2,
            help='Dynamics NN depth',
        )
        dynamics_nn.add_argument(
            '--dyn_width',
            type=int,
            default=500,
            help='Dynamics NN SGD width',
        )
        dynamics_nn.add_argument(
            '--dyn_learning_rate',
            type=float,
            default=1e-3,
            help='Dynamics NN learning rate',
        )
        dynamics_nn.add_argument(
            '--dyn_epochs',
            type=int,
            default=60,
            help='Dynamics NN SGD epochs',
        )
        dynamics_nn.add_argument(
            '--dyn_batch_size',
            type=int,
            default=512,
            help='Dynamics NN SGD batch size',
        )
        dynamics_nn.add_argument(
            '--no_delta_norm',
            action='store_true',
            default=False
        )

    def __init__(self, args):
        self.dyn_depth = args.dyn_depth
        self.dyn_width = args.dyn_width
        self.dyn_learning_rate = args.dyn_learning_rate
        self.dyn_epochs = args.dyn_epochs
        self.dyn_batch_size = args.dyn_batch_size
        self.no_delta_norm = args.no_delta_norm


# pylint: disable=too-many-instance-attributes
class LearnedControllerFlags(Flags):
    """
    Bootstrapped MPC uses a neural network to model an MPC controller. These
    arguments define the architecture and learning policy of that neural
    network.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        learner_nn = parser.add_argument_group('Learned Controller NN')
        learner_nn.add_argument(
            '--con_depth',
            type=int,
            default=1,
            help='Learned controller NN depth',
        )
        learner_nn.add_argument(
            '--con_width',
            type=int,
            default=32,
            help='Learned controller NN width',
        )
        learner_nn.add_argument(
            '--con_learning_rate',
            type=float,
            default=1e-4,
            help='Learned controller NN learning rate',
        )
        learner_nn.add_argument(
            '--con_epochs',
            type=int,
            default=60,
            help='Learned controller SGD epochs',
        )
        learner_nn.add_argument(
            '--con_batch_size',
            type=int,
            default=512,
            help='Learned controller SGD batch size',
        )
        learner_nn.add_argument(
            '--explore_std',
            type=float,
            default=0.0,
            help='TODO(vladf): Document.',
        )
        learner_nn.add_argument(
            '--no_extra_explore',
            action='store_true',
            help='TODO(vlad17)',
        )
        learner_nn.add_argument(
            '--deterministic_learner',
            action='store_true',
            help='TODO(vlad17)',
        )

    def __init__(self, args):
        self.con_depth = args.con_depth
        self.con_width = args.con_width
        self.con_learning_rate = args.con_learning_rate
        self.con_epochs = args.con_epochs
        self.con_batch_size = args.con_batch_size
        self.explore_std = args.explore_std
        self.no_extra_explore = args.no_extra_explore
        self.deterministic_learner = args.deterministic_learner


class MpcFlags(Flags):
    """MPC flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        mpc = parser.add_argument_group('MPC')
        mpc.add_argument(
            '--mpc_simulated_paths',
            type=int,
            default=1000,
            help="Number of simulated MPC rollouts"
        )
        mpc.add_argument(
            '--mpc_horizon',
            type=int,
            default=15,
            help='Horizon of simulated MPC rollouts',
        )
        mpc.add_argument(
            '--delay',
            type=int,
            default=0,
            help='Delay for dagger',
        )

    def __init__(self, args):
        self.mpc_simulated_paths = args.mpc_simulated_paths
        self.mpc_horizon = args.mpc_horizon
        self.delay = args.delay


def _get_parser():
    """Gets a parser for all flags."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='agent')

    mpc_parser = subparsers.add_parser('mpc')
    ExperimentFlags.add_flags(mpc_parser)
    AlgorithmFlags.add_flags(mpc_parser)
    DynamicsFlags.add_flags(mpc_parser)
    MpcFlags.add_flags(mpc_parser)

    random_parser = subparsers.add_parser('random')
    ExperimentFlags.add_flags(random_parser)
    AlgorithmFlags.add_flags(random_parser)
    DynamicsFlags.add_flags(random_parser)

    bootstrap_parser = subparsers.add_parser('bootstrap')
    ExperimentFlags.add_flags(bootstrap_parser)
    AlgorithmFlags.add_flags(bootstrap_parser)
    DynamicsFlags.add_flags(bootstrap_parser)
    MpcFlags.add_flags(bootstrap_parser)
    LearnedControllerFlags.add_flags(bootstrap_parser)

    dagger_parser = subparsers.add_parser('dagger')
    ExperimentFlags.add_flags(dagger_parser)
    AlgorithmFlags.add_flags(dagger_parser)
    DynamicsFlags.add_flags(dagger_parser)
    MpcFlags.add_flags(dagger_parser)
    LearnedControllerFlags.add_flags(dagger_parser)

    return parser


def get_all_flags():
    """Get all flags."""
    parser = _get_parser()
    args = parser.parse_args()
    exp_flags = ExperimentFlags(args)
    alg_flags = AlgorithmFlags(args)
    dyn_flags = DynamicsFlags(args)
    mpc_flags = None
    con_flags = None

    if args.agent in ['mpc', 'bootstrap', 'dagger']:
        mpc_flags = MpcFlags(args)
    if args.agent in ['bootstrap', 'dagger']:
        con_flags = LearnedControllerFlags(args)

    return AllFlags(exp_flags, alg_flags, dyn_flags, mpc_flags, con_flags)
