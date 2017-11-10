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
import subprocess

_SUBFLAGS = [
    'experiment', 'algorithm', 'dynamics', 'mpc', 'controller', 'learner']

AllFlags = collections.namedtuple('AllFlags', _SUBFLAGS)


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

    def name(self):
        """Flag group name"""
        return self.__class__.__name__.replace('Flags', '').lower()


class ExperimentFlags(Flags):
    """Generic experiment flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            # git_hash is a byte string; we want a string.
            git_hash = git_hash.decode('utf-8')
            # git_hash also comes with an extra \n at the end, which we remove.
            git_hash = git_hash.strip()
        except subprocess.CalledProcessError:
            git_hash = ""

        experiment = parser.add_argument_group('experiment')
        experiment.add_argument(
            '--exp_name',
            type=str,
            default='unnamed_experiment',
            help='the name of the experiment',
        )
        experiment.add_argument(
            '--git_hash',
            type=str,
            default=git_hash,
            help='the git hash of the code being used',
        )
        experiment.add_argument(
            '--seed',
            type=int,
            default=[3],
            nargs='+',
            help='seeds for each trial'
        )
        experiment.add_argument(
            '--verbose',
            action='store_true',
            help='print debugging statements'
        )

    def __init__(self, args):
        self.exp_name = args.exp_name
        self.git_hash = args.git_hash
        self.seed = args.seed
        self.verbose = args.verbose


class AlgorithmFlags(Flags):
    """Generic algorithm flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        algorithm = parser.add_argument_group('algorithm')
        algorithm.add_argument(
            '--env_name',
            type=str,
            default='hc-hard',
            help='environment to use',
        )
        algorithm.add_argument(
            '--onpol_iters',
            type=int,
            default=1,
            help='number of outermost on policy aggregation iterations',
        )
        algorithm.add_argument(
            '--onpol_paths',
            type=int,
            default=10,
            help='number of rollouts per on policy iteration',
        )
        algorithm.add_argument(
            '--random_paths',
            type=int,
            default=10,
            help='number of purely random paths (to warm up dynamics)',
        )
        algorithm.add_argument(
            '--horizon',
            type=int,
            default=1000,
            help='real rollout horizon',
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
        self.frame_skip = args.frame_skip


class DynamicsFlags(Flags):
    """
    We use a neural network to model an environment's dynamics. These flags
    define the architecture and learning policy of neural network.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        dynamics_nn = parser.add_argument_group('dynamics')
        dynamics_nn.add_argument(
            '--dyn_depth',
            type=int,
            default=2,
            help='dynamics NN depth',
        )
        dynamics_nn.add_argument(
            '--dyn_width',
            type=int,
            default=500,
            help='dynamics NN width',
        )
        dynamics_nn.add_argument(
            '--dyn_learning_rate',
            type=float,
            default=1e-3,
            help='dynamics NN learning rate',
        )
        dynamics_nn.add_argument(
            '--dyn_epochs',
            type=int,
            default=60,
            help='dynamics NN epochs',
        )
        dynamics_nn.add_argument(
            '--dyn_batch_size',
            type=int,
            default=512,
            help='dynamics NN batch size',
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


class ControllerFlags(Flags):
    """
    Bootstrapped MPC uses a neural network to model an MPC controller. These
    arguments define the architecture and learning policy of that neural
    network.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        learner_nn = parser.add_argument_group('learned controller')
        learner_nn.add_argument(
            '--con_depth',
            type=int,
            default=5,
            help='learned controller NN depth',
        )
        learner_nn.add_argument(
            '--con_width',
            type=int,
            default=32,
            help='learned controller NN width',
        )
        learner_nn.add_argument(
            '--con_learning_rate',
            type=float,
            default=1e-3,
            help='learned controller NN learning rate',
        )
        learner_nn.add_argument(
            '--con_epochs',
            type=int,
            default=100,
            help='learned controller epochs',
        )
        learner_nn.add_argument(
            '--con_batch_size',
            type=int,
            default=512,
            help='learned controller batch size',
        )
        return learner_nn

    def __init__(self, args):
        self.con_depth = args.con_depth
        self.con_width = args.con_width
        self.con_learning_rate = args.con_learning_rate
        self.con_epochs = args.con_epochs
        self.con_batch_size = args.con_batch_size

    def name(self):
        return 'controller'


class DeltaLearner(ControllerFlags):
    """
    A learner with a dirac delta (deterministic) policy.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        learner_nn = ControllerFlags.add_flags(parser)
        learner_nn.add_argument(
            '--explore_std',
            type=float,
            default=0.0,
            help='if exactly 0, explore with a uniform policy on the first '
            'simulated step; else use a Gaussian with the specified std',
        )

    def __init__(self, args):
        super().__init__(args)
        self.explore_std = args.explore_std


class GaussianLearner(ControllerFlags):
    """
    A learner with a Gaussian policy, with optional exploration.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        learner_nn = ControllerFlags.add_flags(parser)
        learner_nn.add_argument(
            '--no_extra_explore',
            action='store_true',
            help='don\'t add extra noise to the first action proposed'
            ' by stochastic learners',
        )

    def __init__(self, args):
        super().__init__(args)
        self.no_extra_explore = args.no_extra_explore


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
            help='number of simulated MPC rollouts'
        )
        mpc.add_argument(
            '--mpc_horizon',
            type=int,
            default=15,
            help='horizon of simulated MPC rollouts',
        )
        mpc.add_argument(
            '--delay',
            type=int,
            default=0,
            help='delay for dagger',
        )

    def __init__(self, args):
        self.mpc_simulated_paths = args.mpc_simulated_paths
        self.mpc_horizon = args.mpc_horizon
        self.delay = args.delay


_ALL_SUBPARSERS = {
    'random': [ExperimentFlags, AlgorithmFlags, DynamicsFlags],
    'mpc': [ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags],
    'gaussian_bootstrap': [
        ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags,
        GaussianLearner],
    'zero_bootstrap': [
        ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags],
    'delta_bootstrap': [
        ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags,
        DeltaLearner],
    'gaussian_dagger': [
        ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags,
        GaussianLearner],
    'delta_dagger': [
        ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags,
        DeltaLearner],
    'gaussian_learneronly': [
        ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags,
        GaussianLearner],
    'delta_learneronly': [
        ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags,
        DeltaLearner],
    'zero_learneronly': [
        ExperimentFlags, AlgorithmFlags, MpcFlags, DynamicsFlags],
}


def _get_parser():
    """Gets a parser for all flags."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='agent')

    for name, subparser_flags in _ALL_SUBPARSERS.items():
        subparser = subparsers.add_parser(name)
        for flag in subparser_flags:
            flag.add_flags(subparser)

    return parser


def _parse_args(args):
    """Reads args from argparse flags into classes"""

    allflags_args = {subflag: None for subflag in _SUBFLAGS}
    for flag in _ALL_SUBPARSERS[args.agent]:
        flag = flag(args)
        allflags_args[flag.name()] = flag
    allflags_args = [allflags_args[subflag] for subflag in _SUBFLAGS]

    return AllFlags(*allflags_args)


def get_all_flags():
    """Get all flags."""
    parser = _get_parser()
    args = parser.parse_args()
    parsed_args = _parse_args(args)

    return parsed_args
