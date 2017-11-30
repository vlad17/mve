"""Flags common to all experiments."""

import subprocess

from flags import Flags
from envs import (WhiteBoxHalfCheetahEasy, WhiteBoxHalfCheetahHard,
                  WhiteBoxAntEnv, WhiteBoxWalker2dEnv)


class ExperimentFlags(Flags):  # pylint: disable=too-many-instance-attributes
    """Flags common to all experiments."""

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
        experiment.add_argument(
            '--env_name',
            type=str,
            default='hc-hard',
            help='environment to use',
        )
        experiment.add_argument(
            '--frame_skip',
            type=int,
            default=1,
        )
        experiment.add_argument(
            '--horizon',
            type=int,
            default=1000,
            help='real rollout maximum horizon',
        )
        experiment.add_argument(
            '--bufsize',
            type=int,
            default=int(1e6),
            help='transition replay buffer maximum size',
        )

    @staticmethod
    def name():
        return "experiment"

    def __init__(self, args):
        self.exp_name = args.exp_name
        self.git_hash = args.git_hash
        self.seed = args.seed
        self.verbose = args.verbose
        self.env_name = args.env_name
        self.frame_skip = args.frame_skip
        self.horizon = args.horizon
        self.bufsize = args.bufsize

    # TODO(mwhittaker): Consistently choose "mk" or "make".
    def mk_env(self):
        """Generates an unvectorized env."""
        if self.env_name == 'hc-hard':
            return WhiteBoxHalfCheetahHard(self.frame_skip)
        elif self.env_name == 'hc-easy':
            return WhiteBoxHalfCheetahEasy(self.frame_skip)
        elif self.env_name == 'ant':
            return WhiteBoxAntEnv(self.frame_skip)
        elif self.env_name == 'walker2d':
            return WhiteBoxWalker2dEnv(self.frame_skip)
        else:
            raise ValueError('env {} unsupported'.format(self.env_name))

    def log_directory(self):
        """return a root name for the log directory of this experiment"""
        exp_name = self.exp_name
        env_name = self.env_name
        return "{}_{}".format(exp_name, env_name)
