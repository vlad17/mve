"""
This module defines the flags common to all experiments,
and the general experimental procedure.
"""

import json
import os
import shutil
import subprocess
import time

from gym import wrappers
import numpy as np
import tensorflow as tf

from flags import Flags, convert_flags_to_json
from envs import (WhiteBoxHalfCheetahEasy, WhiteBoxHalfCheetahHard,
                  WhiteBoxAntEnv, WhiteBoxWalker2dEnv)
import log
import reporter
from utils import seed_everything


class ExperimentFlags(Flags):
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
        experiment.add_argument(
            '--render_every',
            type=int,
            default=0,
            help='if possible, render an episode every render_every episodes. '
            'If set to 0 then no rendering.'
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
        self.render_every = args.render_every
        self.render_directory = None  # set by experiment_main

    def make_env(self):
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

    def render_env(self, env, uid):
        """Take a regular env and make it render-able."""
        limited_env = wrappers.TimeLimit(env, max_episode_steps=self.horizon)
        directory = os.path.join(self.render_directory, str(uid))
        render_env = wrappers.Monitor(limited_env, directory)
        return render_env

    def log_directory(self):
        """return a root name for the log directory of this experiment"""
        exp_name = self.exp_name
        env_name = self.env_name
        return "{}_{}".format(exp_name, env_name)


def _make_data_directory(name):
    """
    _make_data_directory(name) will create a directory data/name and return the
    directory's name. If a data/name directory already exists, then it will be
    renamed data/name-i where i is the smallest integer such that data/name-i
    does not already exist. For example, imagine the data/ directory has the
    following contents:

        data/foo
        data/foo-1
        data/foo-2
        data/foo-3

    Then, make_data_directory("foo") will rename data/foo to data/foo-4 and
    then create a fresh data/foo directory.
    """
    # Make the data directory if it does not already exist.
    if not os.path.exists('data'):
        os.makedirs('data')

    name = os.path.join('data', name)
    ctr = 0
    logdir = name
    while os.path.exists(logdir):
        logdir = name + '-{}'.format(ctr)
        ctr += 1
    if ctr > 0:
        log.debug('Experiment already exists, moved old one to {}.', logdir)
        shutil.move(name, logdir)

    os.makedirs(name)
    with open(os.path.join(name, 'starttime.txt'), 'w') as f:
        print(time.strftime("%d-%m-%Y_%H-%M-%S"), file=f)

    return name


def experiment_main(flags, experiment_fn, subflags=None):
    """
    Given parsed arguments, this method does the high-level governance
    required for running an iterated experiment.

    In particular, for every seed s0, ..., sn, this creates
    a directory in data/${exp_name}_${env_name}/si for each i.

    With logging, summary reporting, and parameters appropriately
    configured, for every sub-directory and its corresponding
    seed, a new experiment is invoked on a fresh, seeded environment.

    experiment_fn(flags) is called, with the seed modified in args
    to indicate which seed was used. If subflags is not None
    then experiment_fn(flags, subflags) is called.
    """
    log.init(flags.experiment.verbose)
    logdir_name = flags.experiment.log_directory()
    logdir = _make_data_directory(logdir_name)
    np.seterr(all='raise')

    all_seeds = flags.experiment.seed
    for seed in all_seeds:
        # Save params to disk.
        logdir_seed = os.path.join(logdir, str(seed))
        os.makedirs(logdir_seed)
        flags.experiment.seed = seed
        with open(os.path.join(logdir_seed, 'params.json'), 'w') as f:
            flag_dict = convert_flags_to_json(flags)
            json.dump(flag_dict, f, sort_keys=True, indent=4)

        # Rendering directory
        rendir = os.path.join(logdir_seed, 'render')
        flags.experiment.render_directory = rendir

        # Run experiment.
        g = tf.Graph()
        with g.as_default():
            with reporter.create(logdir_seed, flags.experiment.verbose):
                seed_everything(seed)
                if subflags is not None:
                    experiment_fn(flags, subflags)
                else:
                    experiment_fn(flags)
