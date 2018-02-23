"""
This module defines the flags common to all experiments,
and the general experimental procedure.
"""

from contextlib import contextmanager, closing
import distutils.util
import json
import os
import shutil
import subprocess
import time

from gym import wrappers
import tensorflow as tf

from context import context
from flags import Flags, ArgSpec
import env_info
import log
import reporter
from utils import seed_everything


class ExperimentFlags(Flags):
    """Flags common to all experiments."""

    @staticmethod
    def _generate_arguments():
        yield ArgSpec(
            name='exp_name', type=str, default='unnamed_experiment',
            help='the name of the experiment')
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            # git_hash is a byte string; we want a string.
            git_hash = git_hash.decode('utf-8')
            # git_hash also comes with an extra \n at the end, which we remove.
            git_hash = git_hash.strip()
        except subprocess.CalledProcessError:
            git_hash = ""
        yield ArgSpec(
            name='git_hash', type=str, default=git_hash,
            help='the git hash of the code being used')
        yield ArgSpec(
            name='seed',
            type=int,
            default=[3],
            nargs='+',
            help='seeds for each trial')
        yield ArgSpec(
            name='verbose',
            type=distutils.util.strtobool,
            help='print debugging statements')
        yield ArgSpec(
            name='env_name',
            type=str,
            default='hc',
            help='environment to use',)
        yield ArgSpec(
            name='horizon',
            type=int,
            default=1000,
            help='real rollout maximum horizon',)
        yield ArgSpec(
            name='bufsize',
            type=int, default=int(1e6),
            help='transition replay buffer maximum size',)
        yield ArgSpec(
            name='render_every',
            type=int,
            default=0,
            help='if possible, render an episode every render_every timesteps.'
            ' If set to 0 then no rendering.')
        yield ArgSpec(
            name='evaluate_every',
            type=int,
            default=0,
            help='evaluate diagnostics episode every evaluate_every timesteps.'
            ' If set to 0 then no evaluating.')
        yield ArgSpec(
            name='save_every',
            type=int,
            default=0,
            help='save all persistent TF variables every save_every timesteps.'
            ' Do not save if set to 0')
        yield ArgSpec(
            name='discount',
            type=float,
            default=0.99,
            help='discount factor for the reward calculations')
        yield ArgSpec(
            name='reward_scaling',
            default=1,
            type=float,
            help='Amount to scale all rewards by. Pass -1 to use hard-coded '
            'reward scaling, based on environment.'
        )
        yield ArgSpec(
            name='timesteps',
            default=1000000,
            type=int,
            help='approximate number of timesteps to collect')
        yield ArgSpec(
            name='env_parallelism',
            default=8,
            type=int,
            help='maximum parallelism to be used in simultaneous environment '
            'evaluation: this is already minimized with the cpu count')
        yield ArgSpec(
            name='tf_parallelism',
            default=None,
            type=int,
            help='maximum number of CPUs to be used by TF exec thread pools')

    def __init__(self):
        super().__init__('experiment', 'experiment governance',
                         list(ExperimentFlags._generate_arguments()))
        self._num_rendered = 0
        self._num_saved = 0
        self._num_evaluated = 0
        self._last_rendered = -1
        self._last_saved = -1
        self._last_evaluated = -1

    @contextmanager
    def render_env(self):
        """Create a renderable env (context). Only call at most once per ts"""
        uid = reporter.timestep()
        with closing(env_info.make_env()) as env:
            limited_env = wrappers.TimeLimit(
                env, max_episode_steps=self.horizon)
            render_directory = reporter.logging_directory()
            directory = os.path.join(render_directory, str(uid))
            with closing(wrappers.Monitor(limited_env, directory)) as rendered:
                yield rendered

    def log_directory(self):
        """return a root name for the log directory of this experiment"""
        exp_name = self.exp_name
        env_name = self.env_name
        return "{}_{}".format(exp_name, env_name)

    def should_continue(self):
        """
        Returns true iff we have not yet collected all the allotted timesteps.
        Should be called at most once per timestep.
        """
        return reporter.timestep() < self.timesteps

    def should_render(self):
        """
        Should we render in the current timestep?
        """
        if self.render_every == 0:
            return False
        if reporter.timestep() == self._last_rendered:
            return True
        next_render = self._num_rendered * self.render_every
        if reporter.timestep() >= next_render:
            while (self._num_rendered * self.render_every <=
                   reporter.timestep()):
                self._num_rendered += 1

            self._last_rendered = reporter.timestep()
            return True
        return False

    def should_save(self):
        """
        Should we save in the current timestep?
        """
        if self.save_every == 0:
            return False
        if reporter.timestep() == self._last_saved:
            return True
        next_save = self._num_saved * self.save_every
        if reporter.timestep() >= next_save:
            while (self._num_saved * self.save_every <=
                   reporter.timestep()):
                self._num_saved += 1
            self._last_saved = reporter.timestep()
            return True
        return False

    def should_evaluate(self):
        """
        Should we evaluate in the current timestep?
        """
        if self.evaluate_every == 0:
            return False
        if reporter.timestep() == self._last_evaluated:
            return True
        next_evaluate = self._num_evaluated * self.evaluate_every
        if reporter.timestep() >= next_evaluate:
            while (self._num_evaluated * self.evaluate_every <=
                   reporter.timestep()):
                self._num_evaluated += 1
            self._last_evaluated = reporter.timestep()
            return True
        return False


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


def experiment_main(flags, experiment_fn):
    """
    Given parsed arguments, this method does the high-level governance
    required for running an iterated experiment (including graph
    construction and session management)

    In particular, for every seed s0, ..., sn, this creates
    a directory in data/${exp_name}_${env_name}/si for each i.

    With logging, summary reporting, and parameters appropriately
    configured, for every sub-directory and its corresponding
    seed, a new experiment is invoked on a fresh, seeded environment.

    experiment_fn(flags) is called, with the seed modified in args
    to indicate which seed was used.
    """
    log.init(flags.experiment.verbose)
    logdir_name = flags.experiment.log_directory()
    logdir = _make_data_directory(logdir_name)

    all_seeds = flags.experiment.seed
    for seed in all_seeds:
        # Save params to disk.
        logdir_seed = os.path.join(logdir, str(seed))
        os.makedirs(logdir_seed)
        flags.experiment.seed = seed
        with open(os.path.join(logdir_seed, 'params.json'), 'w') as f:
            json.dump(flags.asdict(), f, sort_keys=True, indent=4)

        g = tf.Graph()
        with g.as_default():
            seed_everything(seed)
            context().flags = flags
            with reporter.create(logdir_seed, flags.experiment.verbose), \
                    env_info.create():
                experiment_fn(flags)
