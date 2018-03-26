"""
This module defines the flags common to all experiments,
and the general experimental procedure.
"""

from contextlib import contextmanager, closing, ExitStack
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
        yield ArgSpec(
            name='seed',
            type=int,
            default=3)
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
            name='discount',
            type=float,
            default=0.99,
            help='discount factor for the reward calculations')
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


def _make_data_directory(name):
    """
    _make_data_directory(name) will create a directory data/name and return the
    directory's name. If a data/name directory already exists, then it will be
    renamed data/name-i where i is the smallest integer such that data/name-i
    does not already exist. For example, imagine the data/ directory has the
    following contents:

        data/foo
        data/foo-old-1
        data/foo-old-2
        data/foo-old-3

    Then, make_data_directory("foo") will rename data/foo to data/foo-4 and
    then create a fresh data/foo directory.
    """
    # Make the data directory if it does not already exist.
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)

    name = os.path.join('data', name)
    ctr = 0
    logdir = name
    while os.path.exists(logdir):
        logdir = name + '-old-{}'.format(ctr)
        ctr += 1
    if ctr > 0:
        log.debug('Experiment already exists, moved old one to {}.', logdir)
        shutil.move(name, logdir)

    os.makedirs(name)
    with open(os.path.join(name, 'starttime.txt'), 'w') as f:
        print(time.strftime("%d-%m-%Y_%H-%M-%S"), file=f)

    return name


@contextmanager
def setup_experiment_context(
        flags, create_logdir=True, create_reporter=True, create_env_info=True):
    """
    Given parsed arguments, this method does the high-level governance
    required for running an iterated experiment (including graph
    construction and session management).

    In particular, if the seed is s, then this method creates
    a directory in data/${exp_name}_${env_name}/s.

    With logging, summary reporting, and parameters appropriately
    configured, for every sub-directory and its corresponding
    seed, a new experiment may be invoked on a fresh, seeded environment,
    within this context.

    Note that reporter can only be true if logdir is true.
    """
    assert not create_reporter or create_logdir, \
        'reporter can only be created if logdir is'

    seed = flags.experiment.seed
    if create_logdir:
        log.init(flags.experiment.verbose)
        logdir = flags.experiment.log_directory()
        logdir = os.path.join(logdir, str(seed))
        logdir = _make_data_directory(logdir)

        # Save params to disk.
        with open(os.path.join(logdir, 'params.json'), 'w') as f:
            params = flags.asdict()
            params['git_hash'] = _git_hash()
            json.dump(params, f, sort_keys=True, indent=4)

    with ExitStack() as stack:
        g = tf.Graph()
        stack.enter_context(g.as_default())
        seed_everything(seed)
        context().flags = flags
        if create_reporter:
            stack.enter_context(
                reporter.create(logdir, flags.experiment.verbose))
        if create_env_info:
            stack.enter_context(env_info.create())
        yield


def _git_hash():
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(module_dir)
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=package_dir)
        # git_hash is a byte string; we want a string.
        git_hash = git_hash.decode('utf-8')
        # git_hash also comes with an extra \n at the end, which we remove.
        git_hash = git_hash.strip()
    except subprocess.CalledProcessError:
        git_hash = '<no git hash available>'
    return git_hash
