"""Measure env speed"""

from contextlib import closing
from functools import partial

import tensorflow as tf

from context import flags
from experiment import experiment_main, ExperimentFlags
from ddpg_learner import DDPGLearner, DDPGFlags
from ddpg.ddpg import generate_actor as generate_ddpg_actor
from flags import Flags, ArgSpec, parse_args
from sample import sample_venv
from utils import timeit, as_controller
from zero_learner import ZeroLearner
from venv.actor_venv import ActorVenv
from venv.serial_venv import SerialVenv
from venv.parallel_venv import ParallelVenv


def _generate_zero_actor():
    return ZeroLearner().act


def _generate_actor_venv(gen_actor, m):
    return ActorVenv(gen_actor, m)


def train(_):
    """
    Train constrained MPC with the specified flags and subflags.
    """
    if flags().run.controller == 'zero':
        act = ZeroLearner().act
        generate_actor = _generate_zero_actor
    elif flags().run.controller == 'ddpg':
        learner = DDPGLearner()
        act = learner.act
        generate_actor = generate_ddpg_actor
        tf.get_default_session().run(tf.global_variables_initializer())
    else:
        raise ValueError('controller {} unrecognized'.format(
            flags().run.controller))
    generate_actor_venv = partial(_generate_actor_venv, generate_actor)

    nenvs = flags().run.nenvs
    repeats = flags().run.repeats
    controller = as_controller(act)
    with closing(SerialVenv(nenvs)) as venv:
        with timeit('singleproc venv'):
            for _ in range(repeats):
                _ = sample_venv(venv, controller)
    with closing(ActorVenv(generate_actor, nenvs)) as venv:
        tf.get_default_session().run(tf.global_variables_initializer())
        obs = venv.reset()
        with timeit('pushdown venv'):
            for _ in range(repeats):
                venv.multi_step_actor(obs, flags().experiment.horizon)
    with closing(ParallelVenv(nenvs)) as venv:
        with timeit('multiproc venv'):
            for _ in range(repeats):
                _ = sample_venv(venv, controller)
    with closing(ParallelVenv(nenvs, generate_actor_venv)) as venv:
        obs = venv.reset()
        with timeit('multiproc+pushdown venv'):
            for _ in range(repeats):
                venv.multi_step_actor(obs, flags().experiment.horizon)


class _RunFlags(Flags):
    def __init__(self):
        arguments = [
            ArgSpec(
                name='nenvs',
                type=int,
                default=32,
                help='parallel envs to run'),
            ArgSpec(
                name='controller',
                type=str,
                default='zero',
                help='controller to use, one of ddpg or zero'),
            ArgSpec(
                name='repeats',
                type=int,
                default=3,
                help='repetitions to run')]
        super().__init__('run', 'run flags for testenv', arguments)


if __name__ == "__main__":
    _flags = [ExperimentFlags(), _RunFlags(), DDPGFlags()]
    _args = parse_args(_flags)
    experiment_main(_args, train)
