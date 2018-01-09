"""Measure env speed"""

from contextlib import closing

import tensorflow as tf

from context import flags
from experiment import experiment_main, ExperimentFlags
from ddpg_learner import DDPGLearner, DDPGFlags
from flags import Flags, ArgSpec, parse_args
from sample import sample_venv
import server_registry
from utils import timeit, as_controller
from venv.ddpg_actor_venv import DDPGActorVenv
from venv.serial_venv import SerialVenv
from venv.threaded_venv import ThreadedVenv
from venv.parallel_venv import ParallelVenv, ParallelVenvFlags


def train(_):
    """
    Train constrained MPC with the specified flags and subflags.
    """
    nenvs = flags().run.nenvs
    act = DDPGLearner().act
    with closing(ParallelVenv(nenvs, DDPGActorVenv, need_tf=True)) as tfvenv:
        init = tf.global_variables_initializer()
        with server_registry.make_default_session():
            init.run()
            _train(act, tfvenv)


def _train(act, remote_actor_venv):
    nenvs = flags().run.nenvs
    repeats = flags().run.repeats
    controller = as_controller(act)
    if flags().run.run_serial:
        with closing(SerialVenv(nenvs)) as venv:
            with timeit('singleproc venv'):
                for _ in range(repeats):
                    _ = sample_venv(venv, controller)
    with closing(ThreadedVenv(nenvs)) as venv:
        with timeit('multithreaded venv'):
            for _ in range(repeats):
                _ = sample_venv(venv, controller)
    with closing(ParallelVenv(nenvs)) as venv:
        with timeit('multiproc venv'):
            for _ in range(repeats):
                _ = sample_venv(venv, controller)
    obs = remote_actor_venv.reset()
    with timeit('multiproc+pushdown venv'):
        for _ in range(repeats):
            remote_actor_venv.multi_step_actor(obs, flags().experiment.horizon)


class _RunFlags(Flags):
    def __init__(self):
        arguments = [
            ArgSpec(
                name='nenvs',
                type=int,
                default=32,
                help='parallel envs to run'),
            ArgSpec(
                name='repeats',
                type=int,
                default=3,
                help='repetitions to run'),
            ArgSpec(
                name='run_serial',
                default=False,
                action='store_true',
                help='run serial venv for comparison')]
        super().__init__('run', 'run flags for testenv', arguments)


if __name__ == "__main__":
    _flags = [ExperimentFlags(), _RunFlags(), DDPGFlags(), ParallelVenvFlags()]
    _args = parse_args(_flags)
    experiment_main(_args, train)
