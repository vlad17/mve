"""Measure env speed"""

from contextlib import closing

from context import flags
import env_info
from experiment import experiment_main, ExperimentFlags
from flags import Flags, ArgSpec, parse_args
from sample import sample_venv
from utils import timeit, as_controller
from zero_learner import ZeroLearner
from venv.serial_venv import SerialVenv
from venv.parallel_venv import ParallelVenv


def train(_):
    """
    Train constrained MPC with the specified flags and subflags.
    """
    nenvs = flags().run.nenvs
    repeats = flags().run.repeats
    controller = as_controller(ZeroLearner().act)
    with closing(SerialVenv(nenvs, env_info.make_env)) as venv:
        with timeit('singleproc venv'):
            for _ in range(repeats):
                sample_venv(venv, controller)
    with closing(ParallelVenv(nenvs)) as venv:
        with timeit('multithread venv'):
            for _ in range(repeats):
                sample_venv(venv, controller)


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
                help='repetitions to run')]
        super().__init__('run', 'run flags for testenv', arguments)


if __name__ == "__main__":
    _flags = [ExperimentFlags(), _RunFlags()]
    _args = parse_args(_flags)
    experiment_main(_args, train)
