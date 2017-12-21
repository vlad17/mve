"""Generate DDPG rollouts and train on them"""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from dataset import Dataset
from ddpg_learner import DDPGLearner
from experiment import ExperimentFlags, experiment_main
from flags import (parse_args, Flags, ArgSpec)
from learner import as_controller
from multiprocessing_env import make_venv
import reporter
from sample import sample_venv
from utils import timeit


def _train(args):
    env = args.experiment.make_env()
    data = Dataset.from_env(env, args.experiment.horizon,
                            args.experiment.bufsize)
    venv = make_venv(args.experiment.make_env, 1)
    learner = args.run.make_ddpg(env, args.experiment.discount)

    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for _ in range(args.run.episodes):
        with timeit('learner fit'):
            if data.size:
                learner.fit(data)

        with timeit('sample learner'):
            controller = as_controller(learner)
            paths = sample_venv(venv, controller, args.experiment.horizon)
            data.add_paths(paths)

        with timeit('gathering statistics'):
            rewards = [path.rewards.sum() for path in paths]
            reporter.add_summary_statistics('reward', rewards)

        reporter.advance_iteration()


class RunFlags(Flags):
    """Flags relevant for running DDPG over multiple iterations."""

    def __init__(self):
        arguments = [
            ArgSpec(
                name='episodes',
                type=int,
                default=300,
                help='number episodes to train on')]
        arguments += DDPGLearner.FLAGS
        super().__init__('run', 'run flags for ddpg', arguments)
        self.discount = None

    def make_ddpg(self, env, discount):
        """Create a DDPGLearner with the specifications from this invocation"""
        self.discount = discount
        return DDPGLearner(env, self)


if __name__ == "__main__":
    flags = [ExperimentFlags(), RunFlags()]
    _args = parse_args(flags)
    experiment_main(_args, _train)
