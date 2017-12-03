"""Generate DDPG rollouts and train on them"""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from dataset import Dataset, one_shot_dataset
from ddpg_learner_flags import DDPGLearnerFlags
from experiment import ExperimentFlags, experiment_main
from flags import (parse_args, Flags)
from multiprocessing_env import make_venv
import reporter
from sample import sample_venv
from utils import (timeit, create_tf_session)


def _train(args):
    env = args.experiment.make_env()
    data = Dataset.from_env(env, args.experiment.horizon,
                            args.experiment.bufsize)
    venv = make_venv(args.experiment.make_env, 1)
    sess = create_tf_session()

    learner = args.ddpg.make_learner(env)

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for _ in range(args.run.episodes):
        with timeit('learner fit'):
            if data.size:
                learner.fit(data)

        with timeit('sample learner'):
            paths = sample_venv(venv, learner, args.experiment.horizon)
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = one_shot_dataset(paths)
            returns = most_recent.per_episode_rewards()

        reporter.add_summary_statistics('return', returns)
        reporter.advance_iteration()

    sess.__exit__(None, None, None)


class RunFlags(Flags):
    """Flags relevant for running DDPG over multiple iterations."""

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Adds flags to an argparse parser."""
        if argument_group is None:
            argument_group = parser.add_argument_group('run')
        argument_group.add_argument(
            '--episodes',
            type=int,
            default=300,
            help='number episodes to train on',
        )

    @staticmethod
    def name():
        return "run"

    def __init__(self, args):
        self.episodes = args.episodes


if __name__ == "__main__":
    flags = [ExperimentFlags, RunFlags, DDPGLearnerFlags]
    _args = parse_args(flags)
    experiment_main(_args, _train)
