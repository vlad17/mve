"""Generate DDPG rollouts and train on them"""

from contextlib import closing

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from context import flags
from dataset import Dataset
from ddpg_learner import DDPGLearner, DDPGFlags
from dynamics import DynamicsFlags, NNDynamicsModel
from dynamics_metrics import DynamicsMetricsFlags
import env_info
from experiment import ExperimentFlags, experiment_main
from flags import (parse_args, Flags, ArgSpec)
import tfnode
from persistable_dataset import (
    add_dataset_to_persistance_registry, PersistableDatasetFlags)
import reporter
from sample import sample
from utils import timeit, as_controller, timesteps, make_session_as_default


def _train(_):
    with closing(env_info.make_env()) as env:
        data = Dataset.from_env(env, flags().experiment.horizon,
                                flags().experiment.bufsize)
        add_dataset_to_persistance_registry(data, flags().persistable_dataset)
        if flags().ddpg.mixture_estimator == 'learned':
            dynamics = NNDynamicsModel(env, data, flags().dynamics)
        else:
            dynamics = None
        learner = DDPGLearner(dynamics=dynamics)
        with make_session_as_default():
            tf.global_variables_initializer().run()
            tf.get_default_graph().finalize()
            tfnode.restore_all()

            _loop(env, data, learner, dynamics)


def _loop(env, data, learner, dynamics):
    paths = []
    for itr in range(flags().run.episodes):
        if dynamics:
            with timeit('dynamics fit'):
                dynamics.fit(data, timesteps(paths))

        with timeit('learner fit'):
            learner.fit(data, timesteps(paths))

        with timeit('sample learner'):
            controller = as_controller(learner.act)
            paths = [sample(env, controller, render=False)]
            data.add_paths(paths)

        with timeit('gathering statistics'):
            rewards = [path.rewards.sum() for path in paths]
            reporter.add_summary_statistics('reward', rewards)

        reporter.advance(paths)
        if flags().experiment.should_render(itr):
            with flags().experiment.render_env(itr + 1) as render_env:
                sample(render_env, controller, render=True)
        if flags().experiment.should_save(itr):
            tfnode.save_all(itr + 1)


class RunFlags(Flags):
    """Flags relevant for running DDPG over multiple iterations."""

    def __init__(self):
        arguments = [
            ArgSpec(
                name='episodes',
                type=int,
                default=300,
                help='number episodes to train on')]
        super().__init__('run', 'run flags for ddpg', arguments)


if __name__ == "__main__":
    _flags = [ExperimentFlags(), RunFlags(), PersistableDatasetFlags(),
              DDPGFlags(), DynamicsFlags(), DynamicsMetricsFlags()]
    _args = parse_args(_flags)
    experiment_main(_args, _train)
