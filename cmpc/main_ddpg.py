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
from flags import parse_args
import tfnode
from persistable_dataset import (
    add_dataset_to_persistance_registry, PersistableDatasetFlags)
import reporter
from sample import sample, Sampler
from utils import timeit, as_controller, make_session_as_default


def _train(_):
    with closing(env_info.make_env()) as env:
        sampler = Sampler(env)
        data = Dataset.from_env(env, flags().experiment.horizon,
                                flags().experiment.bufsize)
        add_dataset_to_persistance_registry(data, flags().persistable_dataset)
        need_dynamics = (
            flags().ddpg.mixture_estimator == 'learned' or
            flags().ddpg.imaginary_buffer > 0)
        if need_dynamics:
            dynamics = NNDynamicsModel(env, data, flags().dynamics)
        else:
            dynamics = None

        learner = DDPGLearner(dynamics=dynamics)
        with make_session_as_default():
            tf.global_variables_initializer().run()
            tf.get_default_graph().finalize()
            tfnode.restore_all()

            _loop(sampler, data, learner, dynamics)


def _loop(sampler, data, learner, dynamics):
    while flags().experiment.should_continue():
        steps_sampled = 0 if data.size == 0 else sampler.nsteps()

        if dynamics:
            with timeit('dynamics fit'):
                dynamics.fit(data, steps_sampled)

        with timeit('learner fit'):
            learner.fit(data, steps_sampled)

        with timeit('sample learner'):
            controller = as_controller(learner.act)
            n_episodes = sampler.sample(controller, data)

        reporter.advance(sampler.nsteps(), n_episodes)
        if flags().experiment.should_render():
            with flags().experiment.render_env() as render_env:
                sample(render_env, controller, render=True)
        if flags().experiment.should_save():
            tfnode.save_all(reporter.timestep())


if __name__ == "__main__":
    _flags = [ExperimentFlags(), PersistableDatasetFlags(),
              DDPGFlags(), DynamicsFlags(), DynamicsMetricsFlags()]
    _args = parse_args(_flags)
    experiment_main(_args, _train)
