"""Generate SAC rollouts and train on them"""

from contextlib import closing

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from context import flags
from dataset import Dataset
from sac_learner import SACLearner, SACFlags
from dynamics import DynamicsFlags, NNDynamicsModel
from dynamics_metrics import DynamicsMetricsFlags
import env_info
from experiment import ExperimentFlags, experiment_main
from flags import parse_args
import tfnode
from persistable_dataset import (
    add_dataset_to_persistance_registry, PersistableDatasetFlags)
import reporter
from sample import sample, Sampler, SamplerFlags
from utils import timeit, as_controller, make_session_as_default

# TODO need to dedup this main file and common abstractions
# e.g., imaginary buffer, between SAC and DDPG. All implementation details too.


def train(_):
    """
    Runs the SAC + MVE training procedure, reading from the global flags.
    """
    with closing(env_info.make_env()) as env:
        sampler = Sampler(env)
        data = Dataset.from_env(env, flags().experiment.horizon,
                                flags().experiment.bufsize)
        add_dataset_to_persistance_registry(data, flags().persistable_dataset)
        need_dynamics = (
            flags().sac.q_target_mixture or
            flags().sac.imaginary_buffer > 0)
        if need_dynamics:
            dynamics = NNDynamicsModel(env, data, flags().dynamics)
        else:
            dynamics = None

        learner = SACLearner(dynamics=dynamics)
        with make_session_as_default():
            tf.global_variables_initializer().run()
            tf.get_default_graph().finalize()
            tfnode.restore_all()

            _loop(sampler, data, learner, dynamics)


def _loop(sampler, data, learner, _):
    while flags().experiment.should_continue():
        with timeit('sample learner'):
            controller = as_controller(learner.act)
            n_episodes = sampler.sample(controller, data)
        reporter.advance(sampler.nsteps(), n_episodes)

#        if dynamics:
#             with timeit('dynamics fit'):
#                 dynamics.fit(data, sampler.nsteps())

        with timeit('learner fit'):
            learner.fit(data, sampler.nsteps())

        if flags().experiment.should_render():
            with flags().experiment.render_env() as render_env:
                sample(render_env, controller, render=True)
        if flags().experiment.should_save():
            tfnode.save_all(reporter.timestep())

        reporter.report()


ALL_SAC_FLAGS = [ExperimentFlags(), PersistableDatasetFlags(),
                 SACFlags(), DynamicsFlags(), DynamicsMetricsFlags(),
                 SamplerFlags()]

if __name__ == "__main__":
    _args = parse_args(ALL_SAC_FLAGS)
    experiment_main(_args, train)
