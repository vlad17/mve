"""Generate possibly constrained MPC rollouts."""

from contextlib import closing

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from dataset import Dataset
from dynamics import DynamicsFlags, NNDynamicsModel
from dynamics_metrics import DynamicsMetricsFlags, DynamicsMetrics
import env_info
from experiment import experiment_main, ExperimentFlags
from immutable_dataset import ImmutableDataset
from flags import parse_args
from mpc_flags import MPCFlags
from persistable_dataset import (
    add_dataset_to_persistance_registry, PersistableDatasetFlags)
from colocation_flags import ColocationFlags
from random_shooter_flags import RandomShooterFlags
from ddpg_learner import DDPGFlags
from cloning_learner import CloningLearnerFlags
import tfnode
from sample import sample_venv, sample
from utils import timeit, timesteps, make_session_as_default
import reporter


def train(args):
    """
    Train constrained MPC with the specified flags and subflags.
    """
    with closing(env_info.make_env()) as env, \
        closing(env_info.make_venv(args.mpc.onpol_paths)) as venv, \
        closing(DynamicsMetrics(
            args.mpc.mpc_horizon, env_info.make_env,
            args.dynamics_metrics, args.experiment.discount)) as dyn_metrics:

        # TF graph construction
        data = Dataset.from_env(
            env, args.experiment.horizon, args.experiment.bufsize)
        dyn_model = NNDynamicsModel(env, data, args.dynamics)
        controller = args.mpc.make_mpc(dyn_model)
        add_dataset_to_persistance_registry(data, args.persistable_dataset)

        # actually run stuff
        with make_session_as_default():
            tf.global_variables_initializer().run()
            tf.get_default_graph().finalize()
            _train(args, venv, dyn_metrics, data, dyn_model, controller)


def _train(args, venv, dyn_metrics, data, dyn_model, controller):
    tfnode.restore_all()
    paths = []

    for itr in range(args.mpc.onpol_iters):
        with timeit('dynamics fit'):
            dyn_model.fit(data, timesteps(paths))

        with timeit('controller fit'):
            controller.fit(data, timesteps(paths))

        with timeit('sample controller'):
            paths = sample_venv(venv, controller)
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = ImmutableDataset(paths)
            most_recent.log_reward()
            dyn_metrics.log(most_recent)
            controller.log(most_recent)

        reporter.advance(paths)

        if args.experiment.should_render(itr):
            with args.experiment.render_env(itr + 1) as render_env:
                sample(render_env, controller, render=True)

        if args.experiment.should_save(itr):
            tfnode.save_all(itr + 1)


def flags_to_parse():
    """Flags that BMPC should parse"""
    flags = [ExperimentFlags(), MPCFlags(), DynamicsFlags(),
             DynamicsMetricsFlags(), PersistableDatasetFlags(),
             RandomShooterFlags(), DDPGFlags(), CloningLearnerFlags(),
             ColocationFlags()]
    return flags


if __name__ == "__main__":
    _flags = flags_to_parse()
    _args = parse_args(_flags)
    experiment_main(_args, train)
