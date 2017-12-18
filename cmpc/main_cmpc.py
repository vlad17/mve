"""Generate possibly constrained MPC rollouts."""

from contextlib import closing

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from dataset import Dataset
from dynamics import DynamicsFlags, NNDynamicsModel
from dynamics_metrics import DynamicsMetricsFlags, DynamicsMetrics
from experiment import experiment_main, ExperimentFlags
from immutable_dataset import ImmutableDataset
from flags import parse_args
from mpc_flags import SharedMPCFlags
from persistable_dataset import (
    add_dataset_to_persistance_registry, PersistableDatasetFlags)
from colocation_flags import ColocationFlags
from random_shooter_flags import RandomShooterFlags
import tfnode
from multiprocessing_env import make_venv
from sample import sample_venv, sample
from utils import timeit
import reporter


def train(args):
    """
    Train constrained MPC with the specified flags and subflags.
    """
    with closing(args.experiment.make_env()) as env, \
        closing(make_venv(args.experiment.make_env, args.mpc.onpol_paths)) \
        as venv, \
        closing(DynamicsMetrics(
            args.mpc.mpc_horizon, args.experiment.make_env,
            args.dynamics_metrics)) as dyn_metrics:
        _train(args, env, venv, dyn_metrics)


def _train(args, env, venv, dyn_metrics):
    data = Dataset.from_env(
        env, args.experiment.horizon, args.experiment.bufsize)
    controller_flags = args.subflag
    dyn_model = NNDynamicsModel(env, data, args.dynamics)
    controller = controller_flags.make_mpc(
        env, dyn_model, args.mpc.mpc_horizon)
    add_dataset_to_persistance_registry(data, args.persistable_dataset)

    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()
    tfnode.restore_all()

    for itr in range(args.mpc.onpol_iters):
        with timeit('dynamics fit'):
            if data.size:
                dyn_model.fit(data)

        with timeit('controller fit'):
            if data.size:
                controller.fit(data)

        with timeit('sample controller'):
            paths = sample_venv(venv, controller, args.experiment.horizon)
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = ImmutableDataset(paths)
            most_recent.log_reward()
            dyn_metrics.log(most_recent)
            controller.log(most_recent)

        reporter.advance_iteration()

        if args.experiment.should_render(itr):
            render_env = args.experiment.render_env(env, itr + 1)
            sample(
                render_env, controller, args.experiment.horizon, render=True)

        if args.experiment.should_save(itr):
            tfnode.save_all(itr + 1)


def flags_to_parse():
    """Flags that BMPC should parse"""
    flags = [ExperimentFlags(), SharedMPCFlags(), DynamicsFlags(),
             DynamicsMetricsFlags(), PersistableDatasetFlags()]
    subflags = RandomShooterFlags.all_subflags()
    subflags.append(ColocationFlags())
    return flags, subflags


if __name__ == "__main__":
    _flags, _subflags = flags_to_parse()
    _args = parse_args(_flags, _subflags)
    experiment_main(_args, train)
