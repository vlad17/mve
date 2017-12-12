"""Generate possibly constrained MPC rollouts."""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf

from dataset import Dataset, one_shot_dataset
from dynamics import DynamicsFlags, NNDynamicsModel
from experiment import experiment_main, ExperimentFlags
from flags import parse_args
from mpc_flags import SharedMPCFlags
from colocation_flags import ColocationFlags
from random_shooter_flags import RandomShooterFlags
from multiprocessing_env import make_venv
from sample import sample_venv, sample
from utils import (timeit, create_tf_session)
from warmup import add_warmup_data, WarmupFlags
import reporter


def train(args):
    """
    Train constrained MPC with the specified flags and subflags.
    """
    env = args.experiment.make_env()
    data = Dataset.from_env(env, args.experiment.horizon,
                            args.experiment.bufsize)
    with timeit('gathering warmup data'):
        add_warmup_data(args, data)
    venv = make_venv(args.experiment.make_env, args.mpc.onpol_paths)
    sess = create_tf_session()
    controller_flags = args.subflag

    dyn_model = NNDynamicsModel(
        env, data, args.dynamics, args.mpc.mpc_horizon,
        args.experiment.make_env)
    controller = controller_flags.make_mpc(
        env, dyn_model, args.mpc.mpc_horizon)
    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

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
            most_recent = one_shot_dataset(paths)
            most_recent.log_reward_bias(args.mpc.mpc_horizon)
            returns = most_recent.per_episode_rewards()
            reporter.add_summary_statistics('reward', returns)
            dyn_model.log(most_recent)
            controller.log(most_recent)

        reporter.advance_iteration()

        if args.experiment.render_every > 0 and (
                (itr + 1) % args.experiment.render_every == 0 or
                itr == 0):
            render_env = args.experiment.render_env(env, itr + 1)
            _ = sample(
                render_env, controller, args.experiment.horizon, render=True)

    sess.__exit__(None, None, None)


def flags_to_parse():
    """Flags that BMPC should parse"""
    flags = [ExperimentFlags(), SharedMPCFlags(), DynamicsFlags(),
             WarmupFlags()]
    subflags = RandomShooterFlags.all_subflags()
    subflags.append(ColocationFlags())
    return flags, subflags


if __name__ == "__main__":
    _flags, _subflags = flags_to_parse()
    _args = parse_args(_flags, _subflags)
    experiment_main(_args, train)
