"""Generate possibly constrained MPC rollouts."""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np

from cloning_learner import CloningLearnerFlags
from dataset import Dataset, one_shot_dataset
from ddpg_learner_flags import DDPGLearnerFlags
from dynamics import DynamicsFlags, NNDynamicsModel
from experiment import experiment_main, ExperimentFlags
from flags import parse_args_with_subcmds
from learner_flags import NoLearnerFlags
from mpc_flags import MpcFlags, RandomShooterFlags
from multiprocessing_env import make_venv
from sample import sample_venv
from utils import (timeit, create_tf_session)
from warmup import add_warmup_data, WarmupFlags
from zero_learner import ZeroLearnerFlags
import reporter


def train(args, learner_flags):
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

    if not args.dynamics.normalize and args.warmup.warmup_paths_random > 0:
        raise ValueError('What are you running random warmup paths for '
                         'without normalization, you silly goose?')

    dyn_model = NNDynamicsModel(env, data, args.dynamics)
    learner = learner_flags.make_learner(env)
    controller = args.mpc.make_mpc(
        env, dyn_model, env.tf_reward, learner, args)
    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for _ in range(args.mpc.onpol_iters):
        with timeit('dynamics fit'):
            if data.size:
                dyn_model.fit(data)

        with timeit('learner fit'):
            if data.size and learner:
                learner.fit(data)

        with timeit('sample controller'):
            paths = sample_venv(venv, controller, args.experiment.horizon)

        with timeit('adding paths to dataset'):
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = one_shot_dataset(paths)
            returns = most_recent.per_episode_rewards()
            mse = dyn_model.dataset_mse(most_recent)
            bias, _ = most_recent.reward_bias(args.mpc.horizon)
            ave_bias = bias.mean()
            ave_sqerr = np.square(bias).mean()
            # TODO: bootstrap ave_bias ci, ave_sqerr ci
            if learner:
                learner.log(most_recent)
                # out-of-band learner evaluation
                learner_paths = sample_venv(venv, learner, data.max_horizon)
                learner_data = one_shot_dataset(learner_paths)
                learner_returns = learner_data.per_episode_rewards()
                reporter.add_summary_statistics(
                    'learner reward', learner_returns)

        reporter.add_summary_statistics('reward', returns)
        reporter.add_summary('dynamics mse', mse)
        reporter.add_summary('reward bias', ave_bias)
        reporter.add_summary('reward mse', ave_sqerr)
        reporter.advance_iteration()

    sess.__exit__(None, None, None)


def flags_to_parse():
    """Flags that BMPC should parse"""
    flags = [ExperimentFlags, MpcFlags, DynamicsFlags, WarmupFlags,
             RandomShooterFlags]
    subflags = [CloningLearnerFlags, DDPGLearnerFlags,
                ZeroLearnerFlags, NoLearnerFlags]
    return flags, subflags


if __name__ == "__main__":
    _flags, _subflags = flags_to_parse()
    _args, _learner_flags = parse_args_with_subcmds(_flags, _subflags)
    experiment_main(_args, train, _learner_flags)
