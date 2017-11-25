"""Generate bootstrapped MPC rollouts."""

import collections
import json
import os

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np

from bootstrapped_mpc import BootstrappedMPC
from dataset import Dataset, one_shot_dataset
from ddpg_learner_flags import DDPGLearnerFlags
from deterministic_learner_flags import DeterministicLearnerFlags
from dynamics import DynamicsFlags, NNDynamicsModel
from experiment_flags import ExperimentFlags
from flags import (convert_flags_to_json, parse_args_with_subcmds)
from mpc_flags import MpcFlags
from multiprocessing_env import mk_venv
from sample import sample_venv
from stochastic_learner_flags import StochasticLearnerFlags
from utils import (make_data_directory, seed_everything, timeit,
                   create_tf_session, log_statistics)
from warmup import add_warmup_data, WarmupFlags
from zero_learner_flags import ZeroLearnerFlags
import log
import logz


def _train(args, learner_flags, smoothing, status_reporter):
    env = args.experiment.mk_env()
    data = Dataset.from_env(env, args.experiment.horizon,
                            args.experiment.bufsize)
    with timeit('gathering warmup data'):
        add_warmup_data(args, data)
    venv = mk_venv(args.experiment.mk_env, args.mpc.onpol_paths)
    sess = create_tf_session()

    dyn_model = NNDynamicsModel(venv, sess, data, args.dynamics)
    learner = learner_flags.make_learner(venv, sess)
    controller = BootstrappedMPC(
        venv, dyn_model, args.mpc.mpc_horizon,
        env.tf_reward, args.mpc.mpc_simulated_paths, learner, sess)
    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    historical_returns = collections.deque()
    for itr in range(args.mpc.onpol_iters):
        with timeit('dynamics fit'):
            if data.size:
                dyn_model.fit(data)

        with timeit('learner fit'):
            if data.size:
                learner.fit(data)

        with timeit('sample controller'):
            paths = sample_venv(venv, controller, args.experiment.horizon)

        with timeit('adding paths to dataset'):
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = one_shot_dataset(paths)
            returns = most_recent.per_episode_rewards()
            mse = dyn_model.dataset_mse(most_recent)
            bias, _ = most_recent.reward_bias(args.mpc.mpc_horizon)
            ave_bias = bias.mean()
            ave_sqerr = np.square(bias).mean()
            # TODO: bootstrap ave_bias ci, ave_sqerr ci
            learner.log(most_recent)
            # out-of-band learner evaluation
            learner_paths = sample_venv(venv, learner, data.max_horizon)
            learner_data = one_shot_dataset(learner_paths)
            learner_returns = learner_data.per_episode_rewards()
            log_statistics('learner-return', learner_returns)

        if len(historical_returns) == smoothing:
            historical_returns.popleft()
        historical_returns.append(np.mean(returns))
        if len(historical_returns) == smoothing and status_reporter:
            status_reporter(np.mean(historical_returns))

        logz.log_tabular('iteration', itr)
        log_statistics('return', returns)
        logz.log_tabular('dynamics-mse', mse)
        logz.log_tabular('standardized-reward-bias', ave_bias)
        logz.log_tabular('standardized-reward-mse', ave_sqerr)
        logz.dump_tabular()

    sess.__exit__(None, None, None)


def main(args, learner_flags, smoothing=None, status_reporter=None):
    """
    With args and learner_flags as specified in __main__ below,
    this runs the specified bootstrapped_mpc.

    Reports the average performance of the last smoothing number
    of iterations for all seeds.
    """
    log.init(args.experiment.verbose)
    logdir_name = args.experiment.log_directory()
    logdir = make_data_directory(logdir_name)

    for seed in args.experiment.seed:
        # Save params to disk.
        logdir_seed = os.path.join(logdir, str(seed))
        logz.configure_output_dir(logdir_seed)
        with open(os.path.join(logdir_seed, 'params.json'), 'w') as f:
            json.dump(convert_flags_to_json(args), f, sort_keys=True, indent=4)

        # Run experiment.
        g = tf.Graph()
        with g.as_default():
            seed_everything(seed)
            _train(args, learner_flags, smoothing or 1, status_reporter)


def flags_to_parse():
    """Flags that BMPC should parse"""
    flags = [ExperimentFlags, MpcFlags, DynamicsFlags, WarmupFlags]
    subflags = [DeterministicLearnerFlags, StochasticLearnerFlags,
                DDPGLearnerFlags, ZeroLearnerFlags]
    return flags, subflags


if __name__ == "__main__":
    _flags, _subflags = flags_to_parse()
    _args, _learner_flags = parse_args_with_subcmds(_flags, _subflags)
    main(_args, _learner_flags)
