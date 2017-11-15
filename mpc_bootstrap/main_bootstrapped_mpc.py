"""Generate bootstrapped MPC rollouts."""

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
                   create_tf_session)
from warmup import add_warmup_data, WarmupFlags
from zero_learner_flags import ZeroLearnerFlags
import log
import logz


def _train(args, learner_flags):
    env = args.experiment.mk_env()
    data = Dataset.from_env(env, args.mpc.horizon, args.mpc.bufsize)
    with timeit('gathering warmup data'):
        add_warmup_data(args, data)
    venv = mk_venv(args.experiment.mk_env, args.mpc.onpol_paths)
    sess = create_tf_session()

    dyn_model = NNDynamicsModel(venv, sess, data, args.dynamics)
    learner = learner_flags.make_learner(venv, sess, data)
    controller = BootstrappedMPC(
        venv, dyn_model, args.mpc.mpc_horizon,
        env.tf_reward, args.mpc.mpc_simulated_paths, learner, sess)

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for itr in range(args.mpc.onpol_iters):
        with timeit('dynamics fit'):
            if data.obs.size:
                dyn_model.fit(data)

        with timeit('learner fit'):
            if data.obs.size:
                learner.fit(data)

        with timeit('sample controller'):
            paths = sample_venv(venv, controller, args.mpc.horizon)

        with timeit('adding paths to dataset'):
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = one_shot_dataset(paths)
            returns = most_recent.per_episode_rewards()
            mse = dyn_model.dataset_mse(most_recent)
            bias, zero_bias = most_recent.reward_bias(args.mpc.mpc_horizon)
            ave_bias = bias.mean() / np.fabs(zero_bias.mean())
            ave_sqerr = np.square(bias).mean() / np.square(zero_bias).mean()
            # TODO: bootstrap ave_bias ci, ave_sqerr ci

            learner.log(most_recent)

            # out-of-band learner evaluation
            # TODO: move this (get paths, make empty dataset, fill it with
            # paths) sequence into a utility method and dedup
            learner_paths = sample_venv(venv, learner, data.max_horizon)
            learner_data = one_shot_dataset(learner_paths)
            learner_returns = learner_data.per_episode_rewards()
            # TODO: make logging an Average,Std,Min,Max a utility method
            logz.log_tabular('LearnerAverageReturn', np.mean(learner_returns))
            logz.log_tabular('LearnerStdReturn', np.std(learner_returns))
            logz.log_tabular('LearnerMinimumReturn', np.min(learner_returns))
            logz.log_tabular('LearnerMaximumReturn', np.max(learner_returns))

        logz.log_tabular('Iteration', itr)
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))
        logz.log_tabular('DynamicsMSE', mse)
        logz.log_tabular('StandardizedRewardBias', ave_bias)
        logz.log_tabular('StandardizedRewardMSE', ave_sqerr)
        logz.dump_tabular()

    sess.__exit__(None, None, None)


def _main(args, learner_flags):
    log.init(args.experiment.verbose)

    exp_name = args.experiment.exp_name
    env_name = args.experiment.env_name
    datadir = "{}_{}".format(exp_name, env_name)
    logdir = make_data_directory(datadir)

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
            _train(args, learner_flags)


if __name__ == "__main__":
    flags = [ExperimentFlags, MpcFlags, DynamicsFlags, WarmupFlags]
    subflags = [DeterministicLearnerFlags, StochasticLearnerFlags,
                DDPGLearnerFlags, ZeroLearnerFlags]
    _args, _learner_flags = parse_args_with_subcmds(flags, subflags)
    _main(_args, _learner_flags)
