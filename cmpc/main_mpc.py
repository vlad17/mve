"""Generate MPC rollouts."""

import os
import json

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np

from dataset import Dataset, one_shot_dataset
from dynamics import DynamicsFlags, NNDynamicsModel
from mpc import MPC
from mpc_flags import MpcFlags
from experiment_flags import ExperimentFlags
from flags import (convert_flags_to_json, parse_args)
from multiprocessing_env import mk_venv
from sample import sample_venv
from utils import (make_data_directory, seed_everything, timeit,
                   create_tf_session)
from warmup import add_warmup_data, WarmupFlags
import log
import reporter


def _train(args):
    env = args.experiment.mk_env()
    data = Dataset.from_env(env, args.experiment.horizon,
                            args.experiment.bufsize)
    with timeit('gathering warmup data'):
        add_warmup_data(args, data)
    venv = mk_venv(args.experiment.mk_env, args.mpc.onpol_paths)
    sess = create_tf_session()

    dyn_model = NNDynamicsModel(venv, sess, data, args.dynamics)
    controller = MPC(
        venv, dyn_model, args.mpc.mpc_horizon,
        env.tf_reward, args.mpc.mpc_simulated_paths, sess, learner=None)

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for _ in range(args.mpc.onpol_iters):
        with timeit('dynamics fit'):
            if data.size:
                dyn_model.fit(data)

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

        reporter.add_summary_statistics('reward', returns)
        reporter.add_summary('dynamics mse', mse)
        reporter.add_summary('reward bias', ave_bias)
        reporter.add_summary('reward mse', ave_sqerr)
        reporter.advance_iteration()

    sess.__exit__(None, None, None)


def _main(args):
    log.init(args.experiment.verbose)
    logdir_name = args.experiment.log_directory()
    logdir = make_data_directory(logdir_name)

    for seed in args.experiment.seed:
        # Save params to disk.
        logdir_seed = os.path.join(logdir, str(seed))
        os.makedirs(logdir_seed)
        with open(os.path.join(logdir_seed, 'params.json'), 'w') as f:
            json.dump(convert_flags_to_json(args), f, sort_keys=True, indent=4)

        # Run experiment.
        g = tf.Graph()
        with g.as_default():
            with reporter.create(logdir_seed, args.experiment.verbose):
                seed_everything(seed)
                _train(args)


if __name__ == "__main__":
    _args = parse_args([ExperimentFlags, MpcFlags, DynamicsFlags, WarmupFlags])
    _main(_args)
