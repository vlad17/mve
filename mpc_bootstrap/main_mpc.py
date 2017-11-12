"""Generate MPC rollouts."""

import os
import json

# import mujoco for weird dlopen reasons
import mujoco_py # pylint: disable=unused-import
import tensorflow as tf
import numpy as np

from dynamics_flags import DynamicsFlags
from mpc_flags import MpcFlags
from experiment_flags import ExperimentFlags
from flags import (convert_flags_to_json, parse_args)
from multiprocessing_env import mk_venv
from random_policy import RandomPolicy
from sample import sample_venv
from utils import (Dataset, make_data_directory, seed_everything, timeit)
import log
import logz


def _train(args):
    # Generate random rollouts.
    env = args.experiment.mk_env()
    data = Dataset(env, args.mpc.horizon)
    if args.mpc.random_paths > 0:
        with timeit('generating random rollouts'):
            venv = mk_venv(args.experiment.mk_env, args.mpc.random_paths)
            random_policy = RandomPolicy(venv)
            paths = sample_venv(venv, random_policy, args.mpc.horizon)
            data.add_paths(paths)

    venv = mk_venv(args.experiment.mk_env, args.mpc.onpol_paths)

    # Initialize tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    opt_opts = config.graph_options.optimizer_options
    opt_opts.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)

    dyn_model = args.dynamics.make_dynamics(venv, sess, data)
    controller = args.mpc.make_controller(env, venv, sess, dyn_model)

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for itr in range(args.mpc.onpol_iters):
        with timeit('dynamics fit'):
            dyn_model.fit(data)

        with timeit('sample controller'):
            paths = sample_venv(venv, controller, args.mpc.horizon)

        with timeit('adding paths to dataset'):
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = Dataset(venv, args.mpc.horizon)
            most_recent.add_paths(paths)
            returns = most_recent.rewards.sum(axis=0)
            mse = dyn_model.dataset_mse(most_recent)
            bias, zero_bias = most_recent.reward_bias(args.mpc.mpc_horizon)
            ave_bias = bias.mean() / np.fabs(zero_bias.mean())
            ave_sqerr = np.square(bias).mean() / np.square(zero_bias).mean()

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


def _main(args):
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
            _train(args)


if __name__ == "__main__":
    _args = parse_args([ExperimentFlags, MpcFlags, DynamicsFlags])
    _main(_args)
