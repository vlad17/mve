"""Generate bootstrapped MPC rollouts."""

import json
import os

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np

from bootstrapped_mpc_flags import BootstrappedMpcFlags
from ddpg_learner_flags import DDPGLearnerFlags
from deterministic_learner_flags import DeterministicLearnerFlags
from dynamics_flags import DynamicsFlags
from experiment_flags import ExperimentFlags
from flags import (convert_flags_to_json, parse_args_with_subcmds)
from multiprocessing_env import mk_venv
from sample import sample_venv
from stochastic_learner_flags import StochasticLearnerFlags
from utils import (Dataset, make_data_directory, seed_everything, timeit,
                   create_tf_session)
from warmup import add_warmup_data, WarmupFlags
from zero_learner_flags import ZeroLearnerFlags
import log
import logz


def _train(args, learner_flags):
    env = args.experiment.mk_env()
    data = Dataset(env, args.bmpc.horizon)
    with timeit('gathering warmup data'):
        add_warmup_data(args, args.bmpc, data)
    venv = mk_venv(args.experiment.mk_env, args.bmpc.onpol_paths)
    sess = create_tf_session()

    dyn_model = args.dynamics.make_dynamics(venv, sess, data)
    learner = learner_flags.make_learner(venv, sess, data)
    controller = args.bmpc.make_controller(env, venv, sess, dyn_model, learner)

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for itr in range(args.bmpc.onpol_iters):
        with timeit('learner fit'):
            if data.stationary_obs().size:
                learner.fit(data, use_labelled=False)

        with timeit('sample controller'):
            paths = sample_venv(venv, controller, args.bmpc.horizon)

        with timeit('adding paths to dataset'):
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = Dataset(venv, args.bmpc.horizon)
            most_recent.add_paths(paths)
            returns = most_recent.rewards.sum(axis=0)
            mse = dyn_model.dataset_mse(most_recent)
            bias, zero_bias = most_recent.reward_bias(args.bmpc.mpc_horizon)
            ave_bias = bias.mean() / np.fabs(zero_bias.mean())
            ave_sqerr = np.square(bias).mean() / np.square(zero_bias).mean()
            # TODO: bootstrap ave_bias ci, ave_sqerr ci
            controller.log(horizon=args.bmpc.horizon, most_recent=most_recent)

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
    flags = [ExperimentFlags, BootstrappedMpcFlags, DynamicsFlags, WarmupFlags]
    subflags = [DeterministicLearnerFlags, StochasticLearnerFlags,
                DDPGLearnerFlags, ZeroLearnerFlags]
    _args, _learner_flags = parse_args_with_subcmds(flags, subflags)
    _main(_args, _learner_flags)
