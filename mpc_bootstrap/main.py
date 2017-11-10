"""Bootstrap MPC entry point."""

import json
import os
import random

# import mujoco for weird dlopen reasons
import mujoco_py # pylint: disable=unused-import
import numpy as np
import tensorflow as tf

from controllers import (
    RandomController, MPC, BootstrappedMPC, DaggerMPC,
    DeterministicLearner, StochasticLearner, LearnerOnly,
    DDPGLearner, ZeroLearner)
from dynamics import NNDynamicsModel
from envs import WhiteBoxHalfCheetahEasy, WhiteBoxHalfCheetahHard
from log import debug
import log
from multiprocessing_env import MultiprocessingEnv
from utils import Path, Dataset, timeit, make_data_directory
import flags
import logz


def sample(env,
           controller,
           horizon=1000):
    """
    Assumes env is vectorized to some number of states at once
    """
    obs_n = env.reset()
    controller.reset(len(obs_n))
    paths = [Path(env, obs, horizon) for obs in obs_n]
    for t in range(horizon):
        acs_n, predicted_rewards_n = controller.act(obs_n)
        obs_n, reward_n, done_n, _ = env.step(acs_n)
        for p, path in enumerate(paths):
            done_n[p] |= path.next(obs_n[p], reward_n[p], acs_n[p],
                                   predicted_rewards_n[p])
            if done_n[p]:
                assert t + 1 == horizon, (t + 1, horizon)
    return paths


def _mklearner(venv, all_flags, sess, data):
    learner_name = all_flags.algorithm.agent.split('_')[0]
    if learner_name == 'zero':
        learner = ZeroLearner(venv)
    elif learner_name == 'delta':
        learner = DeterministicLearner(
            env=venv,
            learning_rate=all_flags.controller.con_learning_rate,
            depth=all_flags.controller.con_depth,
            width=all_flags.controller.con_width,
            batch_size=all_flags.controller.con_batch_size,
            epochs=all_flags.controller.con_epochs,
            explore_std=all_flags.controller.explore_std,
            sess=sess)
    elif learner_name == 'gaussian':
        learner = StochasticLearner(
            env=venv,
            learning_rate=all_flags.controller.con_learning_rate,
            depth=all_flags.controller.con_depth,
            width=all_flags.controller.con_width,
            batch_size=all_flags.controller.con_batch_size,
            epochs=all_flags.controller.con_epochs,
            no_extra_explore=all_flags.controller.no_extra_explore,
            sess=sess)
    elif learner_name == 'ddpg':
        learner = DDPGLearner(venv, data, sess, all_flags.controller)
    else:
        raise ValueError('learner type {} unsupported'.format(learner_name))
    return learner


def _train(all_flags, logdir): # pylint: disable=too-many-branches
    # Save params to disk.
    params = flags.flags_to_json(all_flags)
    logz.configure_output_dir(logdir)
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)

    # Log arguments to sanity check that they are what they should be.
    for x in params:
        debug("{:20} = {}", x, params[x])

    def mk_env():
        """Generates an unvectorized env."""
        env_name = all_flags.algorithm.env_name
        if env_name == 'hc-hard':
            return WhiteBoxHalfCheetahHard(all_flags.algorithm.frame_skip)
        elif env_name == 'hc-easy':
            return WhiteBoxHalfCheetahEasy(all_flags.algorithm.frame_skip)
        else:
            raise ValueError('env {} unsupported'.format(env_name))

    def mk_vectorized_env(n):
        """Generates vectorized multiprocessing env."""
        envs = [mk_env() for _ in range(n)]
        mp_env = MultiprocessingEnv(envs)
        seeds = [int(s) for s in np.random.randint(0, 2 ** 30, size=n)]
        mp_env.seed(seeds)
        return mp_env

    # Need random initial data to kickstart dynamics
    original_env = mk_env()
    data = Dataset(original_env, all_flags.algorithm.horizon)
    random_controller = RandomController(original_env)
    if all_flags.algorithm.random_paths > 0:
        venv = mk_vectorized_env(all_flags.algorithm.random_paths)
        paths = sample(venv, random_controller, all_flags.algorithm.horizon)
        data.add_paths(paths)

    venv = mk_vectorized_env(all_flags.algorithm.onpol_paths)

    # Build dynamics model and MPC controllers.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    opt_opts = config.graph_options.optimizer_options
    opt_opts.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    dyn_model = NNDynamicsModel(
        env=venv,
        sess=sess,
        norm_data=data,
        depth=all_flags.dynamics.dyn_depth,
        width=all_flags.dynamics.dyn_width,
        learning_rate=all_flags.dynamics.dyn_learning_rate,
        epochs=all_flags.dynamics.dyn_epochs,
        batch_size=all_flags.dynamics.dyn_batch_size,
        no_delta_norm=all_flags.dynamics.no_delta_norm)

    if all_flags.algorithm.agent == 'mpc':
        controller = MPC(env=venv,
                         dyn_model=dyn_model,
                         horizon=all_flags.mpc.mpc_horizon,
                         reward_fn=original_env.tf_reward,
                         num_simulated_paths=all_flags.mpc.mpc_simulated_paths,
                         sess=sess,
                         learner=None)
    elif all_flags.algorithm.agent == 'random':
        controller = random_controller
    elif all_flags.algorithm.agent.split('_')[1] == 'learneronly':
        learner = _mklearner(venv, all_flags, sess, data)
        controller = LearnerOnly(learner)
    elif all_flags.algorithm.agent.split('_')[1] in ['bootstrap', 'dagger']:
        learner = _mklearner(venv, all_flags, sess, data)
        if all_flags.algorithm.agent.split('_')[1] == 'bootstrap':
            controller = BootstrappedMPC(
                env=venv,
                dyn_model=dyn_model,
                horizon=all_flags.mpc.mpc_horizon,
                reward_fn=original_env.tf_reward,
                num_simulated_paths=all_flags.mpc.mpc_simulated_paths,
                learner=learner,
                sess=sess)
        else:  # dagger
            controller = DaggerMPC(
                env=venv,
                dyn_model=dyn_model,
                horizon=all_flags.mpc.mpc_horizon,
                delay=all_flags.mpc.delay,
                reward_fn=original_env.tf_reward,
                num_simulated_paths=all_flags.mpc.mpc_simulated_paths,
                learner=learner,
                sess=sess)
    else:
        agent = all_flags.algorithm.agent
        raise ValueError('agent type {} unrecognized'.format(agent))

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for itr in range(all_flags.algorithm.onpol_iters):
        with timeit('labelling actions'):
            to_label = data.unlabelled_obs()
            labels = controller.label(to_label)
            data.label_obs(labels)

        if not all_flags.algorithm.disable_dynamics:
            with timeit('dynamics fit'):
                dyn_model.fit(data)

        with timeit('controller fit'):
            if data.stationary_obs().size:
                controller.fit(data)

        with timeit('sample controller'):
            paths = sample(venv, controller, all_flags.algorithm.horizon)

        with timeit('adding paths to dataset'):
            data.add_paths(paths)

        with timeit('gathering statistics'):
            most_recent = Dataset(venv, all_flags.algorithm.horizon)
            most_recent.add_paths(paths)
            returns = most_recent.rewards.sum(axis=0)
            mse = 0
            if not all_flags.algorithm.disable_dynamics:
                mse = dyn_model.dataset_mse(most_recent)
            mpc_horizon = 1
            if hasattr(all_flags, 'mpc') and \
               hasattr(all_flags.mpc, 'mpc_horizon'):
                mpc_horizon = all_flags.mpc.mpc_horizon
            bias, zero_bias = most_recent.reward_bias(mpc_horizon)
            ave_bias = bias.mean() / np.fabs(zero_bias.mean())
            ave_sqerr = np.square(bias).mean() / np.square(zero_bias).mean()
            # TODO: bootstrap ave_bias ci, ave_sqerr ci
            controller.log(horizon=all_flags.algorithm.horizon,
                           most_recent=most_recent)

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


def _main():
    # Parse arguments.
    all_flags = flags.get_all_flags()

    # Initialize the logger.
    log.init(all_flags.experiment.verbose)

    # Make data directory if it does not already exist.
    datadir = "{}_{}".format(all_flags.experiment.exp_name,
                             all_flags.algorithm.env_name)
    logdir = make_data_directory(datadir)

    # Run the experiments!
    for seed in all_flags.experiment.seed:
        g = tf.Graph()
        logdir_seed = os.path.join(logdir, str(seed))
        with g.as_default():
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)
            _train(all_flags=all_flags, logdir=logdir_seed)


if __name__ == "__main__":
    _main()
