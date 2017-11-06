"""Bootstrap MPC entry point."""

import shutil
import json
import time
import os

import numpy as np
import tensorflow as tf

from controllers import (
    RandomController, MPC, BootstrappedMPC, DaggerMPC,
    DeterministicLearner, StochasticLearner)
from dynamics import NNDynamicsModel
from envs import WhiteBoxHalfCheetahEasy, WhiteBoxHalfCheetahHard
from log import debug
import log
from multiprocessing_env import MultiprocessingEnv
from utils import Path, Dataset, timeit
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
        acs_n = controller.act(obs_n)
        obs_n, reward_n, done_n, _ = env.step(acs_n)
        for p, path in enumerate(paths):
            done_n[p] |= path.next(obs_n[p], reward_n[p], acs_n[p])
            if done_n[p]:
                assert t + 1 == horizon, (t + 1, horizon)
    return paths


def _train(all_flags, logdir):
    # Save params to disk.
    params = flags.flags_to_json(all_flags)
    logz.configure_output_dir(logdir)
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(params, f)

    # Log arguments to sanity check that they are what they should be.
    for x in params:
        debug("{:20} = {}", x, params[x])

    def mk_env():
        """Generates an unvectorized env."""
        env_name = all_flags.alg.env_name
        if env_name == 'hc-hard':
            return WhiteBoxHalfCheetahHard(all_flags.alg.frame_skip)
        elif env_name == 'hc-easy':
            return WhiteBoxHalfCheetahEasy(all_flags.alg.frame_skip)
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
    venv = mk_vectorized_env(all_flags.alg.random_paths)
    random_controller = RandomController(venv)
    paths = sample(venv, random_controller, all_flags.alg.horizon)
    data = Dataset(venv, all_flags.alg.horizon)
    data.add_paths(paths)
    venv = mk_vectorized_env(all_flags.alg.onpol_paths)

    original_env = mk_env()

    # Build dynamics model and MPC controllers.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    opt_opts = config.graph_options.optimizer_options
    opt_opts.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    dyn_model = NNDynamicsModel(env=venv,
                                sess=sess,
                                norm_data=data,
                                depth=all_flags.dyn.dyn_depth,
                                width=all_flags.dyn.dyn_width,
                                learning_rate=all_flags.dyn.dyn_learning_rate,
                                epochs=all_flags.dyn.dyn_epochs,
                                batch_size=all_flags.dyn.dyn_batch_size,
                                no_delta_norm=all_flags.dyn.no_delta_norm)

    if all_flags.alg.agent == 'mpc':
        controller = MPC(env=venv,
                         dyn_model=dyn_model,
                         horizon=all_flags.mpc.mpc_horizon,
                         reward_fn=original_env.tf_reward,
                         num_simulated_paths=all_flags.mpc.mpc_simulated_paths,
                         sess=sess,
                         learner=None)
    elif all_flags.alg.agent == 'random':
        controller = random_controller
    elif all_flags.alg.agent == 'bootstrap' or all_flags.alg.agent == 'dagger':
        if all_flags.con.deterministic_learner:
            learner = DeterministicLearner(
                env=venv,
                learning_rate=all_flags.con.con_learning_rate,
                depth=all_flags.con.con_depth,
                width=all_flags.con.con_width,
                batch_size=all_flags.con.con_batch_size,
                epochs=all_flags.con.con_epochs,
                explore_std=all_flags.con.explore_std,
                sess=sess)
        else:
            learner = StochasticLearner(
                env=venv,
                learning_rate=all_flags.con.con_learning_rate,
                depth=all_flags.con.con_depth,
                width=all_flags.con.con_width,
                batch_size=all_flags.con.con_batch_size,
                epochs=all_flags.con.con_epochs,
                no_extra_explore=all_flags.con.no_extra_explore,
                sess=sess)
        if all_flags.alg.agent == 'bootstrap':
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
        agent = all_flags.alg.agent
        raise ValueError('agent type {} unrecognized'.format(agent))

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for itr in range(all_flags.alg.onpol_iters):
        with timeit('labelling actions', all_flags.exp.time):
            to_label = data.unlabelled_obs()
            labels = controller.label(to_label)
            data.label_obs(labels)

        with timeit('dynamics fit', all_flags.exp.time):
            dyn_model.fit(data)

        with timeit('controller fit', all_flags.exp.time):
            controller.fit(data)

        with timeit('sample controller', all_flags.exp.time):
            paths = sample(venv, controller, all_flags.alg.horizon)

        data.add_paths(paths)

        with timeit('gathering statistics', all_flags.exp.time):
            most_recent = Dataset(venv, all_flags.alg.horizon)
            most_recent.add_paths(paths)
            returns = most_recent.rewards.sum(axis=0)
            mse = dyn_model.dataset_mse(most_recent)
            controller.log(horizon=all_flags.alg.horizon)

        logz.log_tabular('Iteration', itr)
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))
        logz.log_tabular('DynamicsMSE', mse)
        logz.dump_tabular()

    sess.__exit__(None, None, None)

def _main():
    # Parse arguments.
    all_flags = flags.get_all_flags()

    # Initialize the logger.
    log.init(all_flags.exp.verbose)

    # Make data directory if it does not already exist
    # TODO: this procedure should be in utils
    if not os.path.exists('data'):
        os.makedirs('data')
    logdir_base = all_flags.exp.exp_name + '_' + all_flags.alg.env_name
    logdir_base = os.path.join('data', logdir_base)
    ctr = 0
    logdir = logdir_base
    while os.path.exists(logdir):
        logdir = logdir_base + '-{}'.format(ctr)
        ctr += 1
    if ctr > 0:
        print('experiment already exists, moved old one to', logdir)
        shutil.move(logdir_base, logdir)
        logdir = logdir_base
    os.makedirs(logdir)
    with open(os.path.join(logdir, 'starttime.txt'), 'w') as f:
        print(time.strftime("%d-%m-%Y_%H-%M-%S"), file=f)

    for seed in all_flags.exp.seed:
        g = tf.Graph()
        logdir_seed = os.path.join(logdir, str(seed))
        with g.as_default():
            # Set seed
            np.random.seed(seed)
            tf.set_random_seed(seed)
            _train(all_flags=all_flags, logdir=logdir_seed)


if __name__ == "__main__":
    _main()
