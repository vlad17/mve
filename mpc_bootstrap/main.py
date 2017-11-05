"""Bootstrap MPC entry point."""

import shutil
import json
import time
import os
import numpy as np
import tensorflow as tf
from multiprocessing_env import MultiprocessingEnv
from dynamics import NNDynamicsModel
from controllers import (
    RandomController, MPC, BootstrappedMPC, DaggerMPC,
    DeterministicLearner, StochasticLearner)
from envs import WhiteBoxHalfCheetahEasy, WhiteBoxHalfCheetahHard
import logz
from utils import Path, Dataset, timeit



def sample(env,
           controller,
           horizon=1000,
           render=False):
    """
    Assumes env is vectorized to some number of states at once
    """
    if render:
        raise ValueError('rendering not supported')

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


def _train(env_name='',
           print_time=False,
           frame_skip=1,
           logdir=None,
           render=False,
           dyn_learning_rate=1e-3,
           con_learning_rate=1e-4,
           onpol_iters=10,
           dyn_epochs=1,
           con_epochs=1,
           dyn_batch_size=1,
           con_batch_size=1,
           num_paths_random=10,
           num_paths_onpol=10,
           num_simulated_paths=10000,
           env_horizon=1000,
           mpc_horizon=15,
           dyn_depth=0,
           dyn_width=0,
           con_depth=0,
           con_width=0,
           no_aggregate=False,
           agent='mpc',
           no_delta_norm=False,
           exp_name='', # pylint: disable=unused-argument
           explore_std=0,
           deterministic=False,
           no_extra_explore=False,
           delay=0,
          ):

    locals_ = locals()
    logz.configure_output_dir(logdir)
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(locals_, f)

    def mk_env():
        """Generates an unvectorized env."""
        if env_name == 'hc-hard':
            return WhiteBoxHalfCheetahHard(frame_skip)
        elif env_name == 'hc-easy':
            return WhiteBoxHalfCheetahEasy(frame_skip)
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
    venv = mk_vectorized_env(num_paths_random)
    random_controller = RandomController(venv)
    paths = sample(venv, random_controller,
                   env_horizon, render)
    data = Dataset(venv, env_horizon)
    data.add_paths(paths)
    venv = mk_vectorized_env(num_paths_onpol)

    original_env = mk_env()

    # Build dynamics model and MPC controllers.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    opt_opts = config.graph_options.optimizer_options
    opt_opts.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    dyn_model = NNDynamicsModel(env=venv,
                                norm_data=data,
                                no_delta_norm=no_delta_norm,
                                batch_size=dyn_batch_size,
                                epochs=dyn_epochs,
                                learning_rate=dyn_learning_rate,
                                depth=dyn_depth,
                                width=dyn_width,
                                sess=sess)
    if agent == 'mpc':
        controller = MPC(env=venv,
                         dyn_model=dyn_model,
                         horizon=mpc_horizon,
                         reward_fn=original_env.tf_reward,
                         num_simulated_paths=num_simulated_paths,
                         sess=sess,
                         learner=None)
    elif agent == 'random':
        controller = random_controller
    elif agent == 'bootstrap' or agent == 'dagger':
        if deterministic:
            learner = DeterministicLearner(
                env=venv,
                learning_rate=con_learning_rate,
                depth=con_depth,
                width=con_width,
                batch_size=con_batch_size,
                epochs=con_epochs,
                explore_std=explore_std,
                sess=sess)
        else:
            learner = StochasticLearner(
                env=venv,
                learning_rate=con_learning_rate,
                depth=con_depth,
                width=con_width,
                batch_size=con_batch_size,
                epochs=con_epochs,
                no_extra_explore=no_extra_explore,
                sess=sess)
        if agent == 'bootstrap':
            controller = BootstrappedMPC(
                env=venv,
                dyn_model=dyn_model,
                horizon=mpc_horizon,
                reward_fn=original_env.tf_reward,
                num_simulated_paths=num_simulated_paths,
                learner=learner,
                sess=sess)
        else:  # dagger
            controller = DaggerMPC(
                env=venv,
                dyn_model=dyn_model,
                horizon=mpc_horizon,
                delay=delay,
                reward_fn=original_env.tf_reward,
                num_simulated_paths=num_simulated_paths,
                learner=learner,
                sess=sess)
    else:
        raise ValueError('agent type {} unrecognized'.format(agent))

    sess.__enter__()
    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()

    for itr in range(onpol_iters):
        with timeit('labelling actions', print_time):
            to_label = data.unlabelled_obs()
            labels = controller.label(to_label)
            data.label_obs(labels)

        with timeit('dynamics fit', print_time):
            dyn_model.fit(data)

        with timeit('controller fit', print_time):
            controller.fit(data)

        with timeit('sample controller', print_time):
            paths = sample(venv, controller, env_horizon, render)

        if not no_aggregate:
            data.add_paths(paths)

        with timeit('gathering statistics', print_time):
            most_recent = Dataset(venv, env_horizon)
            most_recent.add_paths(paths)
            returns = most_recent.rewards.sum(axis=0)
            mse = dyn_model.dataset_mse(most_recent)
            controller.log(horizon=env_horizon)

        logz.log_tabular('Iteration', itr)
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))
        logz.log_tabular('DynamicsMSE', mse)
        logz.dump_tabular()

    sess.__exit__(None, None, None)


def _main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hc-hard')
    parser.add_argument('--time', action='store_true', default=False)
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=[3], nargs='+')
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--dyn_learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--con_learning_rate', type=float, default=1e-4)
    parser.add_argument('--onpol_iters', '-n', type=int, default=1)
    parser.add_argument('--dyn_epochs', '-nd', type=int, default=60)
    parser.add_argument('--con_epochs', type=int, default=60)
    parser.add_argument('--dyn_batch_size', '-b', type=int, default=512)
    parser.add_argument('--con_batch_size', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--dyn_depth', type=int, default=2)
    parser.add_argument('--dyn_width', type=int, default=500)
    parser.add_argument('--con_depth', type=int, default=1)
    parser.add_argument('--con_width', type=int, default=32)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    parser.add_argument('--explore_std', type=float, default=0.0)
    parser.add_argument('--no_extra_explore', action='store_true',
                        default=False)
    parser.add_argument('--deterministic_learner', action='store_true',
                        default=False)
    # delay for dagger
    parser.add_argument('--delay', type=int, default=0)
    # For comparisons
    parser.add_argument('--no_aggregate', action='store_true', default=False)
    parser.add_argument('--agent', type=str, default='mpc')
    parser.add_argument('--no_delta_norm', action='store_true', default=False)
    # env
    parser.add_argument('--frame_skip', type=int, default=1)
    args = parser.parse_args()

    # Make data directory if it does not already exist
    # TODO: this procedure should be in utils
    if not os.path.exists('data'):
        os.makedirs('data')
    logdir_base = args.exp_name + '_' + args.env_name
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

    for seed in args.seed:
        g = tf.Graph()
        logdir_seed = os.path.join(logdir, str(seed))
        with g.as_default():
            # Set seed
            np.random.seed(seed)
            tf.set_random_seed(seed)
            _train(env_name=args.env_name,
                   print_time=args.time,
                   frame_skip=args.frame_skip,
                   logdir=logdir_seed,
                   render=args.render,
                   dyn_learning_rate=args.dyn_learning_rate,
                   con_learning_rate=args.con_learning_rate,
                   onpol_iters=args.onpol_iters,
                   dyn_epochs=args.dyn_epochs,
                   con_epochs=args.con_epochs,
                   dyn_batch_size=args.dyn_batch_size,
                   con_batch_size=args.con_batch_size,
                   num_paths_random=args.random_paths,
                   num_paths_onpol=args.onpol_paths,
                   num_simulated_paths=args.simulated_paths,
                   env_horizon=args.ep_len,
                   mpc_horizon=args.mpc_horizon,
                   dyn_depth=args.dyn_depth,
                   dyn_width=args.dyn_width,
                   con_depth=args.con_depth,
                   con_width=args.con_width,
                   no_aggregate=args.no_aggregate,
                   agent=args.agent,
                   no_delta_norm=args.no_delta_norm,
                   exp_name=args.exp_name,
                   explore_std=args.explore_std,
                   deterministic=args.deterministic_learner,
                   no_extra_explore=args.no_extra_explore,
                   delay=args.delay,
                  )


if __name__ == "__main__":
    _main()
