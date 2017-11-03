# pylint: skip-file
# TODO: above is only a stop-gap

"""Bootstrap MPC entry point."""

# Need to monkey-patch universe
# see https://github.com/openai/universe/pull/211
from universe.vectorized import MultiprocessingEnv

import shutil
import json
import time
import os
import numpy as np
import tensorflow as tf
from dynamics import NNDynamicsModel
from controllers import (
    RandomController, MPC, BootstrappedMPC, DaggerMPC, DeterministicLearner, StochasticLearner)
import cost_functions
import logz
from cheetah_env import HalfCheetahEnvNew
from utils import Path, Dataset, timeit
from gym.envs import register


def bugfix_seed_n(worker_n, seed_n):
    accumulated = 0
    for worker in worker_n:
        seed_m = seed_n[accumulated:accumulated + worker.m]
        worker.seed_start(seed_m)
        accumulated += worker.m


def bugfix_seed(self, seed):
    bugfix_seed_n(self.worker_n, seed)
    return [[seed_i] for seed_i in seed]


MultiprocessingEnv._seed = bugfix_seed


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


def train(mk_vectorized_env,
          print_time,
          cost_fn,
          tf_cost_fn,
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
          exp_name='',
          explore_std=0,
          hard_cost=False,
          deterministic=False,
          no_extra_explore=False,
          delay=0,
          ):

    locals_ = locals()
    not_picklable = [
        'mk_vectorized_env',
        'cost_fn',
        'tf_cost_fn']
    for x in not_picklable:
        del locals_[x]
    # requires additional effort to pickle
    env = mk_vectorized_env(num_paths_random)
    locals_['env_id'] = env.spec.id
    logz.configure_output_dir(logdir)
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(locals_, f)

    # Need random initial data to kickstart dynamics
    random_controller = RandomController(env)
    paths = sample(env, random_controller,
                   env_horizon, render)
    data = Dataset(env, env_horizon)
    data.add_paths(paths)
    env = mk_vectorized_env(num_paths_onpol)

    # Build dynamics model and MPC controllers.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    opt_opts = config.graph_options.optimizer_options
    opt_opts.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    dyn_model = NNDynamicsModel(env=env,
                                norm_data=data,
                                no_delta_norm=no_delta_norm,
                                batch_size=dyn_batch_size,
                                epochs=dyn_epochs,
                                learning_rate=dyn_learning_rate,
                                depth=dyn_depth,
                                width=dyn_width,
                                sess=sess)
    if agent == 'mpc':
        controller = MPC(env=env,
                         dyn_model=dyn_model,
                         horizon=mpc_horizon,
                         cost_fn=tf_cost_fn,
                         num_simulated_paths=num_simulated_paths,
                         sess=sess,
                         learner=None)
    elif agent == 'random':
        controller = random_controller
    elif agent == 'bootstrap' or agent == 'dagger':
        if deterministic:
            learner = DeterministicLearner(
                env=env,
                learning_rate=con_learning_rate,
                depth=con_depth,
                width=con_width,
                batch_size=con_batch_size,
                epochs=con_epochs,
                explore_std=explore_std,
                sess=sess)
        else:
            learner = StochasticLearner(
                env=env,
                learning_rate=con_learning_rate,
                depth=con_depth,
                width=con_width,
                batch_size=con_batch_size,
                epochs=con_epochs,
                no_extra_explore=no_extra_explore,
                sess=sess)
        if agent == 'bootstrap':
            controller = BootstrappedMPC(
                env=env,
                dyn_model=dyn_model,
                horizon=mpc_horizon,
                cost_fn=tf_cost_fn,
                num_simulated_paths=num_simulated_paths,
                learner=learner,
                sess=sess)
        else:  # dagger
            controller = DaggerMPC(
                env=env,
                dyn_model=dyn_model,
                horizon=mpc_horizon,
                delay=delay,
                cost_fn=tf_cost_fn,
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
            paths = sample(env, controller, env_horizon, render)

        if not no_aggregate:
            data.add_paths(paths)

        with timeit('gathering statistics', print_time):
            most_recent = Dataset(env, env_horizon)
            most_recent.add_paths(paths)
            returns = most_recent.rewards.sum(axis=0)

            costs = cost_functions.trajectory_cost_fn(
                cost_fn, most_recent.obs, most_recent.acs,
                most_recent.next_obs)

            mse = dyn_model.dataset_mse(most_recent)
            controller.log(horizon=env_horizon)

        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true env reward of rolled out traj using MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))
        logz.log_tabular('DynamicsMSE', mse)

        logz.dump_tabular()

    sess.__exit__(None, None, None)


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('--hard_cost', action='store_true', default=False)
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
    args = parser.parse_args()

    # Make data directory if it does not already exist
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

    # Make env
    if args.env_name == "HalfCheetah-v1":
        env = HalfCheetahEnvNew()
        if args.hard_cost:
            cost_fn = cost_functions.hard_cheetah_cost_fn
            tf_cost_fn = cost_functions.hard_tf_cheetah_cost_fn
        else:
            cost_fn = cost_functions.cheetah_cost_fn
            tf_cost_fn = cost_functions.tf_cheetah_cost_fn

    env_id = env.__class__.__name__ + '-v0'
    entry = env.__class__.__module__ + ':' + env.__class__.__name__
    register(env_id, entry_point=entry)

    def mk_vectorized_env(n):
        env = MultiprocessingEnv(env_id)
        env.configure(n=n)
        seeds = [int(s) for s in np.random.randint(0, 2 ** 30, size=n)]
        env.seed(seeds)
        return env

    for seed in args.seed:
        g = tf.Graph()
        logdir_seed = os.path.join(logdir, str(seed))
        with g.as_default():
            # Set seed
            np.random.seed(seed)
            tf.set_random_seed(seed)
            train(mk_vectorized_env=mk_vectorized_env,
                  print_time=args.time,
                  cost_fn=cost_fn,
                  tf_cost_fn=tf_cost_fn,
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
                  hard_cost=args.hard_cost,
                  deterministic=args.deterministic_learner,
                  no_extra_explore=args.no_extra_explore,
                  delay=args.delay,
                  )


if __name__ == "__main__":
    main()
