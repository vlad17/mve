"""
Evaluate the various q-value estimators for DDPG.
"""
from contextlib import closing
import multiprocessing as mp
import os

import tensorflow as tf
import numpy as np

from context import flags
from dataset import Dataset
from ddpg_learner import DDPGLearner, DDPGFlags
import env_info
from experiment import ExperimentFlags, experiment_main
from flags import (parse_args, Flags, ArgSpec)
import tfnode
from plot import plt, savefig, activate_tex
from qvalues import qvals, oracle_q, offline_oracle_q, corrected_horizon
from persistable_dataset import (
    add_dataset_to_persistance_registry, PersistableDatasetFlags)
import reporter
from sample import sample_venv
from utils import (timeit, as_controller, print_table, make_session_as_default,
                   discounted_rewards, rate_limit)


def _mean_errorbars(vals, label, color, logy=False, ci=None):
    # plots mean with label
    means = [x.mean() for x in vals]
    if logy:
        plt.semilogy(means, label=label, color=color)
        assert ci is None, 'ci {} != None incompat w/ logy'.format(ci)
    else:
        plt.plot(means, label=label, color=color)
    # plots error bars (a ci% CI) on a graph, with an error bar at each
    # index vals[i] (each vals[i] should itself be a list of values)
    if ci is None:
        return
    assert ci >= 0 and ci <= 100, ci
    lo, hi = 50 - (ci / 2), 50 + (ci / 2)
    bots, tops = zip(*[np.percentile(x, [lo, hi]) for x in vals])
    ixs = range(len(vals))
    plt.fill_between(ixs, bots, tops, alpha=0.1,
                     label=r'${}\%$ '.format(ci) + label, color=color)


def _evaluate_oracle(learner, paths, offline_oracle_estimators):
    acs = np.concatenate([path.acs for path in paths])
    obs = np.concatenate([path.obs for path in paths])
    qs = np.concatenate(qvals(paths, flags().experiment.discount))
    subsample = flags().evaluation.evaluate_oracle_subsample
    samples = int(min(max(len(obs) * subsample, 1), len(obs)))
    mixture_horizon = flags().evaluation.mixture_horizon

    print('running diagnostics to verify estimated oracle')
    data = [['h', 'offline oracle mse', 'online oracle mse']]
    par = mp.cpu_count() * 2
    horizons_to_test = {0, 1, mixture_horizon // 2, mixture_horizon - 1}
    horizons_to_test = sorted(list(horizons_to_test))
    with closing(env_info.make_venv(par)) as evalvenv:
        evalvenv.reset()
        for h in horizons_to_test:
            with timeit('  oracle sim iter ' + str(h)):
                h_n = corrected_horizon(paths, h)
                ix = np.random.choice(
                    len(obs), size=samples, replace=False)
                est = oracle_q(
                    learner.critic.target_critique,
                    learner.actor.target_act,
                    obs[ix], acs[ix], evalvenv, h_n[ix])
            online_mse = np.square(est - qs[ix]).mean()
            int_fmt = '{: ' + str(len(str(max(horizons_to_test)))) + 'd}'
            float_fmt = '{:.4g}'
            offline_mse = np.square(offline_oracle_estimators[h] - qs).mean()
            data.append([int_fmt.format(h),
                         float_fmt.format(offline_mse),
                         float_fmt.format(online_mse)])
    print_table(data)


def _evaluate(_):
    if not flags().evaluation.notex:
        activate_tex()

    neps = flags().evaluation.episodes
    venv = env_info.make_venv(neps)
    learner = DDPGLearner()

    with make_session_as_default():
        tf.global_variables_initializer().run()
        tfnode.restore_all()
        controller = as_controller(learner.actor.target_act)
        _evaluate_with_session(neps, venv, learner, controller)


def _sample_venv_fn(venv, controller):
    def _sample(obs):
        paths = sample_venv(venv, controller, obs)
        rewards_nh = np.concatenate([path.rewards for path in paths])
        return [discounted_rewards(rewards_nh.T)]
    return lambda obs: rate_limit(venv.n, _sample, obs)[0]


def _evaluate_with_session(neps, venv, learner, controller):
    data = Dataset.from_env(venv, flags().experiment.horizon,
                            flags().experiment.bufsize)
    add_dataset_to_persistance_registry(data, flags().persistable_dataset)
    tfnode.restore_all()

    with timeit('running controller evaluation (target actor)'):
        paths = sample_venv(venv, controller)

    nsamp = (2 ** 10,)

    with timeit('target Q'):
        obs = np.concatenate([path.obs for path in paths])
        ixs = np.random.randint(len(obs), size=nsamp)
        obs = obs[ixs]

    mixture_horizon = flags().evaluation.mixture_horizon
    sample_fn = _sample_venv_fn(venv, controller)
    with timeit('onpol oracle-n Q, for n = 1, ..., {}'
                .format(mixture_horizon)):
        onpol_qs = np.zeros(len(obs))
        for i in range(10):
            lo = i * len(obs) // 10
            hi = (i + 1) * len(obs) // 10
            with timeit('{:5d} of {:5d} samples'.format(hi, len(obs))):
                onpol_qs[lo:hi] = sample_fn(obs[lo:hi])
        onpol_estimators = []
        for h in range(mixture_horizon):
            print('  onpol oracle', h)
            onpol_estimators.append(oracle_q(
                learner.critic.target_critique, learner.actor.target_act,
                obs, learner.actor.target_act(obs),
                venv, np.full(len(obs), h, dtype=int)))

    with timeit('offpol eval'):
        ixs = np.random.randint(data.size, size=nsamp)
        onpol_obs = obs
        obs = data.obs[ixs]
        offpol_qs = np.zeros(len(obs))
        for i in range(10):
            lo = i * len(obs) // 10
            hi = (i + 1) * len(obs) // 10
            with timeit('{:5d} of {:5d} samples'.format(hi, len(obs))):
                offpol_qs[lo:hi] = sample_fn(obs[lo:hi])
        offpol_estimators = []
        for h in range(mixture_horizon):
            print('  offpol oracle', h)
            offpol_estimators.append(oracle_q(
                learner.critic.target_critique, learner.actor.target_act,
                obs, learner.actor.target_act(obs),
                venv, np.full(len(obs), h, dtype=int)))

    subsample = flags().evaluation.evaluate_oracle_subsample
    if subsample > 0:
        _evaluate_oracle(learner, paths, oracle_estimators)

    info_str = r'(${}$ samples, $\gamma={}$)'.format(
        len(obs), flags().experiment.discount)
    oracle_sqerrs = [np.square(e - onpol_qs) for e in onpol_estimators]
    _mean_errorbars(oracle_sqerrs, 'oracle onpol', 'blue', logy=True)
    offpol_sqerrs = [np.square(e - offpol_qs) for e in offpol_estimators]
    _mean_errorbars(offpol_sqerrs, 'oracle offpol', 'red', logy=True)
    plt.xlabel(r'horizon $h$')
    plt.ylabel(r'$Q$ MSE')
    plt.title(r'$(\hat Q_h-Q^{\pi_{\mathrm{target}}})^2$ MSE ' + info_str)
    print(os.path.join(reporter.logging_directory(), 'mse.pdf'))
    savefig(os.path.join(reporter.logging_directory(), 'mse.pdf'))


class EvaluationFlags(Flags):
    """Flags related to DDPG evaluation."""

    def __init__(self):  # pylint: disable=duplicate-code
        arguments = [
            ArgSpec(
                name='mixture_horizon',
                type=int,
                default=50,
                help='furthest horizon interval to look at'),
            ArgSpec(
                name='notex',
                default=False,
                action='store_true',
                help='disables LaTeX for plot labels'),
            ArgSpec(
                name='evaluate_oracle_subsample',
                type=float,
                default=0.,
                help='the subsampling ratio that determines the number of '
                'timesteps (samples) to use to estimate oracle, '
                'if greater than 0 then samples are taken and used '
                'to print debug information comparing the online and '
                'offline oracles'),
            ArgSpec(
                name='episodes',
                type=int,
                default=32,
                help='number episodes to evaluate with on')]
        super().__init__('evaluation', 'evaluation flags for ddpg', arguments)


if __name__ == "__main__":
    _flags = [ExperimentFlags(), EvaluationFlags(), DDPGFlags(),
              PersistableDatasetFlags()]
    _args = parse_args(_flags)
    experiment_main(_args, _evaluate)
