"""
Evaluate the various q-value estimators for DDPG.
"""
from contextlib import closing
import os

import tensorflow as tf
import numpy as np

from context import flags
from ddpg_learner import DDPGLearner, DDPGFlags
from experiment import ExperimentFlags, experiment_main
from flags import (parse_args, Flags, ArgSpec)
import tfnode
from multiprocessing_env import make_venv
from plot import plt, savefig, activate_tex
from qvalues import qvals, oracle_q, offline_oracle_q, corrected_horizon
import reporter
from sample import sample_venv
from utils import timeit, as_controller, print_table


def _mean_errorbars(vals, label, color, ci=None):
    # plots mean with label
    means = [x.mean() for x in vals]
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
    par = flags().ddpg.oracle_nenvs_with_default()
    horizons_to_test = {0, 1, mixture_horizon // 2, mixture_horizon - 1}
    horizons_to_test = sorted(list(horizons_to_test))
    with closing(make_venv(flags().experiment.make_env, par)) as evalvenv:
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
    venv = make_venv(flags().experiment.make_env, neps)
    learner = DDPGLearner()

    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()
    tfnode.restore_all()

    controller = as_controller(learner.actor.target_act)

    with timeit('running controller evaluation (target actor)'):
        paths = sample_venv(venv, controller, flags().experiment.horizon)
        qs = np.concatenate(qvals(paths, flags().experiment.discount))

    with timeit('target Q'):
        acs = np.concatenate([path.acs for path in paths])
        obs = np.concatenate([path.obs for path in paths])
        ddpg_qs = learner.critic.target_critique(obs, acs)
        current_ddpg_qs = learner.critic.critique(obs, acs)

    mixture_horizon = flags().evaluation.mixture_horizon
    with timeit('target oracle-n Q, for n = 1, ..., {}'
                .format(mixture_horizon)):
        oracle_estimators = []
        for h in range(mixture_horizon):
            oracle_estimators.append(offline_oracle_q(paths, ddpg_qs, h))
        current_oracle_estimators = []
        for h in range(mixture_horizon):
            current_oracle_estimators.append(
                offline_oracle_q(paths, current_ddpg_qs, h))

    subsample = flags().evaluation.evaluate_oracle_subsample
    if subsample > 0:
        _evaluate_oracle(learner, paths, oracle_estimators)

    neps_str = r'(${}$ episodes)'.format(neps)

    oracle_diffs = [e - qs for e in oracle_estimators]
    _mean_errorbars(oracle_diffs, 'oracle', 'blue')
    plt.xlabel(r'horizon $h$')
    plt.ylabel(r'$Q$ bias')
    plt.title(r'$\hat Q_h-Q^{\pi_{\mathrm{target}}}$ bias ' + neps_str)
    savefig(os.path.join(reporter.logging_directory(), 'bias.pdf'))

    oracle_sqerrs = [np.square(diff) for diff in oracle_diffs]
    _mean_errorbars(oracle_sqerrs, 'oracle', 'blue')
    plt.xlabel(r'horizon $h$')
    plt.ylabel(r'$Q$ MSE')
    plt.title(r'$(\hat Q_h-Q^{\pi_{\mathrm{target}}})^2$ MSE ' + neps_str)
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
                default=10,
                help='number episodes to evaluate with on')]
        super().__init__('evaluation', 'evaluation flags for ddpg', arguments)


if __name__ == "__main__":
    _flags = [ExperimentFlags(), EvaluationFlags(), DDPGFlags()]
    _args = parse_args(_flags)
    experiment_main(_args, _evaluate)
