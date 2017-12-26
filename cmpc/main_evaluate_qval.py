"""
Evaluate the various q-value estimators for DDPG.
"""
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
import reporter
from sample import sample_venv
from utils import timeit, qvals, as_controller, hstep


def _evaluate(_):
    neps = flags().evaluation.episodes
    venv = make_venv(flags().experiment.make_env, neps)
    learner = DDPGLearner()

    tf.global_variables_initializer().run()
    tf.get_default_graph().finalize()
    tfnode.restore_all()

    controller = as_controller(learner.mean_policy_act)

    with timeit('running controller evaluation'):
        paths = sample_venv(venv, controller, flags().experiment.horizon)

    acs = np.concatenate([path.acs for path in paths])
    obs = np.concatenate([path.obs for path in paths])
    qs = np.concatenate(qvals(paths, flags().experiment.discount))
    ddpg_qs = learner.critique(obs, acs)
    pathlens = np.cumsum([len(path.obs) for path in paths])[:-1]
    per_path_ddpg_qs = np.split(ddpg_qs, pathlens)

    qh_estimators = [ddpg_qs]
    discount = flags().experiment.discount
    for h in range(1, flags().evaluation.mixture_horizon):
        hsteps = hstep(paths, flags().experiment.discount, h)
        hsteps = np.concatenate(hsteps)
        per_path_hqs = [np.roll(q, -h) for q in per_path_ddpg_qs]
        for q in per_path_hqs:
            q[-h:] = 0
        hqs = np.concatenate(per_path_hqs)
        qh_estimators.append(hsteps + discount ** h * hqs)

    diffs = [est - qs for est in qh_estimators]
    biases = [x.mean() for x in diffs]
    bots0, bots1, tops1, tops0 = zip(
        *[np.percentile(x, [2.5, 16, 84, 97.5]) for x in diffs])
    if not flags().evaluation.notex:
        activate_tex()
    plt.plot(biases, label=r'$\hat Q_h$, true dynamics', color='blue')
    plt.fill_between(range(len(biases)), bots0, bots1, alpha=0.05,
                     label=r'$95\%$', color='blue')
    plt.fill_between(range(len(biases)), bots1, tops1, alpha=0.1,
                     label=r'$68\%$', color='blue')
    plt.fill_between(range(len(biases)), tops1,
                     tops0, alpha=0.05, color='blue')
    plt.xlabel(r'horizon $h$')
    plt.ylabel(r'$Q$ bias')
    plt.title(r'$Q$-value estimation bias (${}$) episodes'.format(neps))
    savefig(os.path.join(reporter.logging_directory(), 'bias.pdf'))

    mses = [np.square(x).mean() for x in diffs]
    plt.plot(mses, label=r'$\hat Q_h$, true dynamics', color='blue')
    plt.xlabel(r'horizon $h$')
    plt.ylabel(r'$Q$ MSE')
    plt.title(r'$Q$-value estimation MSE (${}$ episodes)'.format(neps))
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
                name='episodes',
                type=int,
                default=100,
                help='number episodes to evaluate with on')]
        super().__init__('evaluation', 'evaluation flags for ddpg', arguments)


if __name__ == "__main__":
    _flags = [ExperimentFlags(), EvaluationFlags(), DDPGFlags()]
    _args = parse_args(_flags)
    experiment_main(_args, _evaluate)
