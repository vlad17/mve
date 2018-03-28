"""
Evaluate the various q-value estimators for DDPG.
"""

import tensorflow as tf
import numpy as np

from context import flags
from ddpg_learner import DDPGLearner, DDPGFlags
from dynamics import DynamicsFlags, NNDynamicsModel
import env_info
from experiment import ExperimentFlags, setup_experiment_context
from flags import (parse_args, Flags, ArgSpec)
import tfnode
from memory import Normalizer, NormalizationFlags
from plot import plt, savefig, activate_tex
from qvalues import qvals
from sample import sample_venv
from utils import timeit, make_session_as_default
import seaborn as sns


def _evaluate():
    if not flags().evaluation.notex:
        activate_tex()

    sns.set(font_scale=2.5)

    neps = flags().evaluation.episodes
    venv = env_info.make_venv(neps)
    # TODO need_dynamics should really be a method in ddpg.py
    need_dynamics = (
        flags().ddpg.dynamics_type == 'learned' or
        flags().ddpg.imaginary_buffer > 0)
    norm = Normalizer()
    if need_dynamics:
        dynamics = NNDynamicsModel(norm)
    else:
        dynamics = None

    learner = DDPGLearner(dynamics=dynamics, normalizer=norm)

    with make_session_as_default():
        tf.global_variables_initializer().run()
        tf.get_default_graph().finalize()
        tfnode.restore_all()
        _evaluate_with_session(venv, learner, learner.agent().exploit_act)


def _evaluate_with_session(venv, learner, controller):
    with timeit('running controller evaluation (actor)'):
        paths = sample_venv(venv, controller)
        qs = np.concatenate(qvals(paths, flags().experiment.discount))

    with timeit('target Q'):
        acs = np.concatenate([path.acs for path in paths])
        obs = np.concatenate([path.obs for path in paths])
        ddpg_qs = learner.critic.critique(obs, acs)

    ix = np.argsort(qs)
    qs = qs[ix]
    ddpg_qs = ddpg_qs[ix]
    # Look at inner 80% quantiles only
    n = len(qs) // 10
    qs = qs[n:-n]
    ddpg_qs = ddpg_qs[n:-n]
    qs -= qs.mean()
    qs /= qs.std()
    ddpg_qs -= ddpg_qs.mean()
    ddpg_qs /= ddpg_qs.std()

    sns.kdeplot(qs, ddpg_qs, cmap="Blues", shade=True, shade_lowest=True)
    plt.xlabel(r'normalized returns $Q^{\pi}$')
    plt.ylabel(r'normalized estimated $\hat Q$')
    low_x, high_x = plt.xlim()
    low_y, high_y = plt.ylim()
    if flags().evaluation.lims is not None:
        low_x, high_x, low_y, high_y = flags().evaluation.lims
    low = max(low_x, low_y)
    high = min(high_x, high_y)
    plt.plot([low, high], [low, high],
             color='black', ls='--')
    plt.xlim(low_x, high_x)
    plt.ylim(low_y, high_y)
    plt.title(flags().evaluation.title)
    savefig(flags().evaluation.output_path, legend=False)


class EvaluationFlags(Flags):
    """Flags related to DDPG evaluation."""

    def __init__(self):  # pylint: disable=duplicate-code
        arguments = [
            ArgSpec(
                name='notex',
                default=False,
                action='store_true',
                help='disables LaTeX for plot labels'),
            ArgSpec(
                name='episodes',
                type=int,
                default=10,
                help='number episodes to evaluate with on'),
            ArgSpec(
                name='title',
                default='Q density plot',
                type=str,
                help='where to put the Q plot'),
            ArgSpec(
                name='lims',
                default=None,
                type=float,
                nargs=4,
                help='xlo xhi ylo yhi'),
            ArgSpec(
                name='output_path',
                required=True,
                type=str,
                help='where to put the Q plot')]
        super().__init__('evaluation', 'evaluation flags for ddpg', arguments)


if __name__ == "__main__":
    _flags = [ExperimentFlags(), EvaluationFlags(), DDPGFlags(),
              DynamicsFlags(), NormalizationFlags()]
    _args = parse_args(_flags)
    with setup_experiment_context(_args,
                                  create_logdir=False,
                                  create_reporter=False):
        _evaluate()
