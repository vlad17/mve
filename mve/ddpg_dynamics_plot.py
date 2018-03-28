"""Generate DDPG rollouts from pretrained model. Plot H-step error."""

from contextlib import closing

import tensorflow as tf
import numpy as np

from context import flags
from ddpg_learner import DDPGLearner, DDPGFlags
from dynamics import NNDynamicsModel, DynamicsFlags
import env_info
from experiment import ExperimentFlags, setup_experiment_context
from flags import (parse_args, Flags, ArgSpec)
import tfnode
import matplotlib.pyplot as plt
from memory import Normalizer, NormalizationFlags
from persistable_dataset import PersistableDatasetFlags
from sample import sample_venv
from utils import timeit, make_session_as_default


class DDPGDynamicsPlot(Flags):
    """Flags related to DDPG dynamics plotting."""

    def __init__(self):  # pylint: disable=duplicate-code
        arguments = [
            ArgSpec(
                name='plot_nenvs',
                default=32,
                type=int,
                help='where to put the Q plot'),
            ArgSpec(
                name='plot_horizon',
                default=30,
                type=int,
                help='dynamics horizon to plot')]
        super().__init__('dyn_plot', 'dynamics error plot settings', arguments)


def _run_dyn_plot():

    N = flags().dyn_plot.plot_nenvs
    with closing(env_info.make_venv(N)) as venv:
        norm = Normalizer()
        dynamics = NNDynamicsModel(norm)

        s = tf.placeholder(tf.float32, [None, env_info.ob_dim()])
        a = tf.placeholder(tf.float32, [None, env_info.ac_dim()])
        ns = dynamics.predict_tf(s, a)

        def predict_dyn(ss, aa):
            """closure over dyn prediction"""
            return tf.get_default_session().run(
                ns, feed_dict={s: ss, a: aa})

        learner = DDPGLearner(dynamics=dynamics, normalizer=norm)
        with make_session_as_default():
            tf.global_variables_initializer().run()
            tf.get_default_graph().finalize()
            tfnode.restore_all()

            _loop(venv, learner, predict_dyn)


def _loop(venv, learner, dynamics):
    H = flags().dyn_plot.plot_horizon
    with timeit('sample learner'):
        paths = sample_venv(venv, learner.agent().exploit_act)

    N = len(paths) * 1000
    obdim = env_info.ob_dim()
    acdim = env_info.ac_dim()

    samples_initial_states = np.empty([N, obdim])
    samples_states = np.empty([H, N, obdim])
    samples_actions = np.empty([H, N, acdim])

    for k in range(N):
        i = np.random.randint(len(paths))
        path = paths[i]
        j = np.random.randint(0, path.max_horizon - H - 1)
        samples_initial_states[k] = path.obs[j]
        samples_states[:, k, :] = path.obs[j + 1: j + 1 + H]
        samples_actions[:, k, :] = path.acs[j: j + H]

    predicted_states = np.empty([H, N, obdim])
    current_state = samples_initial_states
    for h in range(H):
        next_pred_state = dynamics(current_state, samples_actions[h])
        predicted_states[h] = next_pred_state
        current_state = next_pred_state

    errors_hn = np.linalg.norm(predicted_states - samples_states, axis=2)
    means = np.mean(errors_hn, axis=1)
    stds = np.std(errors_hn, axis=1)
    low = means - stds
    high = means + stds

    fs = 20
    plt.title('H-step open loop error', fontsize=fs)
    plt.xlabel('horizon', fontsize=fs)
    plt.ylabel('dynamics $L_2$ norm error', fontsize=fs)
    plt.plot(range(1, H + 1), means, color='blue')
    plt.fill_between(range(1, H + 1), low, high, alpha=0.1)
    plt.plot(range(1, H + 1), low, color='blue', ls='--')
    plt.plot(range(1, H + 1), high, color='blue', ls='--')
    plt.savefig('dyn-mse.pdf', format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    _flags = [ExperimentFlags(), PersistableDatasetFlags(),
              DDPGFlags(), DynamicsFlags(), DDPGDynamicsPlot(),
              NormalizationFlags()]
    _args = parse_args(_flags)
    with setup_experiment_context(_args,
                                  create_logdir=False,
                                  create_reporter=False):
        _run_dyn_plot()
