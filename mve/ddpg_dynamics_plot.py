"""Generate DDPG rollouts from pretrained model. Plot H-step error."""

from contextlib import closing
import distutils.util

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
# flake8: noqa pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

from context import flags
from ddpg_learner import DDPGLearner
from dynamics import NNDynamicsModel
import env_info
from experiment import setup_experiment_context
from flags import (parse_args, Flags, ArgSpec)
import tfnode
from log import debug
from main_ddpg import ALL_DDPG_FLAGS
from memory import Normalizer, Dataset
from persistable_dataset import add_dataset_to_persistance_registry
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
                name='training_timesteps',
                default=0,
                type=int,
                help='this value is multiplied by '
                'dynamics_batches_per_timestep to compute the number of '
                'batches to train the dynamics with'),
            ArgSpec(
                name='plot_horizon',
                default=30,
                type=int,
                help='dynamics horizon to plot'),
            ArgSpec(
                name='normalize',
                default=False,
                type=distutils.util.strtobool,
                help='whether to normalize the dynamics data'
            ),
            ArgSpec(
                name='outfile',
                default='dyn-mse.pdf',
                type=str,
                help='outfile name'
            ),
            ArgSpec(
                name='ndatapoints',
                default=0,
                type=int,
                help='number of datapoints to use for training dynamics; 0 if '
                'to use all of them')
        ]
        super().__init__('dyn_plot', 'dynamics error plot settings', arguments)


def _run_dyn_plot():

    N = flags().dyn_plot.plot_nenvs
    with closing(env_info.make_venv(N)) as venv:
        norm = Normalizer()
        dynamics = NNDynamicsModel(norm)

        masks = [flags().dynamics.dynamics_early_stop]
        data = Dataset(flags().experiment.bufsize, masks)
        add_dataset_to_persistance_registry(data)

        s = tf.placeholder(tf.float32, [None, env_info.ob_dim()])
        a = tf.placeholder(tf.float32, [None, env_info.ac_dim()])
        ns = dynamics.predict_tf(s, a)

        def predict_dyn(ss, aa):
            """closure over dyn prediction"""
            return tf.get_default_session().run(
                ns, feed_dict={s: ss, a: aa})

        need_dynamics = (
            flags().ddpg.dynamics_type == 'learned' or
            flags().ddpg.imaginary_buffer > 0)
        learner = DDPGLearner(dynamics=(dynamics if need_dynamics else None))
        with make_session_as_default():
            tf.global_variables_initializer().run()
            tf.get_default_graph().finalize()
            tfnode.restore_all()

            debug('loaded dataset of size {}', data.size)
            debug('env dimension {}', env_info.ob_dim())

            if flags().dyn_plot.ndatapoints > 0:
                trunc_size = flags().dyn_plot.ndatapoints
                debug('truncating dataset to size {}', trunc_size)
                datastate = data.get_state()
                for k, v in datastate.items():
                    datastate[k] = v[:trunc_size]
                data.set_state(datastate)

            if flags().dyn_plot.normalize:
                norm.update_stats(data)
            # else default statistics are 0 mean and 1 std, so they have no
            # effect
            dynamics.fit(data, flags().dyn_plot.training_timesteps)

            _loop(venv, learner, predict_dyn)


def _loop(venv, learner, dynamics):
    H = flags().dyn_plot.plot_horizon
    with timeit('sample learner'):
        paths = sample_venv(venv, learner.agent().exploit_act)

    orig_len = len(paths)
    paths = [path for path in paths if len(path.obs) >= H]
    if len(paths) < orig_len:
        debug('dropped {} of {} trajectories for {}-step error analysis '
              'due to early termination',
              orig_len - len(paths), orig_len, H)

    N = len(paths) * 1000
    obdim = env_info.ob_dim()
    acdim = env_info.ac_dim()

    samples_initial_states = np.empty([N, obdim])
    samples_states = np.empty([H, N, obdim])
    samples_actions = np.empty([H, N, acdim])

    for k in range(N):
        i = np.random.randint(len(paths))
        path = paths[i]
        j = np.random.randint(0, len(path.obs) - H - 1)
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
    low, high = np.percentile(errors_hn, (50 - 34, 50 + 34), axis=1)

    fs = 20
    plt.title('H-step open loop error', fontsize=fs)
    plt.xlabel('horizon', fontsize=fs)
    plt.ylabel('dynamics $L_2$ norm error', fontsize=fs)
    plt.plot(range(1, H + 1), means, color='blue')
    plt.fill_between(range(1, H + 1), low, high, alpha=0.1)
    plt.plot(range(1, H + 1), low, color='blue', ls='--')
    plt.plot(range(1, H + 1), high, color='blue', ls='--')
    filename = flags().dyn_plot.outfile
    plt.savefig(filename, format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    _flags = ALL_DDPG_FLAGS[:]
    _flags.append(DDPGDynamicsPlot())
    _args = parse_args(_flags)
    with setup_experiment_context(_args,
                                  create_logdir=False,
                                  create_reporter=False):
        _run_dyn_plot()
