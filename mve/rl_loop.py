"""
The main loop of the reinforcement learning algorithms considered
here.

Continuously loop, sampling more data, learning, and evaluating.
"""

from contextlib import closing

import tensorflow as tf

from context import flags
from dataset import Dataset
from dynamics_metrics import DynamicsMetrics, TFStateUnroller
import env_info
from flags import Flags, ArgSpec
import tfnode
from persistable_dataset import add_dataset_to_persistance_registry
from sample import Sampler, sample, sample_venv
import reporter
from utils import timeit, make_session_as_default

class RLLoopFlags(Flags):
    """Specification for the dynamics metrics"""

    def __init__(self):
        args = [
            ArgSpec(
                name='dynamics_evaluation_envs',
                default=64,
                type=int,
                help='number of environments to use for evaluating dynamics'),
            ArgSpec(
                name='dynamics_evaluation_horizon',
                default=10,
                type=int,
                help='number of environments to use for evaluating dynamics'),
            ArgSpec(
                name='learner_evaluation_envs',
                default=16,
                type=int,
                help='number of environments to use for evaluating learner')]
        super().__init__('loop', 'rl loop', args)


def rl_loop(learner, dynamics):
    """
    Assumiung that the learner and dynamics (None if no dynamics) have been
    constructed, this creates a setting for the RL agent to sample from
    and put collected data into.

    This method creates a TF session and loops continuously, training
    the agent and evaluating at intervals specified by the experiment
    flags.
    """
    if dynamics:
        unroller = TFStateUnroller(
            flags().loop.dynamics_evaluation_horizon,
            learner.tf_action, dynamics.predict_tf)
    else:
        unroller = None
    with closing(env_info.make_env()) as env, \
         closing(DynamicsMetrics(
             flags().loop.dynamics_evaluation_horizon,
             flags().loop.dynamics_evaluation_envs,
             flags().experiment.discount)) as dm:
        sampler = Sampler(env)
        data = Dataset(flags().experiment.horizon, flags().experiment.bufsize)
        add_dataset_to_persistance_registry(data, flags().persistable_dataset)
        with make_session_as_default():
            tf.global_variables_initializer().run()
            tf.get_default_graph().finalize()
            tfnode.restore_all()

            learner_nenvs = flags().loop.learner_evaluation_envs
            with closing(env_info.make_venv(learner_nenvs)) as venv:
                _loop(sampler, data, learner, dynamics, venv, dm, unroller)

def _loop(sampler, data, learner, dynamics, eval_venv, dyn_metrics, unroller):
    while flags().experiment.should_continue():
        with timeit('sample learner'):
            agent = learner.agent()
            n_episodes = sampler.sample(agent.explore_act, data)
        reporter.advance(sampler.nsteps(), n_episodes)

        if dynamics:
            with timeit('dynamics fit'):
                dynamics.fit(data, sampler.nsteps())

        with timeit('learner fit'):
            learner.train(data, sampler.nsteps())

        agent = learner.agent()

        if flags().experiment.should_render():
            with flags().experiment.render_env() as render_env:
                sample(render_env, agent.exploit_act, render=True)

        if flags().experiment.should_save():
            tfnode.save_all(reporter.timestep())

        if flags().experiment.should_evaluate():
            paths = sample_venv(eval_venv, agent.exploit_act)
            rews = [path.rewards.sum() for path in paths]
            reporter.add_summary_statistics('current policy reward', rews)

            paths = sample_venv(eval_venv, agent.explore_act)
            rews = [path.rewards.sum() for path in paths]
            reporter.add_summary_statistics('exploration policy reward', rews)

            learner.evaluate(data)

            if unroller:
                nsamples = flags().loop.dynamics_evaluation_envs * 8
                planned_transitions = unroller.eval_dynamics(data, nsamples)
                dyn_metrics.evaluate(*planned_transitions)

        reporter.report()
