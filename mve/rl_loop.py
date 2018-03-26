"""
The main loop of the reinforcement learning algorithms considered
here.

Continuously loop, sampling more data, learning, and evaluating.
"""

from contextlib import closing

import tensorflow as tf

from context import flags
from dataset import Dataset
import env_info
import tfnode
from persistable_dataset import add_dataset_to_persistance_registry
from sample import Sampler, sample, sample_venv
import reporter
from utils import timeit, make_session_as_default

def rl_loop(learner, dynamics):
    """
    Assumiung that the learner and dynamics (None if no dynamics) have been
    constructed, this creates a setting for the RL agent to sample from
    and put collected data into.

    This method creates a TF session and loops continuously, training
    the agent and evaluating at intervals specified by the experiment
    flags.
    """
    with closing(env_info.make_env()) as env:
        sampler = Sampler(env)
        data = Dataset(flags().experiment.horizon, flags().experiment.bufsize)
        add_dataset_to_persistance_registry(data, flags().persistable_dataset)
        with make_session_as_default():
            tf.global_variables_initializer().run()
            tf.get_default_graph().finalize()
            tfnode.restore_all()

            # TODO --learner_evaluation_envs flag here
            # TODO then also change --evaluation_envs in dynamics_metrics
            # into a --dynamics_evaluation_envs
            # TODO closing dynamics_metrics here, change its evaluation_envs
            # flag to dynamic
            with closing(env_info.make_venv(16)) as venv:
                _loop(sampler, data, learner, dynamics, venv)

def _loop(sampler, data, learner, dynamics, eval_venv):
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

        reporter.report()
