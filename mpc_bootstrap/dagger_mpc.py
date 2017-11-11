"""Dagger based bootstrapped MPC."""

import numpy as np

from controller import Controller
from mpc import MPC
from sample import sample_venv
from utils import Dataset
import logz

class DaggerMPC(Controller):
    """
    Like BootstrappedMPC, but use the learned policy to take actions
    and DAgger to learn.
    """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=None,
                 reward_fn=None,
                 num_simulated_paths=None,
                 learner=None,
                 delay=5,
                 sess=None):
        self.learner = learner
        self.mpc = MPC(
            env, dyn_model, horizon, reward_fn, num_simulated_paths, sess,
            learner)
        # TODO: the goal of this delay is to let the dynamics learn how
        # the expert behaves so that the expert gets good (since it has a
        # useful dynamics function)
        # But this isn't enough. The dynamics model can't learn
        # on learner transitions and be used by the MPC controller to plan:
        # only dynamics that are trained on controller transitions are
        # usuable. Maybe we can get rid of the delay; and split our
        # real rollouts between the MPC controller and the learner?
        # that way dynamics can be learning both transitions /
        # both transition distributions.
        self.delay = delay
        self.env = env
        # the first round should always be labelled since it's random data
        # this is a bit awkward with delay, we just use a flag.
        self.first_round = True

    def act(self, states_ns):
        if self.delay > 0:
            return self.mpc.act(states_ns)
        return self.learner.act(states_ns)

    def label(self, states_ns):
        if self.delay > 0 and not self.first_round:
            return None
        return self.mpc.act(states_ns)[0]

    def fit(self, data):
        if self.delay > 0 and not self.first_round:
            self.delay -= 1
            use_labelled = False
        else:
            self.first_round = False
            use_labelled = True

        self.learner.fit(data, use_labelled=use_labelled)

    def log(self, **kwargs):
        if self.delay > 0:
            returns = [0]
        else:
            horizon = kwargs['horizon']
            paths = sample_venv(self.env, self.mpc, horizon)
            data = Dataset(self.env, horizon)
            data.add_paths(paths)
            returns = data.rewards.sum(axis=0)
        logz.log_tabular('ExpertAverageReturn', np.mean(returns))
        logz.log_tabular('ExpertStdReturn', np.std(returns))
        logz.log_tabular('ExpertMinimumReturn', np.min(returns))
        logz.log_tabular('ExpertMaximumReturn', np.max(returns))
        self.learner.log(**kwargs)
