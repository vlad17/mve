"""Bootstrapped MPC controller."""

import numpy as np

from controller import Controller
from mpc import MPC
from sample import sample_venv
from utils import Dataset
import logz

class BootstrappedMPC(Controller):
    """
    A bootstrapping version of the MPC controller. Learn a policy from MPC
    rollout data, and use it to run the simulations after the first action.
    """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=None,
                 reward_fn=None,
                 num_simulated_paths=None,
                 learner=None,
                 sess=None):
        self.env = env
        self.learner = learner
        self.mpc = MPC(
            env, dyn_model, horizon, reward_fn, num_simulated_paths, sess,
            learner)

    def act(self, states_ns):
        return self.mpc.act(states_ns)

    def fit(self, data):
        self.learner.fit(data, use_labelled=False)

    def log(self, **kwargs):
        horizon = kwargs['horizon']
        paths = sample_venv(self.env, self.learner, horizon)
        data = Dataset(self.env, horizon)
        data.add_paths(paths)
        returns = data.rewards.sum(axis=0)
        logz.log_tabular('LearnerAverageReturn', np.mean(returns))
        logz.log_tabular('LearnerStdReturn', np.std(returns))
        logz.log_tabular('LearnerMinimumReturn', np.min(returns))
        logz.log_tabular('LearnerMaximumReturn', np.max(returns))
        self.learner.log(**kwargs)
