"""Bootstrapped MPC controller."""

from mpc import MPC
from policy import Policy

class BootstrappedMPC(Policy):
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
