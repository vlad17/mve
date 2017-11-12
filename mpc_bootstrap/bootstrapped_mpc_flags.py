"""Bootstrapped MPC flags."""

from bootstrapped_mpc import BootstrappedMPC
from mpc_flags import MpcFlags

class BootstrappedMpcFlags(MpcFlags):
    """Bootstrapped MPC flags."""

    @staticmethod
    def add_flags(parser, argument_group=None):
        if argument_group is None:
            argument_group = parser.add_argument_group('bootstrapped_mpc')
        MpcFlags.add_flags(parser, argument_group)

    @staticmethod
    def name():
        return "bmpc"

    # pylint: disable=signature-differs
    # pylint: disable=arguments-differ
    def make_controller(self, env, venv, sess, dyn_model, learner):
        """Make a BootstrappedMPC controller."""
        return BootstrappedMPC(
            env=venv,
            dyn_model=dyn_model,
            horizon=self.mpc_horizon,
            reward_fn=env.tf_reward,
            num_simulated_paths=self.mpc_simulated_paths,
            learner=learner,
            sess=sess)
