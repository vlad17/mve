"""DeterministicLearner flags."""

from deterministic_learner import DeterministicLearner
from neural_network_learner_flags import NeuralNetworkLearnerFlags


class DeterministicLearnerFlags(NeuralNetworkLearnerFlags):
    """DeterministicLearner flags."""

    @staticmethod
    def add_flags(parser, argument_group=None):
        if argument_group is None:
            argument_group = parser.add_argument_group('determinstic learner')

        NeuralNetworkLearnerFlags.add_flags(parser, argument_group)
        argument_group.add_argument(
            '--explore_std',
            type=float,
            default=0.0,
            help='if exactly 0, explore with a uniform policy on the first '
            'simulated step; else use a Gaussian with the specified std',
        )

    @staticmethod
    def name():
        return 'delta'

    def __init__(self, args):
        super().__init__(args)
        self.explore_std = args.explore_std

    def make_learner(self, venv, sess):
        """Make a DeterministicLearner."""
        return DeterministicLearner(
            env=venv,
            learning_rate=self.con_learning_rate,
            depth=self.con_depth,
            width=self.con_width,
            batch_size=self.con_batch_size,
            epochs=self.con_epochs,
            explore_std=self.explore_std,
            sess=sess)
