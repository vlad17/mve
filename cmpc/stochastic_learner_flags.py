"""StochasticLearner flags."""

from stochastic_learner import StochasticLearner
from neural_network_learner_flags import NeuralNetworkLearnerFlags


class StochasticLearnerFlags(NeuralNetworkLearnerFlags):
    """StochasticLearner flags."""

    @staticmethod
    def add_flags(parser, argument_group=None):
        if argument_group is None:
            argument_group = parser.add_argument_group('stochastic learner')

        NeuralNetworkLearnerFlags.add_flags(parser, argument_group)
        argument_group.add_argument(
            '--no_extra_explore',
            action='store_true',
            help='don\'t add extra noise to the first action proposed'
            ' by stochastic learners',
        )

    @staticmethod
    def name():
        return 'gaussian'

    def __init__(self, args):
        super().__init__(args)
        self.no_extra_explore = args.no_extra_explore

    def make_learner(self, venv, sess):
        """Make a StochasticLearner."""
        return StochasticLearner(
            env=venv,
            learning_rate=self.con_learning_rate,
            depth=self.con_depth,
            width=self.con_width,
            batch_size=self.con_batch_size,
            epochs=self.con_epochs,
            no_extra_explore=self.no_extra_explore,
            sess=sess)
