"""Neural network learner flags."""

from learner_flags import LearnerFlags


class NeuralNetworkLearnerFlags(LearnerFlags):
    """
    Some learners (e.g. `DeterministicLearner`, `StochasticLearner`) use a
    nueral network to mimic the behavior of a controller. These flags configure
    the hyperparameters of this neural network.
    """

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Adds flags to an argparse parser."""
        if argument_group is None:
            argument_group = parser.add_argument_group('learner')
        # TODO(mwhittaker): s/con_/learner_/ after main is completely split.
        argument_group.add_argument(
            '--con_depth',
            type=int,
            default=5,
            help='learned controller NN depth',
        )
        argument_group.add_argument(
            '--con_width',
            type=int,
            default=32,
            help='learned controller NN width',
        )
        argument_group.add_argument(
            '--con_learning_rate',
            type=float,
            default=1e-3,
            help='learned controller NN learning rate',
        )
        argument_group.add_argument(
            '--con_epochs',
            type=int,
            default=100,
            help='learned controller epochs',
        )
        argument_group.add_argument(
            '--con_batch_size',
            type=int,
            default=512,
            help='learned controller batch size',
        )
        return argument_group

    @staticmethod
    def name():
        raise NotImplementedError()

    def __init__(self, args):
        self.con_depth = args.con_depth
        self.con_width = args.con_width
        self.con_learning_rate = args.con_learning_rate
        self.con_epochs = args.con_epochs
        self.con_batch_size = args.con_batch_size
