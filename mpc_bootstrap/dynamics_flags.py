"""Dynamics flags."""

from flags import Flags
from dynamics import NNDynamicsModel


class DynamicsFlags(Flags):
    """
    We use a neural network to model an environment's dynamics. These flags
    define the architecture and learning policy of neural network.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        dynamics_nn = parser.add_argument_group('dynamics')
        dynamics_nn.add_argument(
            '--dyn_depth',
            type=int,
            default=2,
            help='dynamics NN depth',
        )
        dynamics_nn.add_argument(
            '--dyn_width',
            type=int,
            default=500,
            help='dynamics NN width',
        )
        dynamics_nn.add_argument(
            '--dyn_learning_rate',
            type=float,
            default=1e-3,
            help='dynamics NN learning rate',
        )
        dynamics_nn.add_argument(
            '--dyn_epochs',
            type=int,
            default=60,
            help='dynamics NN epochs',
        )
        dynamics_nn.add_argument(
            '--dyn_batch_size',
            type=int,
            default=512,
            help='dynamics NN batch size',
        )
        dynamics_nn.add_argument(
            '--dyn_no_delta_norm',
            action='store_true',
            default=False
        )

    @staticmethod
    def name():
        return 'dynamics'

    def __init__(self, args):
        self.dyn_depth = args.dyn_depth
        self.dyn_width = args.dyn_width
        self.dyn_learning_rate = args.dyn_learning_rate
        self.dyn_epochs = args.dyn_epochs
        self.dyn_batch_size = args.dyn_batch_size
        self.dyn_no_delta_norm = args.dyn_no_delta_norm

    def make_dynamics(self, venv, sess, norm_data):
        """Construct a NNDynamicsModel."""
        return NNDynamicsModel(
            env=venv,
            sess=sess,
            norm_data=norm_data,
            depth=self.dyn_depth,
            width=self.dyn_width,
            learning_rate=self.dyn_learning_rate,
            epochs=self.dyn_epochs,
            batch_size=self.dyn_batch_size,
            no_delta_norm=self.dyn_no_delta_norm)
