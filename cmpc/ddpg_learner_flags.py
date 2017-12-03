"""DDPGLearner flags."""

from ddpg_learner import DDPGLearner
from learner_flags import LearnerFlags


# pylint: disable=too-many-instance-attributes
class DDPGLearnerFlags(LearnerFlags):
    """DDPGLearner flags."""

    @staticmethod
    def add_flags(parser, argument_group=None):
        """Add flags for DDPG Learner configuration to the parser"""
        if argument_group is None:
            argument_group = parser.add_argument_group('ddpg learner')

        argument_group.add_argument(
            '--learner_depth',
            type=int,
            default=2,
            help='depth for both actor and critic networks',
        )
        argument_group.add_argument(
            '--learner_width',
            type=int,
            default=64,
            help='width for both actor and critic networks',
        )
        argument_group.add_argument(
            '--actor_lr',
            type=float,
            default=1e-4,
            help='actor network learning rate')
        argument_group.add_argument(
            '--critic_lr',
            type=float,
            default=1e-4,
            help='critic network learning rate')
        argument_group.add_argument(
            '--critic_l2_reg',
            type=float,
            default=1e-2,
            help='DDPG critic regularization constant'
        )
        argument_group.add_argument(
            '--learner_nbatches',
            default=None,
            type=int,
            help='number of minibatches to train on per iteration'
        )
        argument_group.add_argument(
            '--learner_batch_size',
            default=512,
            type=int,
            help='number of minibatches to train on per iteration'
        )

    @staticmethod
    def name():
        return 'ddpg'

    def __init__(self, args):
        self.depth = args.learner_depth
        self.width = args.learner_width
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.critic_l2_reg = args.critic_l2_reg
        self.nbatches = args.learner_nbatches
        self.batch_size = args.learner_batch_size

    def make_learner(self, env):
        """Make a DDPGLearner."""
        return DDPGLearner(env, self)
