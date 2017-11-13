"""DDPGLearner flags."""

from ddpg_learner import DDPGLearner
from neural_network_learner_flags import NeuralNetworkLearnerFlags


class DDPGLearnerFlags(NeuralNetworkLearnerFlags):
    """DDPGLearner flags."""

    @staticmethod
    def add_flags(parser, argument_group=None):
        if argument_group is None:
            argument_group = parser.add_argument_group('ddpg learner')

        NeuralNetworkLearnerFlags.add_flags(parser, argument_group)
        argument_group.add_argument(
            '--action_stddev',
            type=float,
            default=0.2,
            help='desired stddev for injected noise in DDPG actions')
        argument_group.add_argument(
            '--critic_l2_reg',
            type=float,
            default=1e-2,
            help='DDPG critic regularization constant'
        )
        argument_group.add_argument(
            '--critic_lr',
            type=float,
            default=1e-3,
            help='DDPG critic learning rate'
        )
        argument_group.add_argument(
            '--action_noise_exploration',
            default=0,
            type=float,
            help='use DDPG actor action noise for MPC exploration: parameter '
            'sets stddev for normal noise if >0'
        )
        argument_group.add_argument(
            '--param_noise_exploration',
            default=False,
            action='store_true',
            help='use DDPG actor param noise for MPC exploration: only if '
            'action_noise_exploration not set'
        )
        argument_group.add_argument(
            '--param_noise_exploitation',
            default=False,
            action='store_true',
            help='use DDPG actor noise when the learner acts'
        )

    @staticmethod
    def name():
        return 'ddpg'

    def __init__(self, args):
        super().__init__(args)
        self.action_stddev = args.action_stddev
        self.critic_l2_reg = args.critic_l2_reg
        self.critic_lr = args.critic_lr
        self.action_noise_exploration = args.action_noise_exploration
        self.param_noise_exploration = args.param_noise_exploration
        self.param_noise_exploitation = args.param_noise_exploitation

    def make_learner(self, venv, sess, data):
        """Make a DDPGLearner."""
        return DDPGLearner(env=venv, data=data, sess=sess, ddpg_flags=self)
