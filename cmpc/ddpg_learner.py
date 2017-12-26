"""
A learner which uses DDPG: an off-policy RL algorithm based on
policy-gradients.
"""

from context import flags
from flags import Flags, ArgSpec
from learner import Learner
from tfnode import TFNode
from ddpg.ddpg import DDPG
from ddpg.models import Actor, Critic


class DDPGFlags(Flags):
    """DDPG settings"""

    def __init__(self):
        arguments = [
            ArgSpec(
                name='actor_lr',
                type=float,
                default=1e-3, help='actor network learning rate'),
            ArgSpec(
                name='critic_lr',
                type=float,
                default=1e-3, help='critic network learning rate'),
            ArgSpec(
                name='critic_l2_reg',
                type=float,
                default=0.,
                help='DDPG critic regularization constant.'
                ' Default is no regularization, but OpenAI baselines'
                ' magic uses 0.01'),
            ArgSpec(
                name='learner_depth',
                type=int,
                default=2,
                help='depth for both actor and critic networks'),
            ArgSpec(
                name='learner_width',
                type=int,
                default=64,
                help='width for both actor and critic networks'),
            ArgSpec(
                name='learner_nbatches',
                default=4000,
                type=int,
                help='number of minibatches to train on per iteration'),
            ArgSpec(
                name='learner_batch_size',
                default=512,
                type=int,
                help='number of minibatches to train on per iteration'),
            ArgSpec(
                name='target_decay',
                default=0.99,
                type=float,
                help='decay rate for target network'),
            ArgSpec(
                name='explore_stddev',
                default=0.,
                type=float,
                help='goal action standard deviation for exploration'),
            ArgSpec(
                name='incremental_reports',
                default=0,
                type=int,
                help='if >0, the number of intermediate DDPG training reports'
                ' to give'),
            ArgSpec(
                name='restore_ddpg',
                default=None,
                type=str,
                help='restore ddpg from the given path')]
        super().__init__('ddpg', 'DDPG', arguments)


class DDPGLearner(Learner, TFNode):
    """
    Use a DDPG agent to learn. The learner's tf_action
    gives its mean actor policy, but when the learner acts it uses
    parameter-space exploration.
    """

    def __init__(self):
        self._batch_size = flags().ddpg.learner_batch_size
        self._reports = flags().ddpg.incremental_reports
        self._actor = Actor(
            width=flags().ddpg.learner_width,
            depth=flags().ddpg.learner_depth,
            scope='ddpg')
        self._critic = Critic(
            width=flags().ddpg.learner_width,
            depth=flags().ddpg.learner_depth,
            scope='ddpg',
            l2reg=flags().ddpg.critic_l2_reg)
        self._ddpg = DDPG(self._actor, self._critic,
                          discount=flags().experiment.discount,
                          actor_lr=flags().ddpg.actor_lr,
                          critic_lr=flags().ddpg.critic_lr,
                          decay=flags().ddpg.target_decay,
                          scope='ddpg',
                          nbatches=flags().ddpg.learner_nbatches,
                          explore_stddev=flags().ddpg.explore_stddev)
        TFNode.__init__(self, 'ddpg', flags().ddpg.restore_ddpg)

    def custom_init(self):
        self._ddpg.initialize_targets()

    def tf_action(self, states_ns):
        return self._actor.tf_action(states_ns)

    def act(self, states_ns):
        return self._actor.perturbed_act(states_ns)

    def mean_policy_act(self, states_ns):
        """Act in a way that doesn't explore, just uses what we know."""
        return self._actor.act(states_ns)

    def critique(self, states_ns, acs_na):
        """Return the critic's assessment of the given states and actions."""
        return self._critic.critique(states_ns, acs_na)

    def fit(self, data):
        self._ddpg.train(data, self._batch_size, self._reports)
