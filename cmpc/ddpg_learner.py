"""
A learner which uses DDPG: an off-policy RL algorithm based on
policy-gradients.
"""

from flags import ArgSpec
from learner import Learner
from tfnode import TFNode
from ddpg.ddpg import DDPG
from ddpg.models import Actor, Critic


class DDPGLearner(Learner, TFNode):
    """
    Use a DDPG agent to learn. The learner's tf_action
    gives its mean actor policy, but when the learner acts it uses
    parameter-space exploration.
    """

    FLAGS = [
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
            default=1e-2,
            help='DDPG critic regularization constant'),
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
            default=0.05,
            type=float,
            help='goal action standard deviation for exploration'),
        ArgSpec(
            name='restore_ddpg',
            default=None,
            type=str,
            help='restore ddpg from the given path')]

    def __init__(self, env, flags):
        self._nbatches = flags.learner_nbatches
        self._batch_size = flags.learner_batch_size
        self._actor = Actor(
            env, width=flags.learner_width, depth=flags.learner_depth,
            scope='ddpg')
        self._critic = Critic(
            env, width=flags.learner_width, depth=flags.learner_depth,
            scope='ddpg', l2reg=flags.critic_l2_reg)
        self._ddpg = DDPG(env, self._actor, self._critic, flags.discount,
                          actor_lr=flags.actor_lr, critic_lr=flags.critic_lr,
                          decay=flags.target_decay,
                          explore_stddev=flags.explore_stddev)
        TFNode.__init__(self, 'ddpg', flags.restore_ddpg)

    def custom_init(self):
        self._ddpg.initialize_targets()

    def tf_action(self, states_ns):
        return self._actor.tf_action(states_ns)

    def act(self, states_ns):
        return self._actor.act(states_ns)

    def fit(self, data):
        self._ddpg.train(data, self._nbatches, self._batch_size)
