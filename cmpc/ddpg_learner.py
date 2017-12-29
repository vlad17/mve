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
                name='learner_batches_per_timestep',
                default=4,
                type=float,
                help='number of mini-batches to train dynamics per '
                'new sample observed'),
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
                help='goal action standard deviation for exploration,'
                ' after rescaling actions to [0, 1]'),
            ArgSpec(
                name='param_noise_adaption_interval',
                default=50,
                type=int,
                help='how frequently to update parameter-space noise'),
            ArgSpec(
                name='oracle_nenvs',
                default=0,
                type=int,
                help='number of envs to use for oracle Q-estimates, should be '
                'the learner_batch_size for full parallelism (automatically'
                ' set to this if 0)'),
            ArgSpec(
                name='mixture_estimator',
                default=None,
                type=str,
                help='use a mixture estimator for computing target Q '
                'values. If None, do not mix with model-based estimates '
                'at all. If oracle, use an oracle environment to compute '
                'model_horizon-step rewards for mixing with the target '
                'Q network'),
            ArgSpec(
                name='model_horizon',
                default=1,
                type=int,
                help='how many steps to expand Q estimates dynamics'),
            ArgSpec(
                name='restore_ddpg',
                default=None,
                type=str,
                help='restore ddpg from the given path'),
            ArgSpec(
                name='ddpg_min_buf_size',
                default=500,
                type=int,
                help='Minimum number of frames in replay buffer before '
                     'training')
        ]
        super().__init__('ddpg', 'DDPG', arguments)

    def oracle_nenvs_with_default(self):
        """The number of environments an oracle should use."""
        par = self.oracle_nenvs
        par = self.learner_batch_size if par == 0 else par
        return par

    def nbatches(self, timesteps):
        """The number training batches, given this many timesteps."""
        nbatches = self.learner_batches_per_timestep * timesteps
        nbatches = max(int(nbatches), 1)
        return nbatches


class DDPGLearner(Learner, TFNode):
    """
    Use a DDPG agent to learn. The learner's tf_action
    gives its mean actor policy, but when the learner acts it uses
    parameter-space exploration.

    Has actor and critic attributes for internal ddpg.models.Actor
    and ddpg.models.Critic, respectively.
    """

    def __init__(self):
        self._batch_size = flags().ddpg.learner_batch_size
        self.actor = Actor(
            width=flags().ddpg.learner_width,
            depth=flags().ddpg.learner_depth,
            scope='ddpg')
        self.critic = Critic(
            width=flags().ddpg.learner_width,
            depth=flags().ddpg.learner_depth,
            scope='ddpg',
            l2reg=flags().ddpg.critic_l2_reg)
        self._ddpg = DDPG(self.actor, self.critic,
                          discount=flags().experiment.discount,
                          actor_lr=flags().ddpg.actor_lr,
                          critic_lr=flags().ddpg.critic_lr,
                          decay=flags().ddpg.target_decay,
                          scope='ddpg',
                          explore_stddev=flags().ddpg.explore_stddev)
        TFNode.__init__(self, 'ddpg', flags().ddpg.restore_ddpg)

    def custom_init(self):
        self._ddpg.initialize_targets()

    def tf_action(self, states_ns):
        return self.actor.tf_action(states_ns)

    def act(self, states_ns):
        return self.actor.perturbed_act(states_ns)

    def fit(self, data, timesteps):
        if data.size < flags().ddpg.ddpg_min_buf_size:
            return
        nbatches = flags().ddpg.nbatches(timesteps)
        self._ddpg.train(data, nbatches, self._batch_size)
