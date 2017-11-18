"""Glue between DDPG and mpc_bootstrap APIs"""

import numpy as np

from log import debug
from ddpg.models import Actor, Critic
from ddpg.memory import Memory
from ddpg.ddpg import DDPG
from ddpg.noise import AdaptiveParamNoiseSpec


def mkagent(env, dataset, flags):
    """
    Make a DDPG agent with all the bells and whistles, which learns from
    the specified dataset.
    """
    nb_actions = env.action_space.shape[-1]
    stddev = flags.action_stddev
    param_noise = AdaptiveParamNoiseSpec(
        initial_stddev=float(stddev),
        desired_action_stddev=float(stddev))

    kwargs = {'critic_l2_reg': flags.critic_l2_reg,
              'batch_size': flags.con_batch_size,
              'actor_lr': flags.con_learning_rate,
              'critic_lr': flags.critic_lr,
              'enable_popart': False,  # requires return normalization
              'gamma': 0.99,
              'clip_norm': None,
              'param_noise': param_noise,
              'action_noise': None}
    memory = Memory(dataset)
    critic = Critic(width=flags.con_width, depth=flags.con_depth)
    actor = Actor(nb_actions, width=flags.con_width, depth=flags.con_depth)
    ddpg = DDPG(actor, critic, memory, env.observation_space.shape,
                env.action_space.shape, tau=0.01, **kwargs)
    return ddpg


# openai had nb_iterations at around 1/2 of total number of steps
# that have been added to the replay buffer since train() was last called.
# 500 = 1/2 (1000) = 1 HalfCheetah rollout
def train(env, agent, nb_iterations=500, param_noise_adaption_interval=50,
          nprints=5):
    """
    Train a DDPG agent off-policy from its referenced dataset.
    """
    max_action = env.action_space.high
    assert np.all(max_action == -env.action_space.low), \
        (env.action_space.low, env.action_space.high)
    period = max(nb_iterations // nprints, 1)
    distance = 0.
    for itr in range(nb_iterations):
        # Adapt param noise, if necessary.
        if agent.memory.nb_entries >= agent.batch_size and \
           (itr + 1) % param_noise_adaption_interval == 0:
            distance = agent.adapt_param_noise()

        cl, al = agent.train()
        agent.update_target_net()
        if itr == 0 or itr + 1 == nb_iterations or (itr + 1) % period == 0:
            fmt = 'itr {: 6d} critic loss {:7.3f} actor loss {:7.3f} '
            fmt += 'dist {:10.3f}'
            debug(fmt, itr + 1, cl, al, distance)
