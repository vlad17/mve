"""Glue between DDPG and mpc_bootstrap APIs"""

import numpy as np

from log import debug
from ddpg.models import Actor, Critic
from ddpg.memory import openai_batch
from ddpg.ddpg import DDPG
from ddpg.noise import AdaptiveParamNoiseSpec


def mkagent(env, flags):
    """
    Make a DDPG agent with all the bells and whistles.
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
    critic = Critic(width=flags.con_width, depth=flags.con_depth)
    actor = Actor(nb_actions, width=flags.con_width, depth=flags.con_depth)
    ddpg = DDPG(actor, critic, env.observation_space.shape,
                env.action_space.shape, tau=0.01, **kwargs)
    return ddpg


# openai had nb_iterations at around 1/2 of total number of steps
# that have been added to the replay buffer since train() was last called.
# 500 = 1/2 (1000) = 1 HalfCheetah rollout
def train(env, agent, data, nb_iterations=500,
          param_noise_adaption_interval=50, nprints=5):
    """
    Train a DDPG agent off-policy from its referenced dataset.
    """
    max_action = env.action_space.high
    assert np.all(max_action == -env.action_space.low), \
        (env.action_space.low, env.action_space.high)
    period = max(nb_iterations // nprints, 1)
    distance = 0.

    for itr, batch in enumerate(data.sample_many(
            nb_iterations, agent.batch_size)):
        batch = openai_batch(*batch)

        # Adapt param noise, if necessary.
        if data.size >= agent.batch_size and \
           (itr + 1) % param_noise_adaption_interval == 0:
            distance = agent.adapt_param_noise(batch)

        cl, al = agent.train(batch)
        agent.update_target_net()
        if itr == 0 or itr + 1 == nb_iterations or (itr + 1) % period == 0:
            fmt = 'itr {: 6d} critic loss {:7.3f} actor loss {:7.3f} '
            fmt += 'dist {:10.3f}'
            debug(fmt, itr + 1, cl, al, distance)
