import gym
import itertools, pickle, sys, random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
import deepq
# from baselines import deepq
from deepq import EnvSim
from deepq.replay_buffer import ReplayBuffer
from deepq.utils import BatchInput
from baselines.common.schedules import LinearSchedule
import argparse #TODO: use this


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def eval(act, env, n=1):
    score = 0
    done = False
    obs = env.reset()
    for _ in range(n):
        for i in range(500):
            env_action = act(np.array(obs)[None], update_eps=0)[0]
            obs, rew, done, _ = env.step(env_action)
            score += rew
            if i > 200:
                print(done)
            if done:
                env.reset()
                break
    return score/float(n)

def run_experiment(model, horizon=0, gamma=0.99, env_name="CartPole-v0", learning_rate=5e-4,
    buffer_size=50000, train_freq=1, learning_starts=1000, max_iter=150000, batch_size=32,
    target_update_freq=1000, eval_freq=100, ema=False, double_q=True, true_dynamics=True,
    seed=None):
    with U.make_session(8):
            # Create the environment
            env = gym.make(env_name)
            testenv = gym.make(env_name)
            if seed:
                env.seed(seed)
                testenv.seed(seed)
                tf.set_random_seed(seed)
                random.seed(seed)
            sim = EnvSim(testenv)
            # Create all the functions necessary to train the model
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: BatchInput(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                sim=sim,
                horizon=horizon,
                gamma=gamma,
                ema=ema,
                double_q=double_q,
                true_dynamics=true_dynamics
            )
            # Create the replay buffer
            replay_buffer = ReplayBuffer(buffer_size)
            # Create the schedule for exploration starting from 1 (every action is random) down to
            # 0.02 (98% of actions are selected according to values predicted by the model).
            exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

            # Initialize the parameters and copy them to the target network.
            U.initialize()
            update_target()

            scores = []

            episode_rewards = [0.0]
            obs = env.reset()
            for t in itertools.count():
                # Take action and update exploration to the newest value
                action = act(obs[None], update_eps=exploration.value(t))[0]
                new_obs, rew, done, _ = env.step(action)
                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    obs = env.reset()
                    episode_rewards.append(0)

                is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
                # if is_solved:
                #     break
                #     # Show off the result
                #     env.render()
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > learning_starts:
                    for i in range(train_freq):
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                        ret = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                        # import IPython; IPython.embed()
                # Update target network periodically.
                if t % target_update_freq == 0:
                    update_target()
                if t % eval_freq == 0:
                    sc = eval(act, testenv, 3)
                    scores.append(sc)
                    print("SCORE", sc)
                    with open("cartpole-v0-" + str(horizon) + "-testing-"+ str(seed) +"-seed-5.pkl", "wb") as f:
                        pickle.dump(scores, f)

                if done and len(episode_rewards) % 10 == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("mean 10 episode reward", round(np.mean(episode_rewards[-11:-1]), 1))
                    logger.record_tabular("mean 100 episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()
                if t >= max_iter:
                    break

if __name__ == '__main__':
    seed=None
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
    run_experiment(model, horizon=int(sys.argv[1]), target_update_freq=1, ema=True, seed=seed)