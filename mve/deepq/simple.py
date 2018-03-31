import gym
import itertools, pickle, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
import deepq
from deepq import EnvSim
from deepq.replay_buffer import ReplayBuffer
from deepq.utils import BatchInput
from baselines.common.schedules import LinearSchedule

def eval(act, env):
    score = 0
    done = False
    obs = env.reset()
    for i in range(500):
        env_action = act(np.array(obs)[None], update_eps=0)[0]
        obs, rew, done, _ = env.step(env_action)
        score += rew
        if i > 200:
            print(done)
        if done:
            env.reset()
            break
    return score

def run_experiment(model, env_name="CartPole-v0", buffer_size=50000, learning_starts=1000,
    target_update_freq=1000, eval_freq=100, print_freq=10, max_iter=200000, learning_rate=5e-4,
    ema=False, double_q=True, horizon=0, true_dynamics=True, batch_size=32, train_freq=1):
    with U.make_session(8):
        # Create the environment
        env = gym.make(env_name)
        testenv = gym.make(env_name)
        sim = EnvSim(testenv)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
            sim=sim,
            horizon=horizon,
            double_q=double_q,
            ema=ema,
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
            if is_solved:
                break
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > learning_starts:
                for i in range(int(np.ceil(1.0/train_freq))):
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            # Update target network periodically.
            if t % target_update_freq == 0:
                update_target()
            if t % eval_freq == 0:
                sc = eval(act, testenv)
                scores.append(sc)
                print("SCORE", sc)
                sys.stdout.flush()
                with open("cartpole-v0-" + str(horizon) + "-true-5.pkl", "wb") as f:
                    pickle.dump(scores, f)

            if done and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean 10 episode reward", round(np.mean(episode_rewards[-11:-1]), 1))
                logger.record_tabular("mean 100 episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                # if t > 100:
                #     print(episode_rewards[-101:-1], np.mean(episode_rewards[-101:-1]))
                logger.dump_tabular()
            if t >= max_iter:
                break