import gym, sys

import deepq
from deepq import EnvSim


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main(horizon=0):
    env = gym.make("CartPole-v0")
    testenv = gym.make("CartPole-v0")
    sim = EnvSim(gym.make("CartPole-v0"))
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=None,
        horizon=horizon,
        true_dynamics=True,
        sim=sim,
        testenv=testenv,
        datafile="cartpolev0-" + str(horizon) + "-test.pkl",
        gamma=0.99,
        train_freq=1
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main(horizon=int(sys.argv[1]))
