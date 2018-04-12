import gym
import numpy as np

class EnvSim(object):

    def __init__(self, env, dynamics=None):
        self.env = env
        self.env.reset()
        self.action_space = env.action_space.n
        self.dynamics = dynamics

    def simulate(self, state, action):
        data = []
        for i in range(state.shape[0]):
            self.env.reset()
            self.env.env.state[:] = np.ravel(state[i])
            action = np.ravel(action)
            next_state, reward, done, info = self.env.step(action[i])
            data.append(list(next_state) + [reward, done])
        return np.array(data).astype(np.float32)
