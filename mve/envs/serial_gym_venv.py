"""
A serial implementation of vectorized (old) Gym environments. Note that
MuJoCo-1.50-based environments implemented directly in gym2 should NOT
use this class, and instead rely on VectorMJCEnv.
"""

import numpy as np

from .vector_env import VectorEnv


class SerialGymVenv(VectorEnv):
    """
    See VectorEnv documentation. Creates a serially-executed rank-n vectorized
    environment meant for gym envs.
    """

    def __init__(self, n, scalar_env_gen):
        self._envs = [scalar_env_gen() for _ in range(n)]
        env = self._envs[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.n = len(self._envs)
        self._mask = np.ones(self.n, dtype=bool)
        self._seed_uncorr(self.n)

    def set_state_from_ob(self, obs):
        for ob, env in zip(obs, self._envs):
            env.set_state_from_ob(ob)

    def seed(self, seed=None):
        for env, env_seed in zip(self._envs, seed):
            env.seed(env_seed)
        return seed

    def reset(self):
        self._mask[:] = True
        return np.asarray([env.reset() for env in self._envs])

    def step(self, action):
        m = len(action)
        assert m <= len(self._envs), (m, len(self._envs))
        obs = np.empty((m,) + self.observation_space.shape)
        rews = np.empty((m,))
        dones = np.empty((m,), dtype=bool)
        infos = [{}] * m
        for i, (env, ac) in enumerate(zip(self._envs, action)):
            if self._mask[i]:
                ob, rew, done, _ = env.step(ac)
                obs[i] = ob
                rews[i] = rew
                dones[i] = done
                if done:
                    self._mask[i] = False
        return obs, rews, dones, infos

    def close(self):
        for env in self._envs:
            env.close()

    def multi_step(self, acs_hna):
        h, m = acs_hna.shape[:2]
        assert m <= self.n, (m, self.n)
        obs = np.empty((h, m,) + self.observation_space.shape)
        rews = np.empty((h, m,))
        dones = np.empty((h, m,), dtype=bool)
        for i in range(h):
            for j, (env, ac) in enumerate(zip(self._envs, acs_hna[i])):
                if self._mask[j]:
                    ob, rew, done, _ = env.step(ac)
                    obs[i, j] = ob
                    rews[i, j] = rew
                    dones[i, j] = done
                    if done:
                        self._mask[j] = False
        return obs, rews, dones
