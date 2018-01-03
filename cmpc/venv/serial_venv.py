"""
A serial implementation of the vectorized environment.
"""

import numpy as np

from venv.venv_base import VenvBase


class SerialVenv(VenvBase):
    """
    See VenvBase documentation. Creates a serially-executed rank-n vectorized
    environment
    """

    def __init__(self, n, make_env):
        self._envs = [make_env() for _ in range(n)]
        env = self._envs[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.n = len(self._envs)
        self._mask = np.ones(self.n, dtype=bool)
        self._seed_uncorr(self.n)

    def set_state_from_obs(self, obs):
        for ob, env in zip(obs, self._envs):
            env.set_state_from_ob(ob)

    def _seed(self, seed=None):
        for env, env_seed in zip(self._envs, seed):
            env.seed(env_seed)
        return seed

    def _reset(self):
        self._mask[:] = True
        return [env.reset() for env in self._envs]

    def _step(self, action):
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

    def _close(self):
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
