"""
A multithreaded implementation of the vectorized environment.
"""

from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np

from venv.serial_venv import SerialVenv


class ThreadedVenv(SerialVenv):
    """
    A multi-threaded version of SerialVenv.
    """

    def __init__(self, n):
        super().__init__(n)
        self._tpe = ThreadPoolExecutor(max_workers=max(cpu_count(), 1))

    def _step_env(self, out_obs, out_rews, out_dones, i, env, ac):
        if self._mask[i]:
            ob, rew, done, _ = env.step(ac)
            out_obs[i] = ob
            out_rews[i] = rew
            out_dones[i] = done
            if done:
                self._mask[i] = False

    def _step(self, action):
        m = len(action)
        assert m <= len(self._envs), (m, len(self._envs))
        obs = np.empty((m,) + self.observation_space.shape)
        rews = np.empty((m,))
        dones = np.empty((m,), dtype=bool)
        infos = [{}] * m
        futs = []
        f = partial(self._step_env, obs, rews, dones)
        for i, env, ac in zip(range(m), self._envs, action):
            futs.append(self._tpe.submit(f, i, env, ac))
        for fut in as_completed(futs):
            fut.result()
        return obs, rews, dones, infos

    def _close(self):
        super()._close()
        self._tpe.shutdown()
