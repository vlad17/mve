"""
An environment vectorized over multiple processes. Inspired from OpenAI
multiprocessing_env, but otherwise pretty different.

Cleanly exits on the first SIGNINT.
"""

from contextlib import closing
import multiprocessing as mp
import sys

import tensorflow as tf

from context import flags, context
import env_info
import numpy as np
from venv.venv_base import VenvBase
from venv.serial_venv import SerialVenv


def _child_loop(parent_conn, conn, m, parent_flags, id_str, venv_gen):
    parent_conn.close()
    context().flags = parent_flags
    try:
        with env_info.create(), closing(venv_gen(m)) as venv, closing(conn):
            tf.get_default_graph().finalize()
            while True:
                method_name, args = conn.recv()
                if method_name == 'close':
                    return
                method = getattr(venv, method_name)
                result = method(*args)
                try:
                    conn.send((method_name, result))
                except IOError:
                    print('worker swallowing error\n', file=sys.stderr, end='')
    except KeyboardInterrupt:
        print('{} exited cleanly on SIGINT\n'.format(id_str), end='',
              file=sys.stderr)


class _Worker:
    """
    Book-keeping on the parent process side for managing a single child
    process worker (which maintains several environments).
    """

    def __init__(self, num_workers, worker_idx, num_envs, venv_generator):
        self._lo = worker_idx * num_envs // num_workers
        self._hi = (worker_idx + 1) * num_envs // num_workers
        fmt = '{: ' + str(len(str(num_workers))) + 'd}'
        self._id_str = ('worker ' + fmt + ' of ' + fmt).format(
            worker_idx, num_workers)

        ctx = mp.get_context('spawn')
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(target=_child_loop, args=(
            self._conn, child_conn, self._hi - self._lo, flags(),
            self._id_str, venv_generator))
        self._proc.start()
        child_conn.close()

    def _push(self, method_name, args, swallow_errors=False):
        try:
            self._conn.send((method_name, args))
        except IOError as e:
            if swallow_errors:
                msg = 'parent swallowing IOError {} from {}\n'.format(
                    str(e), self._id_str)
                print(msg, end='', file=sys.stderr)
            else:
                raise e

    def _pull(self, expected_name):
        method_name, result = self._conn.recv()
        assert method_name == expected_name, (method_name, expected_name)
        return result

    def set_state_from_obs(self, obs):
        """initiate remote state-setting"""
        obs = obs[self._lo:self._hi]
        self._push('set_state_from_obs', (obs,))

    def set_state_from_obs_finish(self):
        """wait until remote state-setting completes"""
        self._pull('set_state_from_obs')

    def seed(self, seeds):
        """initiate remote seeding"""
        seeds = seeds[self._lo:self._hi]
        self._push('seed', (seeds,))

    def seed_finish(self):
        """wait until remote seeding completes"""
        self._pull('seed')

    def reset(self):
        """initiate remote reset"""
        self._push('reset', tuple())

    def reset_finish(self, out_obs):
        """wait until remote initiate remote resetting completes"""
        obs = self._pull('reset')
        out_obs[self._lo:self._hi] = obs

    def step(self, action):
        """initiate remote step"""
        action = action[self._lo:self._hi]
        self._push('step', (action,))

    def step_finish(self, out_obs, out_rews, out_dones):
        """wait until remote step completes"""
        obs, rews, dones, _ = self._pull('step')
        hi = self._lo + len(obs)
        out_obs[self._lo:hi] = obs
        out_rews[self._lo:hi] = rews
        out_dones[self._lo:hi] = dones

    def multi_step(self, action_hna):
        """initiate remote multi step"""
        action_hma = action_hna[:, self._lo:self._hi]
        self._push('multi_step', (action_hma,))

    def multi_step_finish(self, out_obs, out_rews, out_dones):
        """wait until remote multi step completes"""
        obs, rews, dones = self._pull('multi_step')
        hi = self._lo + obs.shape[1]
        out_obs[:, self._lo:hi] = obs
        out_rews[:, self._lo:hi] = rews
        out_dones[:, self._lo:hi] = dones

    def multi_step_actor(self, obs_ns, num_steps):
        """initiate remote actor-constrained multi step"""
        obs_ms = obs_ns[self._lo:self._hi]
        self._push('multi_step_actor', (obs_ms, num_steps))

    def multi_step_actor_finish(self, out_obs, out_rews, out_dones):
        """wait until remote actor-constrained multi step completes"""
        obs, rews, dones = self._pull('multi_step_actor')
        hi = self._lo + obs.shape[0]
        out_obs[self._lo:hi] = obs
        out_rews[self._lo:hi] = rews
        out_dones[self._lo:hi] = dones

    def close(self):
        """initiate remote close"""
        # racy if, so we swallow errors.
        if self._proc.is_alive():
            self._push('close', tuple(), swallow_errors=True)

    def close_finish(self):
        """join remote worker process"""
        self._conn.close()
        self._proc.join()


class ParallelVenv(VenvBase):
    """
    Multiprocessing-based parallel vectorized environment. On each
    child process, uses venv_generator(m) as the vectorized child
    environment.
    """

    def __init__(self, n, venv_generator=SerialVenv):
        try:
            num_workers = max(min(mp.cpu_count(), n), 1)
        except NotImplementedError:
            num_workers = 1
        self._workers = [_Worker(num_workers, i, n, venv_generator)
                         for i in range(num_workers)]
        with closing(env_info.make_env()) as env:
            self.action_space = env.action_space
            self.observation_space = env.observation_space
            self.reward_range = env.reward_range
        self.n = n
        self._seed_uncorr(n)

    def set_state_from_obs(self, obs):
        for worker in self._workers:
            worker.set_state_from_obs(obs)
        for worker in self._workers:
            worker.set_state_from_obs_finish()

    def _seed(self, seed=None):
        for worker in self._workers:
            worker.seed(seed)
        for worker in self._workers:
            worker.seed_finish()

    def _reset(self):
        out_obs = np.empty((self.n,) + self.observation_space.shape)
        for worker in self._workers:
            worker.reset()
        for worker in self._workers:
            worker.reset_finish(out_obs)
        return out_obs

    def _step(self, action):
        m = len(action)
        assert m <= self.n, m
        obs = np.empty((m,) + self.observation_space.shape)
        rews = np.empty((m,))
        dones = np.empty((m,), dtype=bool)
        infos = [{}] * m
        for worker in self._workers:
            worker.step(action)
        for worker in self._workers:
            worker.step_finish(obs, rews, dones)
        return obs, rews, dones, infos

    def _close(self):
        for worker in self._workers:
            worker.close()
        for worker in self._workers:
            worker.close_finish()

    def multi_step(self, acs_hna):
        h, m = acs_hna.shape[:2]
        assert m <= self.n, (m, self.n)
        obs = np.empty((h, m,) + self.observation_space.shape)
        rews = np.empty((h, m,))
        dones = np.empty((h, m,), dtype=bool)
        for worker in self._workers:
            worker.multi_step(acs_hna)
        for worker in self._workers:
            worker.multi_step_finish(obs, rews, dones)
        return obs, rews, dones

    def multi_step_actor(self, obs_ns, num_steps):
        """See documentaiton in ActorVenv.multi_step_actor"""
        m = len(obs_ns)
        assert m <= self.n, (m, self.n)
        obs = np.empty((m,) + self.observation_space.shape)
        rews = np.empty((m,))
        dones = np.empty((m,), dtype=bool)
        for worker in self._workers:
            worker.multi_step_actor(obs_ns, num_steps)
        for worker in self._workers:
            worker.multi_step_actor_finish(obs, rews, dones)
        return obs, rews, dones
