"""
A serial vectorized environment that is tied to an actor, allowing for
remote actor execution.
"""

import numpy as np
from numpy.polynomial.polynomial import polyval

from context import flags
from venv.serial_venv import SerialVenv
from utils import create_tf_session


def _discounted_rewards(rewards_hn):
    """
    Given an array of rewards collected from n trajectories over h
    steps, where the first axis is the timestep and the second
    is the number of trajectories, returns the discounted cumulative
    reward for each trajectory.
    """
    discount = flags().experiment.discount
    return polyval(discount, rewards_hn)


class ActorVenv(SerialVenv):
    """
    This is a SerialVenv with an embeded TF session,
    which can be used to evaluate remote actor neural networks.

    At construction time, the actor accepts a lambda that takes no
    arguments and generates an actor network on the default graph
    The lambda, when evaluated, should return a closure which
    maps states to actions (in a batched manner).
    """

    def __init__(self, generate_actor, m):
        super().__init__(m)
        self._actor = generate_actor()
        target = 'grpc://' + flags().experiment.tf_host()
        self._sess = create_tf_session(gpu=True, target=target)

    def multi_step_actor(self, obs_ms, num_steps):
        """
        Use the internal actor to execute num_steps steps from the given
        positions, where m <= n. Returns three one-dimensional arrays:

        * Final states
        * Resulting discounted rewards
        * Boolean array indicating whether the final state was terminal or not
        """
        obs_ms = np.asarray(obs_ms)
        m = obs_ms.shape[0]
        assert m <= self.n, (m, self.n)
        all_rews = np.zeros((num_steps, m,))
        dones = np.zeros((m,), dtype=bool)
        self.reset()
        self.set_state_from_obs(obs_ms)
        with self._sess.as_default():
            for i in range(num_steps):
                acs_ma = self._actor(obs_ms)
                for j, (env, ac) in enumerate(zip(self._envs, acs_ma)):
                    if self._mask[j]:
                        ob, rew, done, _ = env.step(ac)
                        obs_ms[j] = ob
                        all_rews[i, j] = rew
                        dones[j] |= done
                        if done:
                            self._mask[j] = False

        rewards = _discounted_rewards(all_rews)
        return obs_ms, rewards, dones

    def _close(self):
        super()._close()
        self._sess.close()
