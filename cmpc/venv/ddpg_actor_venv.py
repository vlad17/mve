"""
A serial vectorized environment that is tied to an actor, allowing for
remote actor execution.
"""

import numpy as np

import tensorflow as tf

from context import flags
from ddpg.models import trainable_vars, Actor
import env_info
import server_registry
from venv.serial_venv import SerialVenv
from utils import discounted_rewards


def _child_scope():
    child_index = server_registry.get_child_index()
    return 'child{}'.format(child_index)


def _target_copy():
    with tf.device(server_registry.parent_device()):
        src_vars = trainable_vars('ddpg', 'target_actor')
    with tf.device(server_registry.child_device()):
        dst_vars = trainable_vars(_child_scope(), 'target_actor')
    assert len(src_vars) == len(dst_vars), (src_vars, dst_vars)
    updates = []
    for src_var, dst_var in zip(src_vars, dst_vars):
        updates.append(tf.assign(dst_var, src_var))
    return tf.group(*updates)


def _generate_actor():
    """
    Generates an actor which acts according to its target policy. To be
    called on a remote process, fetching the parent/master's actor policy.

    Returns a no-argument closure which updates the child network and
    another closure which maps states to actions using the current
    default TF session.
    """
    with tf.device(server_registry.parent_device()):
        # need to construct the parent graph
        Actor(
            width=flags().ddpg.learner_width,
            depth=flags().ddpg.learner_depth,
            scope='ddpg')
    with tf.device(server_registry.child_device()):
        child_copy = Actor(
            width=flags().ddpg.learner_width,
            depth=flags().ddpg.learner_depth,
            scope=_child_scope())
        obs_ph_ns = tf.placeholder(
            tf.float32, [None, env_info.ob_dim()])
        acs_na = child_copy.tf_target_action(obs_ph_ns)
    copy_op = _target_copy()
    return (
        lambda: tf.get_default_session().run(copy_op),
        lambda states: tf.get_default_session().run(acs_na, feed_dict={
            obs_ph_ns: states}))


class DDPGActorVenv(SerialVenv):
    """
    This is a SerialVenv with an embeded actor,
    which can be used to evaluate remote actor neural networks.

    At initialization, creates an actor on the default graph which
    copies the parent actor's parameters before executing actions.
    """

    def __init__(self, m):
        super().__init__(m)
        self._update_actor, self._actor = _generate_actor()

    def multi_step_actor(self, obs_ms, num_steps):
        """
        Use the internal actor to execute num_steps steps from the given
        positions, where m <= n. Returns three one-dimensional arrays:

        * Final states
        * Resulting discounted rewards
        * Boolean array indicating whether the final state was terminal or not
        """
        self._update_actor()
        obs_ms = np.asarray(obs_ms)
        m = obs_ms.shape[0]
        assert m <= self.n, (m, self.n)
        all_rews = np.zeros((num_steps, m,))
        dones = np.zeros((m,), dtype=bool)
        self.reset()
        self.set_state_from_obs(obs_ms)
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

        rewards = discounted_rewards(all_rews)
        return obs_ms, rewards, dones
