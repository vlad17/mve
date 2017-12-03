"""
Optimize the planning problem by randomly sampling in the support
of the constrained space.
"""

import numpy as np
import tensorflow as tf

from policy import Policy
from utils import (create_random_tf_action, get_ac_dim, get_ob_dim)


class RandomShooter(Policy):
    """
    Random-shooter based optimization of the planning objective
    looking mpc_horizon steps ahead.

    By default, with flags.opt_horizon == None or equal to mpc_horizon,
    random shooter performs
    unconstrained MPC planning. Otherwise, learner must not be
    None, and only the first flags.opt_horizon planning steps
    are optimized. The rest are taken according to the learner.
    """

    def __init__(self, env, dyn_model, reward_fn, learner, mpc_horizon, flags):
        if flags.opt_horizon is None:
            flags.opt_horizon = mpc_horizon
        if flags.opt_horizon != mpc_horizon:
            assert learner is not None, learner
            exploit = learner.tf_action
        else:
            exploit = create_random_tf_action(env.action_space)
        assert mpc_horizon > 0, mpc_horizon

        self._ac_dim = get_ac_dim(env)
        self._sims_per_state = flags.simulated_paths

        # compute the rollout in full TF to keep all computation on the GPU
        # a = action dim
        # s = state dim
        # n = batch size = num states for which MPC act * simulated rollouts
        # i = number of states in batch for act
        self._input_state_ph_is = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)])
        state_ns = tf.tile(self._input_state_ph_is, (flags.simulated_paths, 1))

        # policy shooter samples from
        def _policy(t, states_ns):
            explore = create_random_tf_action(env.action_space)
            return tf.cond(t < flags.opt_horizon,
                           lambda: explore(states_ns),
                           lambda: exploit(states_ns))

        def _body(t, state_ns, action_na, rewards):
            next_state_ns = dyn_model.predict_tf(state_ns, action_na)
            next_rewards = reward_fn(
                state_ns, action_na, next_state_ns, rewards)
            next_action_na = _policy(t + 1, next_state_ns)
            return [t + 1, next_state_ns, next_action_na, next_rewards]

        n = tf.shape(state_ns)[0]
        self._first_acs_na = _policy(tf.constant(0), state_ns)
        loop_vars = [0, state_ns, self._first_acs_na, tf.zeros((n,))]
        self._loop = tf.while_loop(
            lambda t, _, __, ___: t < mpc_horizon, _body,
            loop_vars, back_prop=False)

    def _act(self, states):
        nstates = len(states)
        loop_vars, action_na = tf.get_default_session().run(
            [self._loop, self._first_acs_na], feed_dict={
                self._input_state_ph_is: states})
        _, _, _, trajectory_rewards_n = loop_vars

        # p = num simulated paths, i = nstates
        # note b/c of the way tf.tile works we need to reshape by p then i
        per_state_simulation_rewards_ip = trajectory_rewards_n.reshape(
            self._sims_per_state, nstates).T
        best_ac_ix_i = per_state_simulation_rewards_ip.argmax(axis=1)
        action_samples_ipa = np.swapaxes(action_na.reshape(
            self._sims_per_state, nstates, self._ac_dim), 0, 1)
        best_ac_ia = action_samples_ipa[np.arange(nstates), best_ac_ix_i, :]
        best_rewards_ia = per_state_simulation_rewards_ip[
            np.arange(nstates), best_ac_ix_i]

        return best_ac_ia, best_rewards_ia

    def act(self, states_ns):
        # This batch size is specific to HalfCheetah and my setup.
        # A more appropriate version of this method should query
        # GPU memory size, and use the state dimension + MPC horizon
        # to figure out the appropriate batch amount.
        batch_size = 500
        if len(states_ns) <= batch_size:
            return self._act(states_ns)

        acs = np.empty((len(states_ns), self._ac_dim))
        rws = np.empty((len(states_ns),))
        for i in range(0, len(states_ns) - batch_size + 1, batch_size):
            loc = slice(i, i + batch_size)
            ac, rw = self._act(states_ns[loc])
            acs[loc] = ac
            rws[loc] = rw
        return acs, rws
