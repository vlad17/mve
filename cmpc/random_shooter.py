"""
Optimize the planning problem by randomly sampling.
"""

import numpy as np
import tensorflow as tf

from controller import Controller
from dataset import one_shot_dataset
from learner import as_controller
from sample import sample
import reporter
from utils import (create_random_tf_action, get_ac_dim, get_ob_dim)

class RandomShooter(Controller):
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
        self._env = env
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

        # h = horizon
        n = tf.shape(state_ns)[0]
        # to create a matrix of actions, we need to know n, but we won't know
        # it until runtime, so we can't globally initialize the action matrix
        # instead, we initialize it to the right size at every act() call.
        self._actions_hna = tf.get_variable(
            'dynamics_actions', validate_shape=False,
            initializer=tf.zeros([mpc_horizon, n, self._ac_dim]),
            dtype=tf.float32, collections=[])

        def _body(t, state_ns, rewards):
            action_na = _policy(t, state_ns)
            save_action_op = tf.scatter_update(self._actions_hna, t, action_na)
            next_state_ns = dyn_model.predict_tf(state_ns, action_na)
            next_rewards = reward_fn(
                state_ns, action_na, next_state_ns, rewards)
            with tf.control_dependencies([save_action_op]):
                return [t + 1, next_state_ns, next_rewards]

        loop_vars = [0, state_ns, tf.zeros((n,))]
        _, _, self._final_rewards_n = tf.while_loop(
            lambda t, _, __: t < mpc_horizon, _body,
            loop_vars, back_prop=False)
        self._mpc_horizon = mpc_horizon
        self._learner = learner

    def _act(self, states):
        nstates = len(states)
        tf.get_default_session().run(
            self._actions_hna.initializer, feed_dict={
                self._input_state_ph_is: states})
        trajectory_rewards_n = tf.get_default_session().run(
            self._final_rewards_n, feed_dict={
                self._input_state_ph_is: states})
        action_hna = tf.get_default_session().run(self._actions_hna)

        # p = num simulated paths, i = nstates
        # note b/c of the way tf.tile works we need to reshape by p then i
        per_state_simulation_rewards_ip = trajectory_rewards_n.reshape(
            self._sims_per_state, nstates).T
        best_ac_ix_i = per_state_simulation_rewards_ip.argmax(axis=1)
        action_hpia = action_hna.reshape(
            self._mpc_horizon, self._sims_per_state, nstates, self._ac_dim)
        action_samples_hipa = np.swapaxes(action_hpia, 1, 2)
        best_ac_hia = action_samples_hipa[
            :, np.arange(nstates), best_ac_ix_i, :]
        best_rewards_i = per_state_simulation_rewards_ip[
            np.arange(nstates), best_ac_ix_i]

        best_ac_iha = np.swapaxes(best_ac_hia, 0, 1)
        return best_ac_hia[0], best_rewards_i, best_ac_iha

    def act(self, states_ns):
        # This batch size is specific to HalfCheetah and my setup.
        # A more appropriate version of this method should query
        # GPU memory size, and use the state dimension + MPC horizon
        # to figure out the appropriate batch amount.
        batch_size = 500
        from utils import rate_limit
        return rate_limit(batch_size, self._act, states_ns)

    def planning_horizon(self):
        return self._mpc_horizon

    def fit(self, data):
        if self._learner:
            self._learner.fit(data)

    def log(self, most_recent):
        if self._learner:
            self._learner.log(most_recent)
            # out-of-band learner evaluation
            learner = as_controller(self._learner)
            learner_path = sample(
                self._env, learner, most_recent.max_horizon)
            learner_data = one_shot_dataset([learner_path])
            learner_returns = learner_data.per_episode_rewards()
            reporter.add_summary(
                'learner reward', learner_returns[0])
