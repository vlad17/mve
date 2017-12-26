"""
Optimize the planning problem by randomly sampling.
"""

import numpy as np
import tensorflow as tf

from context import flags
from controller import Controller
import env_info
import reporter
from multiprocessing_env import make_venv
from sample import sample_venv
from utils import create_random_tf_action, rate_limit, as_controller


class RandomShooter(Controller):
    """
    Random-shooter based optimization of the planning objective
    looking mpc_horizon steps ahead.

    See RandomShooterFlags for details.
    """

    def __init__(self, dyn_model):
        reward_fn = env_info.reward_fn()
        opt_horizon = flags().random_shooter.opt_horizon_with_default()
        mpc_horizon = flags().mpc.mpc_horizon
        learner = flags().random_shooter.make_learner()
        exploit = learner.tf_action
        exploit = create_random_tf_action(env_info.ac_space())
        assert mpc_horizon > 0, mpc_horizon
        self._sims_per_state = flags().random_shooter.simulated_paths

        # compute the rollout in full TF to keep all computation on the GPU
        # a = action dim
        # s = state dim
        # n = batch size = num states for which MPC act * simulated rollouts
        # i = number of states in batch for act
        self._input_state_ph_is = tf.placeholder(
            tf.float32, [None, env_info.ob_dim()])
        initial_state_ns = tf.tile(
            self._input_state_ph_is, (self._sims_per_state, 1))

        # policy shooter samples from
        def _policy(t, states_ns):
            explore = create_random_tf_action(env_info.ac_space())
            return tf.cond(t < opt_horizon,
                           lambda: explore(states_ns),
                           lambda: exploit(states_ns))

        # h = horizon
        n = tf.shape(initial_state_ns)[0]
        # to create a matrix of actions, we need to know n, but we won't know
        # it until runtime, so we can't globally initialize the action matrix
        # instead, we initialize it to the right size at every act() call.
        self._actions_hna = tf.get_variable(
            'dynamics_actions', validate_shape=False,
            initializer=tf.zeros([mpc_horizon, n, env_info.ac_dim()]),
            dtype=tf.float32, collections=[])
        # the states at zero-axis index i here are the RESULTING states
        # from taking the action at index i in _actions_hna
        self._states_hns = tf.get_variable(
            'dynamics_states', validate_shape=False,
            initializer=tf.zeros([mpc_horizon, n, env_info.ob_dim()]),
            dtype=tf.float32, collections=[])

        def _body(t, state_ns, rewards):
            action_na = _policy(t, state_ns)
            save_action_op = tf.scatter_update(self._actions_hna, t, action_na)
            next_state_ns = dyn_model.predict_tf(state_ns, action_na)
            save_state_op = tf.scatter_update(
                self._states_hns, t, next_state_ns)
            curr_reward = reward_fn(state_ns, action_na, next_state_ns)
            # not super numerically stable discounting, would be better to save
            # per-step rewards and then apply a recurrent discount formula
            t_fl = tf.to_float(t)
            discount = flags().experiment.discount
            next_rewards = rewards + tf.pow(discount, t_fl) * curr_reward
            with tf.control_dependencies([save_action_op, save_state_op]):
                return [t + 1, next_state_ns, next_rewards]

        loop_vars = [0, initial_state_ns, tf.zeros((n,))]
        _, _, self._final_rewards_n = tf.while_loop(
            lambda t, _, __: t < mpc_horizon, _body,
            loop_vars, back_prop=False)
        self._mpc_horizon = mpc_horizon

        self._learner = learner
        self._learner_test_env = make_venv(
            flags().experiment.make_env, 10)

    def _act(self, states):
        nstates = len(states)
        tf.get_default_session().run(
            [self._actions_hna.initializer,
             self._states_hns.initializer], feed_dict={
                 self._input_state_ph_is: states})
        trajectory_rewards_n = tf.get_default_session().run(
            self._final_rewards_n, feed_dict={
                self._input_state_ph_is: states})
        action_hna = tf.get_default_session().run(self._actions_hna)
        state_hns = tf.get_default_session().run(self._states_hns)

        # p = num simulated paths, i = nstates
        # note b/c of the way tf.tile works we need to reshape by p then i
        per_state_simulation_rewards_ip = trajectory_rewards_n.reshape(
            self._sims_per_state, nstates).T
        best_ac_ix_i = per_state_simulation_rewards_ip.argmax(axis=1)
        action_hpia = action_hna.reshape(
            self._mpc_horizon, self._sims_per_state, nstates,
            env_info.ac_dim())
        state_hpis = state_hns.reshape(
            self._mpc_horizon, self._sims_per_state, nstates,
            env_info.ob_dim())
        action_hipa = np.swapaxes(action_hpia, 1, 2)
        state_hips = np.swapaxes(state_hpis, 1, 2)
        best_ac_hia = action_hipa[
            :, np.arange(nstates), best_ac_ix_i, :]
        best_ob_his = state_hips[
            :, np.arange(nstates), best_ac_ix_i, :]

        best_ac_iha = np.swapaxes(best_ac_hia, 0, 1)
        best_ob_ihs = np.swapaxes(best_ob_his, 0, 1)
        return best_ac_hia[0], best_ac_iha, best_ob_ihs

    def act(self, states_ns):
        # This batch size is specific to HalfCheetah and my setup.
        # A more appropriate version of this method should query
        # GPU memory size, and use the state dimension + MPC horizon
        # to figure out the appropriate batch amount.
        batch_size = 500
        return rate_limit(batch_size, self._act, states_ns)

    def planning_horizon(self):
        return self._mpc_horizon

    def fit(self, data):
        self._learner.fit(data)

    def log(self, most_recent):
        # out-of-band learner evaluation
        learner = as_controller(self._learner.act)
        learner_paths = sample_venv(
            self._learner_test_env, learner, most_recent.max_horizon)
        rews = [path.rewards for path in learner_paths]
        reporter.add_summary_statistics('learner reward', rews)
