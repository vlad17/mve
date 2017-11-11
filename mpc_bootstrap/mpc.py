"""Model predictive control (MPC) controller."""

import numpy as np
import tensorflow as tf

from controller import Controller
from ddpg_learner import DDPGLearner
from utils import (create_random_tf_policy, get_ac_dim, get_ob_dim)

class MPC(Controller):  # pylint: disable=too-many-instance-attributes
    """
    Random MPC if learner is None. Otherwise, MPC which takes the action
    specified by learner during simulated rollouts. Otherwise, the learner
    specified a function accepting a TF tensor of batched states and
    returning a TF tensor of batched actions, tf_action.

    In addition, policy should accept a keyword is_initial which indicates
    whether this is the first action in the simulated rollout.
    """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 reward_fn=None,
                 num_simulated_paths=10,
                 sess=None,
                 learner=None):
        self._ac_dim = get_ac_dim(env)
        self.sess = sess
        self.num_simulated_paths = num_simulated_paths

        # compute the rollout in full TF to keep all computation on the GPU
        # a = action dim
        # s = state dim
        # n = batch size = num states for which MPC act * simulated rollouts
        # i = number of states in batch for act
        self.input_state_ph_is = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)], 'mpc_input_state')
        state_ns = tf.tile(self.input_state_ph_is, (num_simulated_paths, 1))
        # use the specified policy during MPC rollouts
        ac_space = env.action_space
        if learner is None:
            # TODO(someone): Rename this. It's confusing that this is called
            # policy, but it's not of type policy.Policy.
            policy = create_random_tf_policy(ac_space)
        else:
            policy = learner.tf_action
        self.initial_action_na = policy(state_ns, is_initial=True)
        self.input_action_ph_na = tf.placeholder(
            tf.float32, [None, self._ac_dim], 'mpc_input_action')

        def _body(t, state_ns, action_na, rewards):
            next_state_ns = dyn_model.predict_tf(state_ns, action_na)
            next_rewards = reward_fn(
                state_ns, action_na, next_state_ns, rewards)
            next_action_na = policy(next_state_ns, is_initial=False)
            return [t + 1, next_state_ns, next_action_na, next_rewards]
        n = tf.shape(state_ns)[0]
        loop_vars = [
            0,
            state_ns,
            self.input_action_ph_na,
            tf.zeros((n,))]
        self.loop = tf.while_loop(lambda t, _, __, ___: t < horizon, _body,
                                  loop_vars, back_prop=False)
        self.learner = learner

    def _act(self, states):
        nstates = len(states)
        # pylint: disable=protected-access
        if self.learner is not None and \
           isinstance(self.learner, DDPGLearner) and \
           hasattr(self.learner._agent, 'acs_initializer'):
            # TODO: I know no good way to solve this crazy hackiness
            # maybe self.learner.reset() somehow, but still need the local
            # placeholder available... Damn tensorflow.
            # pylint:disable=protected-access
            self.sess.run(self.learner._agent.acs_initializer,
                          feed_dict={self.input_state_ph_is: states})

        action_na = self.sess.run(self.initial_action_na,
                                  feed_dict={self.input_state_ph_is: states})
        _, _, _, trajectory_rewards_n = self.sess.run(self.loop, feed_dict={
            self.input_state_ph_is: states,
            self.input_action_ph_na: action_na})

        # p = num simulated paths, i = nstates
        # note b/c of the way tf.tile works we need to reshape by p then i
        per_state_simulation_rewards_ip = trajectory_rewards_n.reshape(
            self.num_simulated_paths, nstates).T
        best_ac_ix_i = per_state_simulation_rewards_ip.argmax(axis=1)
        action_samples_ipa = np.swapaxes(action_na.reshape(
            self.num_simulated_paths, nstates, self._ac_dim), 0, 1)
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
