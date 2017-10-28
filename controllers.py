import tensorflow as tf
import numpy as np
import time
from utils import get_ac_dim, get_ob_dim, build_mlp


class Controller:
    def __init__(self):
        pass

    def get_action(self, state):
        raise NotImplementedError

    def fit(self, data):
        pass

    def reset(self, nstates):
        pass

class RandomController(Controller):
    def __init__(self, env):
        super().__init__()
        self.ac_space = env.action_space

    def get_action(self, states):
        nstates = len(states)
        return self._sample_n(nstates)

    def _sample_n(self, n):
        return np.random.uniform(
            low=self.ac_space.low,
            high=self.ac_space.high,
            size=(n,) + self.ac_space.shape)


class MPCcontroller(Controller):
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 sess=None,
                 policy=None):
        super().__init__()
        self.ac_dim = get_ac_dim(env)
        self.ac_space = env.action_space
        self.sess = sess
        self.num_simulated_paths = num_simulated_paths

        # compute the rollout in full TF to keep all computation on the GPU
        # a = action dim
        # s = state dim
        # n = batch size = num states to get MPC actions for * simulated rollouts
        # i = number of states in batch for get_action
        self.input_state_ph_is = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)], 'mpc_input_state')
        state_ns = tf.tile(self.input_state_ph_is, (num_simulated_paths, 1))
        # use the specified policy during MPC rollouts
        ac_space = env.action_space
        if policy is None:
            policy = self._create_random_policy(ac_space)
        self.initial_action_na = policy(state_ns, is_initial=True)
        self.input_action_ph_na = tf.placeholder(
            tf.float32, [None, self.ac_dim], 'mpc_input_action')
        def body(t, state_ns, action_na, costs):
            next_state_ns = dyn_model.predict_tf(state_ns, action_na)
            next_costs = cost_fn(state_ns, action_na, next_state_ns, costs)
            next_action_na = policy(next_state_ns, is_initial=False)
            return [t + 1, next_state_ns, next_action_na, next_costs]
        n = tf.shape(state_ns)[0]
        loop_vars = [
            tf.constant(0),
            state_ns,
            self.input_action_ph_na,
            tf.zeros((n,))]
        self.loop = tf.while_loop(lambda t, _, __, ___: t < horizon, body,
                                  loop_vars, back_prop=False)

    @staticmethod
    def _create_random_policy(ac_space):
        def policy(state_ns, **_):
            n = tf.shape(state_ns)[0]
            ac_dim = ac_space.low.shape
            ac_na = tf.random_uniform((n,) + ac_dim)
            ac_na *= (ac_space.high - ac_space.low)
            ac_na += ac_space.low
            return ac_na
        return policy

    def _get_action(self, states):
        nstates = len(states)

        action_na = self.sess.run(self.initial_action_na,
                                  feed_dict={self.input_state_ph_is: states})
        _, _, _, trajectory_costs_n = self.sess.run(self.loop, feed_dict={
            self.input_state_ph_is: states,
            self.input_action_ph_na: action_na})

        # p = num simulated paths, i = nstates
        # note b/c of the way tf.tile works we need to reshape by p then i
        per_state_simulation_costs_ip = trajectory_costs_n.reshape(
            self.num_simulated_paths, nstates).T
        best_ac_ix_i = per_state_simulation_costs_ip.argmin(axis=1)
        action_samples_ipa = np.swapaxes(action_na.reshape(
            self.num_simulated_paths, nstates, self.ac_dim), 0, 1)
        best_ac_ia = action_samples_ipa[np.arange(nstates), best_ac_ix_i, :]

        return best_ac_ia

    def get_action(self, states):
        # This batch size is specific to HalfCheetah and my setup.
        # A more appropriate version of this method should query
        # GPU memory size, and use the state dimension + MPC horizon
        # to figure out the appropriate batch amount.
        batch_size = 500
        if len(states) <= batch_size:
            return self._get_action(states)

        acs = np.empty((len(states), self.ac_dim))
        for i in range(0, len(states) - batch_size + 1, batch_size):
            loc = slice(i, i + batch_size)
            acs[loc] = self._get_action(states[loc])
        return acs

class BPTT(Controller):
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=None,
                 cost_fn=None,
                 learning_rate=None,
                 depth=None,
                 width=None,
                 batch_size=None,
                 epochs=None,
                 sess=None):
        super().__init__()
        self.sess = sess
        self.batch_size = batch_size
        self.epochs = epochs
        self.ac_space = env.action_space
        self.ob_dim = get_ob_dim(env)
        self.ac_dim = get_ac_dim(env)
        self.width = width
        self.depth = depth

        # rnn used by policy
        self.rnn = tf.contrib.rnn.OutputProjectionWrapper(
            tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.GRUCell(width) for _ in range(depth)]),
            self.ac_dim,
            activation=tf.sigmoid)
        
        # a = action dim
        # s = state dim
        # n = batch size
        # h = hidden unit size
        self.initial_rnn_state_list_nh = [
            tf.placeholder(tf.float32, [None, width]) for _ in range(depth)]
        self.input_state_ph_ns = tf.placeholder(
            tf.float32, [None, self.ob_dim])
        self.policy_action_na, self.resulting_rnn_state_nh = self._rnn_policy(
            self.input_state_ph_ns, self.initial_rnn_state_list_nh)
        self.maintained_rnn_state = None

        # compute the rollout in full TF to keep all computation on the GPU
        # reuse the policy network for BPTT model-based optimization
        self.bptt_initial_state_ph_ns = tf.placeholder(
            tf.float32, [batch_size, self.ob_dim], "bptt_input_state")
        def body(t, state_ns, rnn_state_nh, costs_n):
            action_na, next_rnn_state_nh = self._rnn_policy(
                state_ns, rnn_state_nh)
            next_state_ns = dyn_model.predict_tf(state_ns, action_na)
            next_costs_n = cost_fn(state_ns, action_na, next_state_ns, costs_n)
            return [t + 1, next_state_ns, next_rnn_state_nh, next_costs_n]
        loop_vars = [
            tf.constant(0),
            self.bptt_initial_state_ph_ns,
            self.rnn.zero_state(batch_size, tf.float32),
            tf.zeros((batch_size,))]
        _, _, _, costs_n = tf.while_loop(
            lambda t, _, __, ___: t < horizon, body, loop_vars)
        self.mean_cost = tf.reduce_mean(costs_n)
        policy_vars = self.rnn.trainable_variables
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.mean_cost, var_list=policy_vars)

    def fit(self, data):
        all_obs = data.stationary_obs()
        nexamples = len(all_obs)
        nbatches = max(nexamples // self.batch_size, 1)
        batches = np.random.randint(nexamples, size=(
            self.epochs * nbatches, self.batch_size))
        for batch_idx in batches:
            input_states_sample = all_obs[batch_idx]
            self.sess.run(self.update_op, feed_dict={
                self.bptt_initial_state_ph_ns: input_states_sample})

    def reset(self, nstates):
        self.maintained_rnn_state = [
            np.zeros((nstates, self.width))
            for _ in range(self.depth)]

    def get_action(self, states_ns):
        feed_dict = {
            self.input_state_ph_ns: states_ns}
        for layer_state_ph, layer_state in zip(self.initial_rnn_state_list_nh,
                                               self.maintained_rnn_state):
            feed_dict[layer_state_ph] = layer_state
        action_na, next_rnn_state_nh = self.sess.run(
            [self.policy_action_na, self.resulting_rnn_state_nh],
            feed_dict=feed_dict)
        self.maintained_rnn_state = next_rnn_state_nh
        return action_na

    def _rnn_policy(self, state_ns, rnn_state_nh):
        ac_na, next_rnn_state_nh = self.rnn(state_ns, rnn_state_nh)
        ac_na *= (self.ac_space.high - self.ac_space.low)
        ac_na += self.ac_space.low
        return ac_na, next_rnn_state_nh

class MPCMF(Controller):
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=None,
                 cost_fn=None,
                 num_simulated_paths=None,
                 learning_rate=None,
                 depth=None,
                 width=None,
                 batch_size=None,
                 epochs=None,
                 dagger=False,
                 explore_std=0, # 0 means use uniform exploration, >0 normal
                 sess=None):
        super().__init__()
        self.sess = sess
        self.batch_size = batch_size
        self.epochs = epochs
        self.ob_dim = get_ob_dim(env)
        self.ac_dim = get_ac_dim(env)
        self.width = width
        self.depth = depth
        self.ac_space = env.action_space
        self.dagger = dagger
        self.explore_std = explore_std

        # create placeholder for training an MPC learner
        # a = action dim
        # s = state dim
        # n = batch size
        self.input_state_ph_ns = tf.placeholder(
            tf.float32, [None, self.ob_dim])
        self.policy_action_na = self._exploit_policy(
            self.input_state_ph_ns, reuse=None)
        self.expert_action_ph_na = tf.placeholder(
            tf.float32, [None, self.ac_dim])
        self.mse = tf.losses.mean_squared_error(
            self.expert_action_ph_na,
            self.policy_action_na)
        
        # use the learner value to expand the MPC (first action is random)
        self.mpc = MPCcontroller(
            env, dyn_model, horizon, cost_fn, num_simulated_paths, sess,
            self._policy)
        
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.mse)

        self._labelled_acs = np.empty((0, self.ac_dim))

    def _exploit_policy(self, states_ns, reuse=True):
        ac_na = build_mlp(
            states_ns, scope='mpcmf_policy_mean',
            n_layers=self.depth, size=self.width, activation=tf.nn.relu,
            output_activation=tf.sigmoid, reuse=reuse)
        ac_na *= self.ac_space.high - self.ac_space.low
        ac_na += self.ac_space.low
        return ac_na

    def _pr(self, data):
        mse = self.sess.run(self.mse, feed_dict={
                self.input_state_ph_ns: data.stationary_obs(),
                self.expert_action_ph_na: self._labelled_acs})
        print(mse)

    def _explore_policy(self, state_ns):
        if self.explore_std == 0:
            # Just a note, I saw that
            # DAgger on the random_policy doesn't learn, expert is too variable
            # so minimum error achieved is still too large.
            random_policy = MPCcontroller._create_random_policy(self.ac_space)
            return random_policy(state_ns)
        
        ac_na = self._exploit_policy(state_ns, reuse=True)
        ac_width = self.ac_space.high - self.ac_space.low
        std_a = tf.constant(ac_width * self.explore_std, tf.float32)
        perturb_na = tf.random_normal([tf.shape(state_ns)[0], self.ac_dim])
        perturb_na *= std_a
        ac_na += perturb_na
        ac_na = tf.minimum(ac_na, self.ac_space.high)
        ac_na = tf.maximum(ac_na, self.ac_space.low)
        return ac_na
        
    def _policy(self, state_ns, is_initial=True):
        if is_initial:
            return self._explore_policy(state_ns)
        else:
            return self._exploit_policy(state_ns, reuse=True)

    def fit(self, data):
        self._add_labels(data)
        all_obs = data.stationary_obs()
        all_acs = self._labelled_acs
        nexamples = len(all_obs)
        assert nexamples == len(all_acs), (nexamples, len(all_acs))
        per_epoch = max(nexamples // self.batch_size, 1)
        batches = np.random.randint(nexamples, size=(
            self.epochs * per_epoch, self.batch_size))
        self._pr(data)          
        for i, batch_idx in enumerate(batches, 1):
            input_states_sample = all_obs[batch_idx]
            label_actions_sample = all_acs[batch_idx]
            self.sess.run(self.update_op, feed_dict={
                self.input_state_ph_ns: input_states_sample,
                self.expert_action_ph_na: label_actions_sample})
            if (i + 1) % (self.epochs * per_epoch // 10) == 0:
                print('epoch', (i + 1) // per_epoch, 'of', self.epochs)
                self._pr(data)

    def get_action(self, states_ns):
        if not self.dagger:
            return self.mpc.get_action(states_ns)

        return self.sess.run(self.policy_action_na, feed_dict={
            self.input_state_ph_ns: states_ns})

    def _add_labels(self, data):
        if not self.dagger:
            # if not in dagger mode, expert's already doing the rollouts
            self._labelled_acs = data.stationary_acs()
            return

        # assumes data is getting appended (so prefix stays the same)

        obs = data.stationary_obs()
        acs = self._labelled_acs
        env_horizon = data.obs.shape[0]
        assert (len(obs) - len(acs)) % env_horizon == 0

        to_label = obs[len(acs):]
        new_acs = self.mpc.get_action(to_label)
        self._labelled_acs = np.concatenate(
            [self._labelled_acs, new_acs], axis=0)
