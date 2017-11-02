"""File for the base classes and implementations for controlling policies."""

import numpy as np
import tensorflow as tf

import logz
from utils import get_ac_dim, get_ob_dim, build_mlp, Dataset


class Policy:
    """A Policy represents a possibly stateful agent."""

    def act(self, states_ns):
        """
        Return the action for every state in states_ns, where the batch size
        is n and the state shape is s.
        """
        raise NotImplementedError

    def reset(self, nstates):
        """
        For stateful RNN-based policies, this should be called at the
        beginning of each rollout to reset the state.
        """
        pass


class Learner(Policy):  # pylint: disable=abstract-method
    """
    A learner acts in a manner that is most conducive to its own learning,
    as long as the resulting states are labelled with correct actions to have
    taken by an expert. Given such a labelled dataset, it can also learn from
    it. Only stateless/stationary policy learners are currently supported
    (don't override reset()).
    """

    def tf_action(self, states_ns, is_initial=False):
        """
        Return the TF tensor for the action that the learner would take.
        The learner may choose to take different actions depending on whether
        is_initial is true or not, which when set indicates that this is
        the first action in a simulated rollout.
        """
        raise NotImplementedError

    def fit(self, obs, acs):
        """Fit the learner to the specified labels."""
        raise NotImplementedError


class Controller(Policy):  # pylint: disable=abstract-method
    """
    A possibly stateful controller, which decides which actions to take.
    A controller might choose to label the dataset.
    """

    def fit(self, data):
        """A controller might fit internal learners here."""
        pass

    def label(self, states_ns):
        """Optionally label a dataset's existing states with new actions."""
        pass

    def log(self, **kwargs):
        """A controller might log additional stats here."""
        pass


class RandomController(Controller):
    """An agent that acts uniformly randomly in the action space."""

    def __init__(self, env):
        self.ac_space = env.action_space

    def act(self, states_ns):
        nstates = len(states_ns)
        return self._sample_n(nstates)

    def _sample_n(self, n):
        return np.random.uniform(
            low=self.ac_space.low,
            high=self.ac_space.high,
            size=(n,) + self.ac_space.shape)


def _create_random_policy(ac_space):
    def _policy(state_ns, **_):
        n = tf.shape(state_ns)[0]
        ac_dim = ac_space.low.shape
        ac_na = tf.random_uniform((n,) + ac_dim)
        ac_na *= (ac_space.high - ac_space.low)
        ac_na += ac_space.low
        return ac_na
    return _policy


class MPC(Controller):
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
                 cost_fn=None,
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
            policy = _create_random_policy(ac_space)
        else:
            policy = learner.tf_action
        self.initial_action_na = policy(state_ns, is_initial=True)
        self.input_action_ph_na = tf.placeholder(
            tf.float32, [None, self._ac_dim], 'mpc_input_action')

        def _body(t, state_ns, action_na, costs):
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
        self.loop = tf.while_loop(lambda t, _, __, ___: t < horizon, _body,
                                  loop_vars, back_prop=False)

    def _act(self, states):
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
            self.num_simulated_paths, nstates, self._ac_dim), 0, 1)
        best_ac_ia = action_samples_ipa[np.arange(nstates), best_ac_ix_i, :]

        return best_ac_ia

    def act(self, states_ns):
        # This batch size is specific to HalfCheetah and my setup.
        # A more appropriate version of this method should query
        # GPU memory size, and use the state dimension + MPC horizon
        # to figure out the appropriate batch amount.
        batch_size = 500
        if len(states_ns) <= batch_size:
            return self._act(states_ns)

        acs = np.empty((len(states_ns), self._ac_dim))
        for i in range(0, len(states_ns) - batch_size + 1, batch_size):
            loc = slice(i, i + batch_size)
            acs[loc] = self._act(states_ns[loc])
        return acs


# TODO: reduce the number of instance attributes here
class DeterministicLearner(Learner):  # pylint: disable=too-many-instance-attributes
    """Only noisy on first action."""

    def __init__(self,
                 env,
                 learning_rate=None,
                 depth=None,
                 width=None,
                 batch_size=None,
                 epochs=None,
                 explore_std=0,  # 0 means use uniform exploration, >0 normal
                 sess=None):
        self.sess = sess
        self.batch_size = batch_size
        self.epochs = epochs
        self.explore_std = explore_std
        self.ac_dim = get_ac_dim(env)
        self.width = width
        self.depth = depth
        self.ac_space = env.action_space

        # create placeholder for training an MPC learner
        # a = action dim
        # s = state dim
        # n = batch size
        self.input_state_ph_ns = tf.placeholder(
            tf.float32, [None, get_ob_dim(env)])
        self.policy_action_na = self._exploit_policy(
            self.input_state_ph_ns, reuse=None)
        self.expert_action_ph_na = tf.placeholder(
            tf.float32, [None, self.ac_dim])
        mse = tf.losses.mean_squared_error(
            self.expert_action_ph_na,
            self.policy_action_na)

        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(mse)

    def _exploit_policy(self, states_ns, reuse=True):
        ac_na = build_mlp(
            states_ns, scope='mpcmf_policy_mean',
            n_layers=self.depth, size=self.width, activation=tf.nn.relu,
            output_activation=tf.sigmoid, reuse=reuse)
        ac_na *= self.ac_space.high - self.ac_space.low
        ac_na += self.ac_space.low
        return ac_na

    def _explore_policy(self, state_ns):
        if self.explore_std == 0:
            random_policy = _create_random_policy(self.ac_space)
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

    def tf_action(self, states_ns, is_initial=True):
        if is_initial:
            return self._explore_policy(states_ns)
        return self._exploit_policy(states_ns, reuse=True)

    def fit(self, obs, acs):
        nexamples = len(obs)
        assert nexamples == len(acs), (nexamples, len(acs))
        per_epoch = max(nexamples // self.batch_size, 1)
        batches = np.random.randint(nexamples, size=(
            self.epochs * per_epoch, self.batch_size))
        for batch_idx in batches:
            input_states_sample = obs[batch_idx]
            label_actions_sample = acs[batch_idx]
            self.sess.run(self.update_op, feed_dict={
                self.input_state_ph_ns: input_states_sample,
                self.expert_action_ph_na: label_actions_sample})

    def act(self, states_ns):
        return self.sess.run(self.policy_action_na, feed_dict={
            self.input_state_ph_ns: states_ns})


class BootstrappedMPC(Controller):
    """
    A bootstrapping version of the MPC controller. Learn a policy from MPC
    rollout data, and use it to run the simulations after the first action.
    """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=None,
                 cost_fn=None,
                 num_simulated_paths=None,
                 learner=None,
                 sess=None):
        self.env = env
        self.learner = learner
        self.mpc = MPC(
            env, dyn_model, horizon, cost_fn, num_simulated_paths, sess,
            learner)

    def act(self, states_ns):
        return self.mpc.act(states_ns)

    def fit(self, data):
        obs = data.stationary_obs()
        acs = data.stationary_acs()
        self.learner.fit(obs, acs)

    def log(self, **kwargs):
        # TODO: get rid of this circular dep once sample.py exists
        from main import sample
        horizon = kwargs['horizon']
        render = False
        paths = sample(self.env, self.learner, horizon, render)
        data = Dataset(self.env, horizon)
        data.add_paths(paths)
        returns = data.rewards.sum(axis=0)
        logz.log_tabular('LearnerAverageReturn', np.mean(returns))
        logz.log_tabular('LearnerStdReturn', np.std(returns))
        logz.log_tabular('LearnerMinimumReturn', np.min(returns))
        logz.log_tabular('LearnerMaximumReturn', np.max(returns))


class DaggerMPC(Controller):
    """
    Like BootstrappedMPC, but use the learned policy to take actions
    and DAgger to learn.
    """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=None,
                 cost_fn=None,
                 num_simulated_paths=None,
                 learner=None,
                 sess=None):
        self.learner = learner
        self.mpc = MPC(
            env, dyn_model, horizon, cost_fn, num_simulated_paths, sess,
            learner)

    def act(self, states_ns):
        return self.learner.act(states_ns)

    def label(self, states_ns):
        return self.mpc.act(states_ns)

    def fit(self, data):
        obs = data.stationary_obs()
        acs = data.stationary_labelled_acs()
        self.learner.fit(obs, acs)
