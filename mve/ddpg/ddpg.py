"""DDPG training."""

import tensorflow as tf
import numpy as np

from context import flags
from dataset import Dataset
from dynamics_metrics import DynamicsMetrics
import env_info
from log import debug
import reporter
from sample import sample_venv
from tf_reporter import TFReporter
from qvalues import qvals, offline_oracle_q, oracle_q
from utils import scale_from_box, flatgrad, timeit


def _tf_seq(a, b_fn):
    # force a temporal dependence on the evaluation of b after a
    # from Haskell :)
    with tf.control_dependencies([a]):
        return b_fn()


def _tf_doif(cond, if_true_fn):
    return tf.cond(
        cond,
        lambda: _tf_seq(if_true_fn(), lambda: tf.constant(0)),
        lambda: 0)


class DDPG:  # pylint: disable=too-many-instance-attributes
    """
    Builds the part of the TF graph responsible for training actor
    and critic networks with the DDPG algorithm
    """

    def __init__(self, actor, critic, discount=0.99, scope='ddpg',  # pylint: disable=too-many-branches
                 actor_lr=1e-3, critic_lr=1e-3, explore_stddev=0.2,
                 learned_dynamics=None):

        self._reporter = TFReporter()

        # Oh, you think you can replace None below with the known
        # batch size flags().ddpg.learner_batch_size, you silly goose?
        # Not unless you want to have a nice 12-hour debugging session!
        # actor.tf_* methods are all built with scope-dependent
        # tf.get_variable calls, which are sensitive to the SHAPE of
        # the neural network input. Other places already have None-sized
        # placeholders, so the "true" actor network is used only when
        # the input shape is also None.
        self.obs0_ph_ns = tf.placeholder(
            tf.float32, shape=[None, env_info.ob_dim()])
        self.obs1_ph_ns = tf.placeholder(
            tf.float32, shape=[None, env_info.ob_dim()])
        self.terminals1_ph_n = tf.placeholder(
            tf.float32, shape=[None])
        self.rewards_ph_n = tf.placeholder(
            tf.float32, shape=[None])
        self.actions_ph_na = tf.placeholder(
            tf.float32, shape=[None, env_info.ac_dim()])

        # actor maximizes current Q or oracle Q
        act_obs0 = actor.tf_action(self.obs0_ph_ns)
        critic_at_actor_n = critic.tf_critic(
            self.obs0_ph_ns,
            act_obs0)
        self._reporter.stats('actor Q', critic_at_actor_n)

        opt = tf.train.AdamOptimizer(
            learning_rate=actor_lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        if flags().ddpg.actor_critic_mixture:
            assert flags().ddpg.mixture_estimator == 'oracle', \
                'only oracle estimator handled for actor-critic mix, ' \
                'got {}'.format(flags().ddpg.mixture_estimator)
            # h-step observations
            h = flags().ddpg.model_horizon
            debug('using oracle Q estimator with {} steps for actor grad', h)
            nenvs = flags().ddpg.learner_batch_size
            # symmetric finite diffs + one central evaluation
            nenvs *= env_info.ac_dim() * 2 + 1
            self._oracle_actor_venv = env_info.make_venv(nenvs)
            self._oracle_actor_venv.reset()

            # This actor loss actually has some trickery going on to get
            # autodiff to do the right thing.
            #
            # The actor-critic based policy gradient of DDPG (wrt policy mu
            # parameters t) for a single state s is:
            #
            # d/dt(Q(s, mu(s))) = J(s) . dQ/da(s, mu(s))
            #
            # with J(s) the Jacobian of mu wrt t at a fixed state s.
            # The above holds by chain rule
            # and linearity of mean. If Q is perfect, the above is the policy
            # gradient by the Deterministic Policy Gradient Theorem (Silver
            # et al 2014) when s is sampled from the current policy's state
            # visitation distribution.
            #
            # When s is sampled from a different distribution then we require
            # a slightly different analysis from Degris et al 2012 to show
            # that J(s) . dQ/da is an improvement direction.
            # For a batch of states
            # s by linearity of the mean function we can average gradients.
            #
            # When using model-expanded critic, Q becomes a function of t
            # (the actions chosen are dependent on Q).
            #
            # Consider a fixed 1-step model-based expansion for a critic C:
            # Q = r(s, a, s') + gamma * C(s', a'), s' = f(s, a), a' = mu(s')
            # A modified version of the Degris et al analysis yields
            # an improvement direction with R(a) = r(s, a, s')
            # J(s) . dR/da + gamma * J(s') . dC/da(s', a')
            # The above can be recovered in an autodiff tool with a loss:
            # mu(s) . dR/da + gamma * C(s', mu(s'))

            # TODO: multistep this will need to be a list of placeholders
            self._dr_ph_na = tf.placeholder(
                tf.float32, shape=[None, env_info.ac_dim()])
            self._expanded_obs_ph_ns = tf.placeholder(
                tf.float32, shape=[None, env_info.ob_dim()])
            self._expanded_terminals_ph_n = tf.placeholder(
                tf.float32, shape=[None])
            self._actor_loss = -1 * tf.reduce_mean(
                tf.reduce_sum(act_obs0 * self._dr_ph_na, axis=1) +
                (1 - self._expanded_terminals_ph_n) * discount *
                critic.tf_critic(
                    self._expanded_obs_ph_ns,
                    actor.tf_action(self._expanded_obs_ph_ns)))
            original_loss = -1 * tf.reduce_mean(critic_at_actor_n)

            new_grad = flatgrad(opt, self._actor_loss, actor.variables)
            old_grad = flatgrad(opt, original_loss, actor.variables)
            cos = tf.reduce_sum(new_grad * old_grad) / (
                tf.norm(new_grad) * tf.norm(old_grad))
            mag = tf.norm(new_grad) / tf.norm(old_grad)
            self._reporter.scalar('expanded actor grad : original cosine',
                                  cos)
            self._reporter.scalar('expanded actor grad : original magnitude',
                                  mag)
            self._reporter.scalar('l2 expanded - original actor grad',
                                  tf.norm(new_grad - old_grad))
        else:
            self._actor_loss = -1 * tf.reduce_mean(critic_at_actor_n)
        self._reporter.scalar('actor loss', self._actor_loss)
        with tf.variable_scope(scope):
            with tf.variable_scope('opt_actor'):
                self._reporter.grads(
                    'actor grad', opt, self._actor_loss, actor.variables)
                optimize_actor_op = opt.minimize(
                    self._actor_loss, var_list=actor.variables)

        # critic minimizes TD-1 error wrt to target Q and target actor
        current_Q_n = critic.tf_critic(
            self.obs0_ph_ns, self.actions_ph_na)
        self._reporter.stats('critic Q', current_Q_n)
        if flags().ddpg.q_target_mixture:
            # h-step observations
            h = flags().ddpg.model_horizon
            if flags().ddpg.mixture_estimator == 'oracle':
                debug('using oracle Q estimator with {} steps as '
                      'target critic', h)
                nenvs = flags().ddpg.learner_batch_size
                self._oracle_q_target_venv = env_info.make_venv(nenvs)
                self._oracle_q_target_venv.reset()
                # the i-th element here is the i-th state after obs1;
                # so h == 0 should equal to obs1_ph_ns
                self._obs_ph_hns = tf.placeholder(
                    tf.float32, shape=[h, None, env_info.ob_dim()])
                # the action taken at the i-th state above above
                self._acs_ph_hna = tf.placeholder(
                    tf.float32, shape=[h, None, env_info.ac_dim()])
                # the resulting done indicator from playing that action
                self._done_ph_hn = tf.placeholder(
                    tf.float32, shape=[h, None])
                # the reward resulting from that action
                self._rew_ph_hn = tf.placeholder(
                    tf.float32, shape=[h, None])
                # this should be the final state resulting from playing
                # self._acs_ph_hna[h-1] on self._obs_ph_hns[h-1]
                self._final_ob_ph_ns = tf.placeholder(
                    tf.float32, shape=[None, env_info.ob_dim()])
                # assume early termination implies reward is 0 from that point
                # on and state is the same
                final_acs_na = actor.tf_target_action(self._final_ob_ph_ns)
                future_Q_n = critic.tf_target_critic(
                    self._final_ob_ph_ns, final_acs_na)
                target_Q_hn = [None] * h
                # big TF graph, could be while loop
                for t in reversed(range(h)):
                    target_Q_hn[t] = self._rew_ph_hn[t] + (
                        (1. - self._done_ph_hn[t]) * discount *
                        future_Q_n)
                    future_Q_n = target_Q_hn[t]
                residual_loss = tf.losses.mean_squared_error(
                    self.rewards_ph_n + (1. - self.terminals1_ph_n) *
                    discount * target_Q_hn[0],
                    current_Q_n)
                for t in range(h):
                    predicted_Q_n = critic.tf_critic(
                        self._obs_ph_hns[t], self._acs_ph_hna[t])
                    if t > 0:
                        weights = 1.0 - self._done_ph_hn[t - 1]
                    else:
                        weights = 1.0 - self.terminals1_ph_n
                    if not flags().ddpg.drop_tdk:
                        residual_loss += tf.losses.mean_squared_error(
                            target_Q_hn[t], predicted_Q_n, weights=weights)
            elif flags().ddpg.mixture_estimator == 'learned':
                debug('using learned-dynamics Q estimator with {} steps as '
                      'target critic', h)
                assert learned_dynamics is not None
                # --- this section is purely for dynamics accuracy recording
                self._dyn_metrics = DynamicsMetrics(
                    h, env_info.make_env, flags().dynamics_metrics, discount)
                self._unroll_states_ph_ns = tf.placeholder(
                    tf.float32, shape=[None, env_info.ob_dim()])
                n = tf.shape(self._unroll_states_ph_ns)[0]
                with tf.variable_scope('scratch-ddpg' + scope):
                    actions_hna = tf.get_variable(
                        'dynamics_actions', initializer=tf.zeros(
                            [h, n, env_info.ac_dim()]),
                        dtype=tf.float32, trainable=False, collections=[],
                        validate_shape=False)
                    states_hns = tf.get_variable(
                        'dynamics_states', initializer=tf.zeros(
                            [h, n, env_info.ob_dim()]),
                        dtype=tf.float32, trainable=False, collections=[],
                        validate_shape=False)
                self._unroll_loop = _tf_unroll(
                    h, self._unroll_states_ph_ns,
                    actor.tf_action, learned_dynamics,
                    actions_hna, states_hns)
                self._initializer = [
                    actions_hna.initializer, states_hns.initializer]
                # --- end section for dyn acc diagnostics

                residual_loss = _tf_compute_model_value_expansion(
                    self.obs0_ph_ns,
                    self.actions_ph_na,
                    self.rewards_ph_n,
                    self.obs1_ph_ns,
                    self.terminals1_ph_n,
                    actor,
                    critic,
                    learned_dynamics)
            else:
                raise ValueError('unrecognized mixture estimator {}'
                                 .format(flags().ddpg.mixture_estimator))
        else:
            next_Q_n = critic.tf_target_critic(
                self.obs1_ph_ns, actor.tf_target_action(self.obs1_ph_ns))
            target_Q_n = self.rewards_ph_n + (1. - self.terminals1_ph_n) * (
                discount * next_Q_n)
            residual_loss = tf.losses.mean_squared_error(
                target_Q_n, current_Q_n)
            self._reporter.stats('target Q', target_Q_n)

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self._critic_loss = residual_loss + reg_loss
        self._reporter.scalar('critic loss', self._critic_loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=critic_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        # perform the critic update after the actor (which is dependent on it)
        # then perform both updates
        with tf.variable_scope(scope):
            with tf.variable_scope('opt_critic'):
                self._reporter.grads(
                    'critic grad', opt, self._critic_loss, critic.variables)
                optimize_critic_op = opt.minimize(
                    self._critic_loss, var_list=critic.variables)
        with tf.control_dependencies([optimize_actor_op, optimize_critic_op]):
            update_targets = tf.group(
                actor.tf_target_update(flags().ddpg.actor_target_rate),
                critic.tf_target_update(flags().ddpg.critic_target_rate))

        for v in actor.variables + critic.variables:
            self._reporter.weights(v.name, v)

        self._copy_targets = tf.group(
            actor.tf_target_update(0),
            critic.tf_target_update(0))

        # Parameter-noise exploration: "train" a noise variable to learn
        # a perturbation stddev for actor network weights that hits the target
        # action stddev. After every gradient step sample how far our current
        # parameter space noise puts our actions from mean.
        with tf.variable_scope(scope), tf.variable_scope('adaption'):
            init_stddev = explore_stddev / (actor.depth + 1)
            adaptive_noise = tf.get_variable(
                'adaptive_noise', trainable=False, initializer=init_stddev)
            # sum of action noise we've seen in the last few updates
            observed_noise_sum = tf.get_variable(
                'observed_noise_sum', trainable=False, initializer=0.)
            # number of observations in sum
            adaption_update_ctr = tf.get_variable(
                'update_ctr', trainable=False, initializer=0)
            self._action_noise = tf.get_variable(
                'prev_mean_action_noise', trainable=False, initializer=0.)
        self._reporter.scalar('adaptive param noise', adaptive_noise)
        self._reporter.scalar('action noise', self._action_noise)
        # This implementation observes the param noise after every gradient
        # step this isn't absolutely necessary but doesn't seem to be a
        # bottleneck.
        #
        # We then adjust parameter noise by the adaption coefficient
        # once every iteration.
        with tf.control_dependencies([optimize_actor_op]):
            re_perturb = actor.tf_perturb_update(adaptive_noise)
            mean_ac = scale_from_box(
                env_info.ac_space(), actor.tf_action(self.obs0_ph_ns))
            with tf.control_dependencies([re_perturb]):
                perturb_ac = scale_from_box(
                    env_info.ac_space(), actor.tf_perturbed_action(
                        self.obs0_ph_ns))
                batch_observed_noise = tf.sqrt(
                    tf.reduce_mean(tf.square(mean_ac - perturb_ac)))
                save_noise = tf.group(
                    tf.assign_add(observed_noise_sum, batch_observed_noise),
                    tf.assign_add(adaption_update_ctr, 1))
        adaption_interval = flags().ddpg.param_noise_adaption_interval
        adaption_rate = 1.01
        with tf.control_dependencies([save_noise]):
            multiplier = tf.cond(
                observed_noise_sum.read_value() <
                explore_stddev * adaption_interval,
                lambda: adaption_rate, lambda: 1 / adaption_rate)
            adapted_noise = adaptive_noise.read_value() * multiplier
            # this is a bit icky, but if it wasn't for TF it'd be easy to read:
            # if we had adaption_interval updates, then adapt the
            # parameter-space noise and clear the running sum and counters.
            # The lambdas are necessary b/c TF relies on op-creation context
            # to create temporal dependencies.
            conditional_update = _tf_doif(
                tf.equal(adaption_update_ctr.read_value(), adaption_interval),
                lambda: _tf_seq(
                    tf.group(
                        tf.assign(adaptive_noise, adapted_noise),
                        tf.assign(self._action_noise,
                                  observed_noise_sum.read_value() /
                                  tf.to_float(
                                      adaption_update_ctr.read_value()))),
                    lambda: tf.group(
                        tf.assign(observed_noise_sum, 0.),
                        tf.assign(adaption_update_ctr, 0))))

        self._optimize = tf.group(update_targets, conditional_update)
        self._actor = actor
        self._critic = critic
        self._venv = env_info.make_venv(16)

        # Sloppily shoving imaginary data for training in here as well.
        # TODO: move this to a separate file (along with lots of other stuff
        # like exploration) when you have time.
        if flags().ddpg.imaginary_buffer > 0:
            assert learned_dynamics is not None
            assert not flags().ddpg.q_target_mixture
            assert not flags().ddpg.actor_critic_mixture
            self._imdata = Dataset(flags().experiment.horizon,
                                   flags().experiment.bufsize)
            self._simulation_states_ph_ns = tf.placeholder(
                tf.float32, shape=[None, env_info.ob_dim()])
            self._sim_actions_na = actor.tf_action(
                self._simulation_states_ph_ns)
            self._sim_next_states_ns = learned_dynamics.predict_tf(
                self._simulation_states_ph_ns, self._sim_actions_na)
            self._sim_rewards_n = env_info.reward_fn()(
                self._simulation_states_ph_ns, self._sim_actions_na,
                self._sim_next_states_ns)

    def evaluate(self, data):
        """misc evaluation"""
        batch_size = flags().ddpg.learner_batch_size
        batch = self._sample(next(data.sample_many(1, batch_size)))
        self._reporter.report(batch)

        if hasattr(self, '_dyn_metrics'):
            self._eval_dynamics(data, '')

        # runs out-of-band trials for less noise performance evaluation
        paths = sample_venv(self._venv, self._actor.target_act)
        rews = [path.rewards.sum() for path in paths]
        reporter.add_summary_statistics('target policy reward', rews)
        acs = np.concatenate([path.acs for path in paths])
        obs = np.concatenate([path.obs for path in paths])
        qs = np.concatenate(qvals(paths, flags().experiment.discount))
        model_horizon = flags().ddpg.model_horizon
        target_qs = self._critic.target_critique(obs, acs)
        qs_estimators = [
            (self._critic.critique(obs, acs), 'critic'),
            (target_qs, 'target'),
            (offline_oracle_q(paths, target_qs, model_horizon),
             'oracle-' + str(model_horizon))]
        for est_qs, name in qs_estimators:
            diffs = est_qs - qs
            if np.all(np.isfinite(diffs)):
                reporter.add_summary_statistics('Q bias/' + name, diffs)
            qmse = np.square(diffs).mean()
            if np.isfinite(qmse):
                reporter.add_summary('Q MSE/' + name, qmse)

        paths = sample_venv(self._venv, self._actor.act)
        rews = [path.rewards.sum() for path in paths]
        reporter.add_summary_statistics('current policy reward', rews)

    def initialize_targets(self):
        """
        New targets are initialized randomly, but they should be initially
        set to equal the initialization of the starting networks.
        """
        debug('copying current network to target for DDPG init')
        tf.get_default_session().run(self._copy_targets)

    def _sample(self, batch):
        obs, next_obs, rewards, acs, terminals = batch
        feed_dict = {
            self.obs0_ph_ns: obs,
            self.obs1_ph_ns: next_obs,
            self.terminals1_ph_n: terminals,
            self.rewards_ph_n: rewards,
            self.actions_ph_na: acs}
        if flags().ddpg.q_target_mixture and \
           flags().ddpg.mixture_estimator == 'oracle':
            self._oracle_expand_states(feed_dict)
        if flags().ddpg.actor_critic_mixture:
            self._oracle_expand_actions(feed_dict)
        return feed_dict

    def _batch_generator(self, data, nbatches, batch_size):
        real_gen = data.sample_many(nbatches, batch_size)
        yield from real_gen
        if flags().ddpg.imaginary_buffer > 0:
            im_batches = int(flags().ddpg.imaginary_buffer * nbatches)
            fake_gen = self._imdata.sample_many(im_batches, batch_size)
            yield from fake_gen

    def train(self, data, nbatches, batch_size, timesteps):
        """Run nbatches training iterations of DDPG"""
        if flags().ddpg.imaginary_buffer > 0:
            with timeit('generating imaginary data'):
                self._generate_data(data, timesteps)

        batches = self._batch_generator(data, nbatches, batch_size)
        for i, batch in enumerate(batches, 1):
            feed_dict = self._sample(batch)
            tf.get_default_session().run(self._optimize, feed_dict)
            if (i % max(nbatches // 10, 1)) == 0:
                cl, al, an = tf.get_default_session().run(
                    [self._critic_loss, self._actor_loss, self._action_noise],
                    feed_dict)
                fmt = '{: ' + str(len(str(nbatches))) + 'd}'
                debug('ddpg ' + fmt + ' of ' + fmt + ' batches - '
                      'critic loss {:.4g} actor loss {:.4g} '
                      'action noise {:.4g}',
                      i, nbatches, cl, al, an)

    def _oracle_expand_states(self, feed_dict):
        h = flags().ddpg.model_horizon
        n = flags().ddpg.learner_batch_size
        initial_states_ns = feed_dict[self.obs1_ph_ns]
        active_n = ~feed_dict[self.terminals1_ph_n].astype(bool)

        obs_hns = np.empty((h, n, env_info.ob_dim()))
        acs_hna = np.empty((h, n, env_info.ac_dim()))
        done_hn = np.empty((h, n))
        rew_hn = np.empty((h, n))

        venv = self._oracle_q_target_venv
        assert n == venv.n, (n, venv.n)
        obs_ns = initial_states_ns
        venv.set_state_from_ob(obs_ns)

        for t in range(h):
            obs_hns[t] = obs_ns
            acs_na = self._actor.target_act(obs_ns)
            obs_ns, reward_n, done_n, _ = venv.step(acs_na)
            acs_hna[t] = acs_na
            done_hn[t] = done_n
            rew_hn[t] = reward_n
            were_not_active = ~active_n
            done_hn[t, were_not_active] = 1.
            rew_hn[t, were_not_active] = 0.
            obs_ns[were_not_active] = obs_hns[t, were_not_active]
            active_n &= ~done_n

        feed_dict[self._obs_ph_hns] = obs_hns
        feed_dict[self._acs_ph_hna] = acs_hna
        feed_dict[self._done_ph_hn] = done_hn
        feed_dict[self._rew_ph_hn] = rew_hn
        feed_dict[self._final_ob_ph_ns] = obs_ns

    def _oracle_Q(self, next_obs):
        h = flags().ddpg.model_horizon
        h_n = np.full(len(next_obs), h, dtype=int)
        next_acs = self._actor.target_act(next_obs)
        model_expanded_Q = oracle_q(
            self._critic.target_critique,
            self._actor.target_act,
            next_obs, next_acs, self._oracle_q_target_venv, h_n)
        return model_expanded_Q

    def _oracle_expand_actions(self, feed_dict):
        obs0_ns = feed_dict[self.obs0_ph_ns]

        # n = mini-batch size
        # e = forward, backward, central evaluations
        next_obs_ns = np.asarray(obs0_ns, dtype=float)
        next_acs_na = np.asarray(self._actor.act(next_obs_ns), dtype=float)
        a = env_info.ac_dim()
        s = env_info.ob_dim()
        e = 2 * a + 1
        obs0_ens = np.tile(next_obs_ns, [e, 1, 1])
        acs0_ena = np.tile(next_acs_na, [e, 1, 1])

        # the best floating point aware step size differentiating a
        # scalar function f at x with symmetric finite differences is
        # cube_root(3|f(x)| * precision / M)
        # where M is a bound on the third-order Taylor expansion term
        # (a local bound on the third derivative if f is thrice-differentiable)
        # fudging the constants at precision = 10^(-16) we get a step size of:
        eps = 1e-5

        for i in range(env_info.ac_dim()):
            acs0_ena[i, :, i] += eps
        for i in range(env_info.ac_dim()):
            acs0_ena[a + i, :, i] -= eps
        # 2 * a + i is for the evaluation along the center

        self._oracle_actor_venv.set_state_from_ob(obs0_ens.reshape(
            -1, s))
        obs1_ens, rew0_en, done_en, _ = self._oracle_actor_venv.step(
            acs0_ena.reshape(-1, a))
        obs1_ens = obs1_ens.reshape(e, -1, s)
        rew0_en = rew0_en.reshape(e, -1)
        done_en = done_en.reshape(e, -1)

        diff_an = rew0_en[:a] - rew0_en[a:2 * a]
        dr_an = diff_an / (2 * eps)
        feed_dict[self._dr_ph_na] = dr_an.T
        feed_dict[self._expanded_obs_ph_ns] = obs1_ens[-1]
        feed_dict[self._expanded_terminals_ph_n] = done_en[-1].astype(float)

    def _eval_dynamics(self, data, prefix):
        # TODO: should really be a separate setting for number of eval samples
        eval_samples = min(flags().dynamics_metrics.evaluation_envs * 10,
                           data.size)
        ixs = np.random.randint(data.size, size=(eval_samples,))

        obs = data.obs[ixs]
        tf.get_default_session().run(
            self._initializer, feed_dict={
                self._unroll_states_ph_ns: obs})

        acs_hna, obs_hns = tf.get_default_session().run(
            self._unroll_loop, feed_dict={
                self._unroll_states_ph_ns: obs})

        planned_acs = np.swapaxes(acs_hna, 0, 1)
        planned_obs = np.swapaxes(obs_hns, 0, 1)
        self._dyn_metrics.log(
            obs, planned_acs, planned_obs, prefix)

    def _generate_data(self, data, timesteps):
        im_to_real_ratio = flags().ddpg.imaginary_buffer
        h = flags().ddpg.model_horizon
        timesteps = int((im_to_real_ratio * timesteps) // h)
        sample_ix = np.random.randint(data.size, size=timesteps)
        obs = data.obs[sample_ix]

        for _ in range(h):
            acs, next_obs, rews = tf.get_default_session().run(
                [self._sim_actions_na,
                 self._sim_next_states_ns,
                 self._sim_rewards_n], feed_dict={
                     self._simulation_states_ph_ns: obs})
            for o, a, n, r in zip(obs, acs, next_obs, rews):
                self._imdata.next(o, n, r, False, a)
            obs = next_obs

# TODO: create general methods for these patterns and abstract from
# random shooter code as well


def _tf_model_value_expand(h, initial_states_ns, act, critique, dynamics,
                           back_prop=False):
    n = tf.shape(initial_states_ns)[0]
    # discount isn't used in a numerically stable way here, consider unrolloing
    # all the rewards and then applying horner's method.
    discount = flags().experiment.discount
    reward_fn = env_info.reward_fn()

    def _body(t, rewards_n, states_ns):
        actions_na = act(states_ns)
        next_states_ns = dynamics.predict_tf(states_ns, actions_na)
        curr_reward = reward_fn(states_ns, actions_na, next_states_ns)
        t_fl = tf.to_float(t)
        next_rewards_n = rewards_n + tf.pow(discount, t_fl) * curr_reward
        return [t + 1, next_rewards_n, next_states_ns]

    _, final_rewards_n, final_states_ns = tf.while_loop(
        lambda t, _, __: t < h, _body,
        [0, tf.zeros((n,)), initial_states_ns],
        back_prop=back_prop)

    h_fl = tf.to_float(h)
    final_critic_n = tf.pow(discount, h_fl) * critique(
        final_states_ns, act(final_states_ns))
    return final_rewards_n + final_critic_n


def _tf_unroll(h, initial_states_ns, act, dynamics, acs_hna, states_hns):
    def _body(t, state_ns):
        action_na = act(state_ns)
        save_action_op = tf.scatter_update(acs_hna, t, action_na)
        next_state_ns = dynamics.predict_tf(state_ns, action_na)
        save_state_op = tf.scatter_update(
            states_hns, t, next_state_ns)
        with tf.control_dependencies([save_action_op, save_state_op]):
            return [t + 1, next_state_ns]

    loop_vars = [0, initial_states_ns]
    loop, _ = tf.while_loop(lambda t, _: t < h, _body,
                            loop_vars, back_prop=False)

    with tf.control_dependencies([loop]):
        return acs_hna.read_value(), states_hns.read_value()


def _tf_compute_model_value_expansion(
        obs0_ns,
        acs0_na,
        rew0_n,
        obs1_ns,
        terminals1_n,
        actor,
        critic,
        dynamics):
    h = flags().ddpg.model_horizon
    reward_fn = env_info.reward_fn()
    discount = flags().experiment.discount

    # the i-th states along the first axis here (of length h) correspond
    # to the states that occur after the obs1 state (the resulting state
    # from the real-data transitions). Thus at position 0 we have
    # exactly obs1 but at position i we have the predicted state i steps
    # past obs1.
    obs_hns = []
    # the action taken at the i-th step above
    acs_hna = []
    # after playing the corresponding action in the corresponding state
    # we get this reward
    rew_hn = []

    # We could use tf.while_loop here, but TF actually handles a
    # static graph much better. Since we know h ahead of time we should use
    # it.

    curr_ob_ns = obs1_ns
    for _ in range(h):
        obs_hns.append(curr_ob_ns)
        ac_na = actor.tf_target_action(curr_ob_ns)
        acs_hna.append(ac_na)
        next_ob_ns = dynamics.predict_tf(curr_ob_ns, ac_na)
        curr_reward_n = reward_fn(curr_ob_ns, ac_na, next_ob_ns)
        rew_hn.append(curr_reward_n)
        curr_ob_ns = next_ob_ns

    # final_ob_ns should be the final state resulting from playing
    # acs_hna[h-1] on obs_hns[h-1]
    final_ob_ns = curr_ob_ns
    final_ac_na = actor.tf_target_action(final_ob_ns)
    final_Q_n = critic.tf_target_critic(final_ob_ns, final_ac_na)

    # we accumulate error in reverse, but we can do this in a single
    # op now and rely on built-in scan operations
    # it's sad that I still have to do this in 2018 to get 1.5x speedup
    n = tf.shape(obs_hns)[1]
    a = env_info.ac_dim()
    s = env_info.ob_dim()
    all_Q_hn = tf.reshape(critic.tf_critic(
        tf.reshape(obs_hns, [-1, s]),
        tf.reshape(acs_hna, [-1, a])), [h, n])
    next_Q_n = final_Q_n
    accum_loss = 0.
    for t in reversed(range(h)):
        target_Q_n = rew_hn[t] + discount * next_Q_n
        curr_Q_n = all_Q_hn[t]
        # the dynamics model doesn't predict terminal states
        # so we only need to remove the terminal states from
        # the batch
        weights = 1.0 - terminals1_n
        curr_residual_loss = tf.losses.mean_squared_error(
            target_Q_n, curr_Q_n, weights=weights)
        if not flags().ddpg.drop_tdk:
            accum_loss += curr_residual_loss
        next_Q_n = target_Q_n

    # compute the full-trajectory TD-h error on obs0 now
    target_Q_n = rew0_n + discount * (1 - terminals1_n) * next_Q_n
    curr_Q_n = critic.tf_critic(obs0_ns, acs0_na)
    loss = accum_loss + tf.losses.mean_squared_error(
        target_Q_n, curr_Q_n)

    return loss
