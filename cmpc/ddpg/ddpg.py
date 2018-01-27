"""DDPG training."""

import tensorflow as tf
import numpy as np

from context import flags
import env_info
from log import debug
import reporter
from sample import sample_venv
from tf_reporter import TFReporter
from qvalues import qvals, offline_oracle_q, oracle_q
from utils import scale_from_box, as_controller, flatgrad


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

    def __init__(self, actor, critic, discount=0.99, scope='ddpg',
                 actor_lr=1e-3, critic_lr=1e-3, explore_stddev=0.2):

        self._reporter = TFReporter()

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
                'only oracle estimator handled, got {}'.format(
                    flags().ddpg.mixture_estimator)
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
        next_Q_n = critic.tf_target_critic(
            self.obs1_ph_ns, actor.tf_target_action(self.obs1_ph_ns))
        if flags().ddpg.q_target_mixture:
            assert flags().ddpg.mixture_estimator == 'oracle', \
                'only oracle estimator handled, got {}'.format(
                    flags().ddpg.mixture_estimator)
            # h-step observations
            h = flags().ddpg.model_horizon
            debug('using oracle Q estimator with {} steps as target critic', h)
            nenvs = flags().ddpg.learner_batch_size
            self._oracle_q_target_venv = env_info.make_venv(nenvs)
            self._oracle_q_target_venv.reset()
            self._next_Q_ph_n = tf.placeholder(
                tf.float32, shape=[None])
            next_Q_n = self._next_Q_ph_n
        target_Q_n = self.rewards_ph_n + (1. - self.terminals1_ph_n) * (
            discount * next_Q_n)
        self._reporter.stats('target Q', target_Q_n)

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self._critic_loss = tf.losses.mean_squared_error(
            target_Q_n, current_Q_n) + reg_loss
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
                actor.tf_target_update(flags().ddpg.actor_decay),
                critic.tf_target_update(flags().ddpg.critic_decay))

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
        self._venv = env_info.make_venv(10)

    def _evaluate(self):
        # runs out-of-band trials for less noise performance evaluation
        paths = sample_venv(self._venv, as_controller(self._actor.target_act))
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

        paths = sample_venv(self._venv, as_controller(self._actor.act))
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
        if flags().ddpg.q_target_mixture:
            model_expanded_Q = self._oracle_Q(next_obs)
            feed_dict[self._next_Q_ph_n] = model_expanded_Q
        if flags().ddpg.actor_critic_mixture:
            self._oracle_expand_actions(feed_dict)
        return feed_dict

    def train(self, data, nbatches, batch_size):
        """Run nbatches training iterations of DDPG"""
        batches = data.sample_many(nbatches, batch_size)
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

        if data.size:
            batch = self._sample(next(data.sample_many(1, batch_size)))
            self._reporter.report(batch)

        self._evaluate()

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
