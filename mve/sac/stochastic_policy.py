"""A stochastic policy"""

import numpy as np
import tensorflow as tf

from context import flags
import env_info
from memory import unscale_acs
from utils import build_mlp, trainable_vars


class StochasticPolicy:
    """
    Provides tensorflow interface to sampling-with-density and acting
    according to it.
    """

    def tf_sample_action_with_log_prob(self, obs_ns):
        """Return a sample from the policy, along with its log probability."""
        raise NotImplementedError  # return sample, log prob

    def tf_greedy_action(self, obs_ns):
        """Return a sample from the greedy version of the policy."""
        raise NotImplementedError

    def act(self, obs_ns):
        """Eager version of sampling an action"""
        raise NotImplementedError

    def greedy_act(self, obs_ns):
        """Eager version of sampling a greedy action"""
        raise NotImplementedError

    def expectation(self, obs_ns, fn):
        """
        Given a function fn : (acs_na, log pi(acs_na)) -> reals
        which maps a TF batch of actions and their likelihoods (taken according
        to this stochastic policy), this returns a tensor which, when
        differentiated wrt the policy parameters,
        gives an estimator for the gradient of E[fn(A)], where A follows
        this stochastic policy. The expectation is wrt the state distribution
        an empirical sample of which should be given as the obs_ns argument
        here.
        """
        pass

    def tf_report(self, reporter, obs_ns):
        """
        Update a TF reporter with various statistics according to the given
        sample of states.
        """
        pass


class SquashedGaussianPolicy(StochasticPolicy):
    """
    This stochastic policy implements a unimodal multivariate
    Gaussian distribution with a diagonal covariance, with the both the mean
    and covariance conditional on the state. This generates
    a latent distribution, which then is then clipped for stability
    and squashed into a compact [-1, 1] interval.
    """

    def __init__(self, scope='sac/policy'):
        """
        Generates a diagonal gaussian policy which keeps its log standard
        deviations between the lower and upper bounds provided as arguments.
        """
        self._scope = scope
        self._obs_ph_ns = tf.placeholder(tf.float32, [None, env_info.ob_dim()])
        self._acs_na, _ = self.tf_sample_action_with_log_prob(self._obs_ph_ns)
        self._greedy_acs_na = self.tf_greedy_action(self._obs_ph_ns)
        self.variables = trainable_vars(self._scope)

    def _tf_mu_logstd(self, obs_ns):
        unclipped_mu_logstd_n2a = build_mlp(
            obs_ns, scope=self._scope,
            output_size=(env_info.ac_dim() * 2),
            n_layers=flags().sac.learner_depth,
            size=flags().sac.learner_width,
            activation=tf.nn.relu,
            reuse=tf.AUTO_REUSE)
        unclipped_mu_na = unclipped_mu_logstd_n2a[:, :env_info.ac_dim()]
        unclipped_logstd_na = unclipped_mu_logstd_n2a[:, env_info.ac_dim():]

        # What follows is black magic clipping for stability. These constants
        # are luckily not specific to any enviornment, since we are going to
        # squash outputs with tanh anyway. In turn, they're mostly determined
        # by the range of tanh and the machine accuracy of floats. However,
        # the logstd upper bound can be higher in this sense, but this was a
        # magic constant taken from Tuomas' code that I do not have the
        # confidence to mess with.
        #
        # TODO:
        # Consider adding a policy regularization term to encourage staying in
        # the active interval, e.g., add loss max(x - ub, 0) ** 2 or something.
        logstd_na = tf.clip_by_value(unclipped_logstd_na, -20, 2)
        mu_na = tf.clip_by_value(unclipped_mu_na, -5, 5)
        return mu_na, logstd_na

    @staticmethod
    def _tf_sample(mu_na, logstd_na):
        return tf.random_normal(tf.shape(mu_na)) * tf.exp(logstd_na) + mu_na

    @staticmethod
    def _tf_log_prob(x_na, mu_na, logstd_na):
        diffs_na = x_na - mu_na
        quadratic_n = tf.reduce_sum(
            (diffs_na * tf.exp(-logstd_na)) ** 2, axis=1)

        norm_factor_n = 2 * tf.reduce_sum(logstd_na, axis=1)
        norm_factor_n += tf.to_float(env_info.ac_dim()) * np.log(2 * np.pi)
        logprob_n = -0.5 * (quadratic_n + norm_factor_n)
        return logprob_n

    def tf_sample_action_with_log_prob(self, obs_ns):
        return self._tf_sample_action_with_log_prob(
            obs_ns, stop_gradient=True)

    def _tf_sample_action_with_log_prob(self, obs_ns, stop_gradient=True):
        # stop_gradient stops the contribution of the parameters to the
        # sampling directly.
        # when using the reparameterization trick, we *do* want the
        # Jacobian effect of the policy parameters on the sampling, so
        # we would have stop_gradient set to false. However for vanilla
        # policy gradient we don't want this on because it would not
        # correspond to performing a valid reinforce gradient estimation.

        # latent mean, logstd, and log prob
        mu_na, logstd_na = self._tf_mu_logstd(obs_ns)

        # stop the gradient for computing the sample log prob, the
        # parameters of the policy should only affect the log prob
        # gradient by their influence on the policy distribution.
        unbounded_sample_na = self._tf_sample(mu_na, logstd_na)
        if stop_gradient:
            unbounded_sample_na = tf.stop_gradient(unbounded_sample_na)

        unbounded_logprob_n = self._tf_log_prob(
            unbounded_sample_na, mu_na, logstd_na)
        # squash with tanh and scale
        logprob_n = self._scale_correction(
            unbounded_sample_na, unbounded_logprob_n)
        sample_na = self._scale(unbounded_sample_na)
        return sample_na, logprob_n

    def _scale(self, unbounded_acs_na):
        unit_acs_na = tf.tanh(unbounded_acs_na)
        return unscale_acs(unit_acs_na)

    @staticmethod
    def _scale_correction(unbounded_acs_na, unbounded_log_prob_n):
        # chain rule to correct for density after modifications
        log_prob_n = unbounded_log_prob_n
        # original paper has 1-tanh**2, we use sech**2 for stability
        # log(sech**2) == 2 * (log 2 - log (e^x + e^-x))
        #              == 2 * (log 2 - |x| - log (1 + e^(-2|x|)))
        abs_na = tf.abs(unbounded_acs_na)
        log_prob_n -= 2 * tf.reduce_sum(
            -1 * abs_na - tf.log1p(tf.exp(-2 * abs_na)) + np.log(2), axis=1)
        # scale to unit action
        log_prob_n -= np.log(0.5) * env_info.ac_dim()
        space = env_info.ac_space()
        # scale to box of action space
        log_prob_n -= np.sum(np.log(space.high - space.low))
        return log_prob_n

    def tf_greedy_action(self, obs_ns):
        mu_na, _ = self._tf_mu_logstd(obs_ns)
        return self._scale(mu_na)

    def act(self, obs_ns):
        return tf.get_default_session().run(self._acs_na, feed_dict={
            self._obs_ph_ns: obs_ns})

    def greedy_act(self, obs_ns):
        return tf.get_default_session().run(self._greedy_acs_na, feed_dict={
            self._obs_ph_ns: obs_ns})

    def tf_report(self, reporter, obs_ns):
        mu_na, logstd_na = self._tf_mu_logstd(obs_ns)
        reporter.stats('logstd', logstd_na)
        reporter.stats('abs(mu)', tf.abs(mu_na))
        acs_na, log_pi_n = self.tf_sample_action_with_log_prob(obs_ns)
        reporter.stats('log pi', log_pi_n)
        reporter.stats('abs(scaled sample acs)', tf.abs(acs_na))

    def expectation(self, obs_ns, fn):
        if flags().sac.reparameterization_trick:
            # for the reparameterization trick, we do not stop the gradient
            # when sampling from the latent distribution, intentionally.
            # This way we can use gradient information of fn to improve the
            # parameters
            onpol_act_na, log_prob_acs_n = (
                self._tf_sample_action_with_log_prob(
                    obs_ns, stop_gradient=False))
            return tf.reduce_mean(fn(onpol_act_na, log_prob_acs_n))
        # else use the reinforce trick
        onpol_act_na, log_prob_acs_n = self._tf_sample_action_with_log_prob(
            obs_ns, stop_gradient=True)
        return tf.reduce_mean(
            log_prob_acs_n * tf.stop_gradient(
                fn(onpol_act_na, log_prob_acs_n)))
