"""Denoising autoencoder"""

import distutils.util

import tensorflow as tf

from context import flags
import env_info
from log import debug
import reporter
from tfnode import TFNode


class DADynamicsModel(TFNode):
    """Stationary neural-network-based dynamics model."""

    def __init__(self, norm, n_components=100, corr_fraction=0.3, lr=1e-2):
        dyn_flags = flags().dynamics
        ob_dim, ac_dim = env_info.ob_dim(), env_info.ac_dim()

        self._norm = norm
        self._input_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'dynamics_input_state')
        self._input_action_ph_na = tf.placeholder(
            tf.float32, [None, ac_dim], 'dynamics_input_action')
        self._next_state_ph_ns = tf.placeholder(
            tf.float32, [None, ob_dim], 'true_next_state_diff')

        # only used if dyn_bn is set to true
        self._dyn_training = tf.placeholder_with_default(
            False, [], 'dynamics_dyn_training_mode')

        # TODO: move to flags
        self.lr = lr
        self.n_components = n_components
        self.corr_fraction = corr_fraction
        self.encoder = None
        self.decoder = None

        # pred_states_ns, deltas_ns = self._predict_tf(
        #     self._input_state_ph_ns, self._input_action_ph_na)
        # true_deltas_ns = self._norm.norm_delta(
        #     self._next_state_ph_ns - self._input_state_ph_ns)
        pred_states_ns, _ = self._predict_tf(
            self._input_state_ph_ns, self._input_action_ph_na)

        # train_mse = tf.losses.mean_squared_error(
        #     true_deltas_ns, deltas_ns)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # self._dyn_loss = reg_loss + train_mse

        self._dyn_loss = tf.sqrt(tf.reduce_mean(tf.square(
            self._input_state_ph_ns - pred_states_ns)))

        self._one_step_pred_error = tf.losses.mean_squared_error(
            self._next_state_ph_ns, pred_states_ns)
        with tf.control_dependencies(update_ops):
            self._update_op = tf.train.AdamOptimizer(
                dyn_flags.dyn_learning_rate).minimize(self._dyn_loss)

        #super().__init__('dynamics', dyn_flags.restore_dynamics)

    def fit(self, data, timesteps):
        """
        Fit the dynamics to the given dataset of transitions, given that
        we saw a certain number of new timesteps
        """
        if not data.size or not timesteps:
            return
        if data.size < flags().dynamics.dyn_min_buf_size:
            return

        # I actually tried out the tf.contrib.train.Dataset API, and
        # it was *slower* than this feed_dict method. I figure that the
        # data throughput per batch is small enough (since the data is so
        # simple) that up-front shuffling as performed here is probably
        # saving us more time.
        nbatches = flags().dynamics.nbatches(timesteps)
        batch_size = flags().dynamics.dyn_batch_size
        for i, batch in enumerate(data.sample_many(nbatches, batch_size), 1):
            obs, next_obs, _, acs, _ = batch
            self._update_op.run(feed_dict={
                self._input_state_ph_ns: obs,
                self._input_action_ph_na: acs,
                self._next_state_ph_ns: next_obs,
                self._dyn_training: True
            })
            if (i % max(nbatches // 10, 1)) == 0:
                dyn_loss, one_step_error = tf.get_default_session().run(
                    [self._dyn_loss, self._one_step_pred_error],
                    feed_dict={
                        self._input_state_ph_ns: obs,
                        self._input_action_ph_na: acs,
                        self._next_state_ph_ns: next_obs,
                    })
                fmt = '{: ' + str(len(str(nbatches))) + 'd}'
                debug('dynamics ' + fmt + ' of ' + fmt + ' batches - '
                      'dyn loss {:.4g} one-step mse {:.4g} ',
                      i, nbatches, dyn_loss, one_step_error)

    def _predict_tf(self, states, actions):
        # state_action_pair = tf.concat([
        #     self._norm.norm_obs(states),
        #     self._norm.norm_acs(actions)], axis=1)


        # standard_predicted_state_diff_ns = self.build_da(state_action_pair)
        # predicted_state_diff_ns = self._norm.denorm_delta(
        #     standard_predicted_state_diff_ns)
        # return states + predicted_state_diff_ns, \
        #     standard_predicted_state_diff_ns

        state_norm = self._norm.norm_obs(states)
        predicted_state_ns = self.build_da(state_norm, reuse=tf.AUTO_REUSE)
        return predicted_state_ns, ()

    def predict_tf(self, states, actions):
        """
        Predict the next state given the current state and action.

        Assumes input TF tensors are batched states and actions.
        Outputs corresponding predicted next state as a TF tensor.
        """
        return self._predict_tf(states, actions)[0]

    def _activation_norm(self, inputs):
        outputs = inputs
        if flags().dynamics.dyn_bn:
            outputs = tf.layers.batch_normalization(
                outputs, training=self._dyn_training)
        if flags().dynamics.dyn_dropout:
            outputs = tf.nn.dropout(
                outputs,
                tf.maximum(
                    1 - tf.to_float(self._dyn_training),
                    flags().dynamics.dyn_dropout))
        return outputs

    def evaluate(self, data, prefix='dynamics'):
        """report dynamics metrics"""
        if not data.size:
            return
        prefix = prefix + ('/' if prefix else '')
        batch_size = flags().dynamics.dyn_batch_size * 10
        batch = next(data.sample_many(1, batch_size))
        obs, next_obs, _, acs, _ = batch
        dyn_loss, one_step_error = tf.get_default_session().run(
            [self._dyn_loss, self._one_step_pred_error],
            feed_dict={
                self._input_state_ph_ns: obs,
                self._input_action_ph_na: acs,
                self._next_state_ph_ns: next_obs,
            })
        reporter.add_summary(prefix + 'one-step mse', one_step_error)
        reporter.add_summary(prefix + 'loss', dyn_loss)

    def build_da(self, input_placeholder, scope='dae', reuse=None):
        """Build a denoising autoencoder. Does not corrupt inputs"""
        with tf.variable_scope(scope, reuse=reuse):
            out = input_placeholder
            output_size = out.shape[1]
            with tf.variable_scope('encode', reuse=reuse):
                out = tf.layers.dense(out, self.n_components, activation=tf.nn.sigmoid)
            with tf.variable_scope('decode', reuse=reuse):
                out = tf.layers.dense(out, output_size, activation=tf.nn.sigmoid)
            return out
            # n_features = out.shape[1]
            #
            # # placeholder for inputs
            # self.x_ = tf.placeholder('float', (None, n_features), name='input')
            # self.x_corr_ = tf.placeholder('float', (None, n_features),
            #                               name='input-corrupted')
            #
            # # corrupt inputs
            # random = tf.random_uniform(self.x_.shape)
            # mask = tf.less(random, self.corr_fraction)
            # self.x_corr = tf.boolean_mask(self.x_, mask)
            #
            # # initialize variables
            # self.W_ = tf.get_variable(
            #     "W", shape=(n_features, self.n_components),
            #     initializer=tf.contrib.layers.xavier_initializer())
            # self.be_ = tf.get_variable(
            #     "bias_encoder", shape=(self.n_components,),
            #     initializer=tf.zeros_initializer())
            # self.bd_ = tf.get_variable(
            #     "bias_decoder", shape=(n_features,),
            #     initializer=tf.zeros_initializer())
            #
            # # create encode layer
            # self.encoder = tf.nn.sigmoid(tf.matmul(
            #     self.x_corr_, self.W_) + self.be_)
            #
            # # create decode layer
            # self.decoder = tf.nn.sigmoid(tf.matmul(
            #     self.encoder, tf.transpose(self.W_)) + self.bd_)
            #
            # return self.decoder





# import tensorflow as tf
# import numpy as np
#
#
# class DADynamicsModel:
#
#     def __init__(self, n_components, corr_fraction=0.3, lr=1e-2):
#         # lr should go in fit or train
#         self.lr = lr
#         self.n_components = n_components
#         self.corr_fraction = corr_fraction
#
#     def fit(self, data, num_epochs=10):
#         n_features = data.shape[1]
#         self._build_model(n_features)
#
#         with tf.Session() as session:
#             for i in range(num_epochs):
#                 for batch in gen_batches(data):
#                     session.run(self.train, feed_dict={
#                         self.x_: batch
#                     })
#
#     def _build_model(self, n_features):
#
#         # placeholder for inputs
#         self.x_ = tf.placeholder('float', (None, n_features), name='input')
#         self.x_corr_ = tf.placeholder('float', (None, n_features),
#                                       name='input-corrupted')
#
#         # corrupt inputs
#         random = tf.random_uniform(self.x_.shape)
#         mask = tf.less(random, self.corr_fraction)
#         self.x_corr = tf.boolean_mask(self.x_, mask)
#
#         # initialize variables
#         self.W_ = tf.get_variable(
#             "W", shape=(n_features, self.n_components),
#             initializer=tf.contrib.layers.xavier_initializer())
#         self.be_ = tf.get_variable(
#             "bias_encoder", shape=(self.n_components,),
#             initializer=tf.zeros_initializer())
#         self.bd_ = tf.get_variable(
#             "bias_decoder", shape=(n_features,),
#             initializer=tf.zeros_initializer())
#
#         # create encode layer
#         self.encoder = tf.nn.sigmoid(tf.matmul(
#             self.x_corr_, self.W_) + self.be_)
#
#         # create decode layer
#         self.decoder = tf.nn.sigmoid(tf.matmul(
#             self.encoder, tf.transpose(self.W_)) + self.bd_)
#
#         # define loss function
#         self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.x_ - self.decoder)))
#
#         # define train function
#         self.train = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
#
#     def encode(self, x):
#         with tf.Session() as session:
#             session.run(self.encoder, feed_dict={self.x_: x})
#
#     def decode(self, x):
#         #TODO: add support for decode
#         pass
#
#
# def gen_batches(data, batch_size):
#     for i in range(0, data.shape[0], batch_size):
#         yield data[i:i+batch_size]
#
#
# if __name__ == '__main__':
#     data = np.random.random((100, 1000))
