"""Denoising autoencoder"""


import tensorflow as tf


class DADynamicsModel:

    def __init__(self, n_components, corr_fraction=0.3, lr=1e-2):
        # lr should go in fit or train
        self.lr = lr
        self.n_components = n_components
        self.corr_fraction = corr_fraction

    def fit(self, data, num_epochs=10):
        n_features = data.shape[1]
        self._build_model(n_features)

        with tf.Session() as session:
            for i in range(num_epochs):
                for batch in gen_batches(data):
                    session.run(self.train, feed_dict={
                        self.x_: batch
                    })


    def _build_model(self, n_features):

        # placeholder for inputs
        self.x_ = tf.placeholder('float', (None, n_features), name='input')
        self.x_corr_ = tf.placeholder('float', (None, n_features),
                                      name='input-corrupted')

        # corrupt inputs
        random = tf.random_uniform(self.x_.shape)
        mask = tf.less(random, self.corr_fraction)
        self.x_corr = tf.boolean_mask(self.x_, mask)

        # initialize variables
        self.W_ = tf.get_variable(
            "W", shape=(n_features, self.n_components),
            initializer=tf.contrib.layers.xavier_initializer())
        self.be_ = tf.get_variable(
            "bias_encoder", shape=(self.n_components,),
            initializer=tf.zeros_initializer())
        self.bd_ = tf.get_variable(
            "bias_decoder", shape=(n_features,),
            initializer=tf.zeros_initializer())

        # create encode layer
        self.encoder = tf.nn.sigmoid(tf.matmul(
            self.x_corr_, self.W_) + self.be_)

        # create decode layer
        self.decoder = tf.nn.sigmoid(tf.matmul(
            self.encoder, tf.transpose(self.W_)) + self.bd_)

        # define loss function
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.x_ - self.decoder)))

        # define train function
        self.train = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)

    def encode(self, x):
        with tf.Session() as session:
            session.run(self.encoder, feed_dict={self.x_: x})

    def decode(self, x):
        #TODO: add support for decode
        pass


def gen_batches(data, batch_size):
    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]