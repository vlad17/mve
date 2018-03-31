import tensorflow.contrib.layers as layers
import deepq
from deepq.simple import run_experiment


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

if __name__ == '__main__':
    run_experiment(model, horizon=int(sys.argv[1]))
