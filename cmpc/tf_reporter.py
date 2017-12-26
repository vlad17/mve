"""
Lightweight class that keeps track of variables and TF tensors that
should be reported. This is useful for large classes where additional
debug logic would get hectic.

This is just a thin wrapper around the reporter module.
"""

import numpy as np
import tensorflow as tf

import reporter


class TFReporter:
    """
    This class does not own or manage any of the tensors it is given. All it
    does is save intermediate values for debugging purposes, which
    are reported on-demand. Note that this class may create intermediate
    operations so the methods {scalar, stats, weights} should be called
    before the graph is finalized. Finally, weights are reported but hidden
    when the reporter prints its summaries. The others are not hidden.

    Exapmle use:

    class TFSquare:
        def __init__(self):
            self._reporter = TFReporter()
            self._input_ph = tf.placeholder(tf.float32, [])
            square = tf.square(self._input_ph)
            self._reporter.scalar('intermediate square', square)
            self._output = square + 1

        def compute_output(self, x):
            fd = {self._input_ph: x}
            self._reporter.report(fd)
            return tf.get_default_session().run(self._output, fd)

    It seems useless for the above case, but imagine many intermediate
    tensors like "square", which would all have to be attributes of TFSquare.
    """

    def __init__(self):
        self._scalars = []
        self._statistics = []
        self._weights = []

    def scalar(self, name, tfscalar):
        """Save an intermediate scalar to report."""
        self._scalars.append((name, tfscalar))

    def stats(self, name, tftensor):
        """Save the statistics for the elements of the flattened tftensor."""
        flat = tf.reshape(tftensor, [-1])
        self._statistics.append((name, flat))

    def weights(self, name, weights):
        """Save the average magnitude of network weights."""
        flat = tf.reshape(weights, [-1])
        ave_magnitude = tf.reduce_mean(tf.abs(flat))
        self._weights.append((name, ave_magnitude))

    def grads(self, name, opt, loss, variables):
        """Save the gradients made by the given optimizer."""
        grad_l2, grad_linf = _flatgrad_norms(opt, loss, variables)
        self.scalar(name + '/2 norm', grad_l2)
        self.scalar(name + '/inf norm', grad_linf)

    def report(self, feed_dict):
        """Report the value of the tensors given the feed."""
        scalar_names, scalars = zip(*self._scalars)
        stats_names, stats = zip(*self._statistics)
        weights_names, weights = zip(*self._weights)
        scalars, stats, weights = tf.get_default_session().run(
            [scalars, stats, weights], feed_dict)

        for name, value in zip(scalar_names, scalars):
            reporter.add_summary(name, value)

        for name, value in zip(stats_names, stats):
            reporter.add_summary_statistics(name, value)

        for name, value in zip(weights_names, weights):
            reporter.add_summary(name, value, hide=True)


def _flatgrad_norms(opt, loss, variables):
    grads_and_vars = opt.compute_gradients(loss, var_list=variables)
    grads = [grad for grad, var in grads_and_vars if var is not None]
    flats = [tf.reshape(x, [-1]) for x in grads]
    grad = tf.concat(flats, axis=0)
    l2 = tf.norm(grad, ord=2)
    linf = tf.norm(grad, ord=np.inf)
    return l2, linf
