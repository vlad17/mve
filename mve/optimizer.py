"""
Routines for unconstrained optimization problems that have underlying
TensorFlow functionals (e.g., neural networks)
"""

import tensorflow as tf
import numpy as np

import log


class TFOptimizer:
    """
    An optimizer instance should be tied to a particular function object.

    Changing data referenced by the function object is allowed between
    optimize() calls.

    All optimizations are minimizations.

    The TFOptimizer might have internal state both in python and TF
    processes, but so long as variables referenced by the internal
    function object and minimize() is called by at most one thread
    at a time, this class is effectively stateless.

    It may not be pure if there is stochasticity in the computation.
    """

    def minimize(self, feed_dict, *args):
        """
        Optimize the function associated with this instance starting at the
        given initial points. Invoke all TF ops with the parameter feed dict.
        """
        raise NotImplementedError


class AdamOptimizer(TFOptimizer):
    """
    Use up to max_steps Adam steps, terminating early if improvement
    falls below the given tolerance.

    f should accept len(arg_dims) TF tensors of the corresponding shape
    and output the function value to optimize for those tensors.

    f shouldn't have any weird shit in the computation graph that messes
    up its TF autograd.
    """

    def __init__(self, f, arg_dims, max_steps, tol, lr):
        self._args = [
            tf.Variable(
                np.zeros(shape), name='adam-arg{}'.format(i), dtype=tf.float32)
            for i, shape in enumerate(arg_dims)]
        opt = tf.train.AdamOptimizer(lr)

        def _body(i, _, prev_objective):
            with tf.control_dependencies([prev_objective]):
                objective_to_optimize = f(*self._args)
                update_op = opt.minimize(
                    objective_to_optimize, var_list=self._args)
            with tf.control_dependencies([update_op]):
                new_objective = f(*self._args)
                return [i + 1, prev_objective - new_objective, new_objective]

        def _cond(i, improvement, _):
            return tf.logical_and(
                i < max_steps,
                improvement > tol)

        self._args_ph = [tf.placeholder(tf.float32, shape)
                         for shape in arg_dims]
        args_assign = tf.group(*(tf.assign(arg, ph) for arg, ph
                                 in zip(self._args, self._args_ph)))

        with tf.control_dependencies([args_assign]):
            loop_vars = [0, tol + 1, f(*self._args)]
            self._optimize = tf.while_loop(
                _cond, _body, loop_vars, back_prop=False)

    def minimize(self, feed_dict, *args):
        assert len(args) == len(self._args), (len(args), len(self._args))
        feed = {ph: arg for ph, arg in zip(self._args_ph, args)}
        feed.update(feed_dict)
        steps, diff, _ = tf.get_default_session().run(self._optimize, feed)
        log.debug('       ---> primal steps taken {} final improvement {}',
                  steps, diff)
        return tf.get_default_session().run(self._args, feed_dict)
