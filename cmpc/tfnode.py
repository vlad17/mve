"""
A TFNode is a savable component in a TF graph.

It corresponds to the variables associated with a subgraph of a TF graph.
"""

import os

import tensorflow as tf

from context import context
from log import debug
import reporter


class TFNode:
    """
    A TFNode derived class should follow these rules:

      * It should follow a consistent variable-naming procedure (instances
        created with the same parameters should have the same-named variables)
      * All TFNode graph components (not just its variables, but also its
        operations and placeholders) should be created at __init__ time.
      * If a TFNode method accepts TF tensors and returns a TF tensor,
        this method should be prefixed with a 'tf_' to indicate
        that it modifies the graph.

    A TFNode saves and restores its associated variables upon request.
    Following the rules above, a compliant subclass wishing to save
    one of it's variables might have a constructor like the following:

    class A(TFNode):
        def __init__(self, restore_path):
            tf.variable_scope('scope_A'):
                self._var = tf.get_variable('var_to_save', [3])
            super().__init__('scope_A', restore_path)

    The first argument should be a string and will correspond to the
    variables returned by this collection:

        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scope_A')

    Upon calling A.restore(), the TFNode will recover values from the
    restore path (according to their names) into A's self._var variable.
    Note that restore() is not called automatically (since initialization
    might clobber the restored value).

    A.save(step) saves the variables in the logging directory, under
    the 'checkpoints' subdirectory,
    after appending the step to the file name, which is the scope
    name followed by '.ckpt'.

    If restore_path is None then restore() is a no-op.
    """

    def __init__(self, scope=None, restore_path=None):
        if restore_path is not None:
            assert scope is not None, scope

        if scope is not None:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
            self._saver = tf.train.Saver(
                var_list=variables,
                save_relative_paths=True,
                pad_step_number=True)
            _add_to_graph(self)
        else:
            self._saver = None
        self._restore = restore_path
        self._scope = scope

    def custom_init(self):
        """
        This method is automatically invoked by the base class in the case
        that the TFNode was not restored. While all global variables have
        been default-initialized by this point, a subclass of TFNode might
        have a custom initialization procedure that it might invoke here.

        By default, no additional custom intialization is assumed.
        """
        pass

    def restore(self):
        """Restore TFNode variables from the given restore path."""
        if self._saver and self._restore is not None:
            self._saver.restore(tf.get_default_session(), self._restore)
            debug('restored {} from {}', self._scope, self._restore)
        else:
            self.custom_init()

    def save(self, step):
        """
        Save variables associated with this TFNode if a save path was specified
        with the suffix equal to the given step.
        """
        if self._saver:
            ckpt_dir = os.path.join(
                reporter.logging_directory(), 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            save = os.path.join(ckpt_dir, self._scope + '.ckpt')
            self._saver.save(tf.get_default_session(), save, step)


def _add_to_graph(tfnode):
    _graph_tfnodes().append(tfnode)


def _graph_tfnodes():
    if context().tfnodes is None:
        context().tfnodes = []
    return context().tfnodes


def restore_all():
    """
    Restore all TFNodes constructed in the current default graph using
    the default session.
    """
    for tfnode in _graph_tfnodes():
        tfnode.restore()


def save_all(step):
    """
    Save all TFNode states, marking the current step as specified.
    """
    for tfnode in _graph_tfnodes():
        tfnode.save(step)
