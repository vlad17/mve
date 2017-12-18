"""
An opaque dataset that is tied to a runtime (volatile) dataset. This
class recovers and saves that dataset's state directly.
"""

import tensorflow as tf

from flags import Flags, ArgSpec
from tfnode import TFNode
from utils import timeit


class PersistableDatasetFlags(Flags):
    """
    Optionally specify where to restore the replay buffer from.
    """

    def __init__(self):
        arguments = [ArgSpec(
            name='restore_buffer',
            default=None,
            type=str,
            help='restore replay buffer from the given path')]
        super().__init__('persistable_dataset', 'replay buffer persistance',
                         arguments)


class _PersistableDataset(TFNode):
    """
    Modifies a dataset's state upon restore() or save().
    """

    def __init__(self, dataset, flags):
        self._transition_attrs = [
            'obs', 'next_obs', 'rewards', 'acs', 'terminals',
            'planned_acs', 'planned_obs']
        transition_attr_shapes = [
            getattr(dataset, attr).shape for attr in self._transition_attrs]
        transition_attr_shapes = [
            (dataset.maxlen,) + shape[1:] for shape in transition_attr_shapes]
        with tf.device('/cpu:0'):
            with tf.variable_scope('persistable_dataset', reuse=False):
                tf_len = tf.get_variable(
                    'len', trainable=False, initializer=0, dtype=tf.int32)
                tf_transitions = {
                    attr: tf.get_variable(
                        attr, trainable=False, shape=shape, dtype=tf.float32)
                    for attr, shape
                    in zip(self._transition_attrs, transition_attr_shapes)}
            len_ph = tf.placeholder(tf.int32, [])
            assign_len = tf.assign(tf_len, len_ph)
            self._set_len = lambda newlen: tf.get_default_session().run(
                assign_len, feed_dict={len_ph: newlen})
            self._transitions = {
                attr: var[:tf_len] for attr, var in tf_transitions.items()}
            transitions_ph = {attr: tf.placeholder(tf.float32)
                              for attr in self._transition_attrs}
            update_transions_op = tf.group(*[
                tf.scatter_update(var, tf.range(tf_len), transitions_ph[attr])
                for attr, var in tf_transitions.items()])
            self._update_trans = lambda data: tf.get_default_session().run(
                update_transions_op, feed_dict={
                    ph: getattr(data, attr)
                    for attr, ph in transitions_ph.items()})
        self._dataset = dataset
        super().__init__('persistable_dataset', flags.restore_buffer)

    def save(self, step):
        with timeit('syncing ringbuffer'):
            self._set_len(self._dataset.size)
            self._update_trans(self._dataset)
        super().save(step)

    def restore(self):
        super().restore()
        transitions = {
            attr: tf.get_default_session().run(tf_transition)
            for attr, tf_transition in self._transitions.items()}
        self._dataset.set_state(transitions)


def add_dataset_to_persistance_registry(dataset, flags):
    """
    Given a dataset.Dataset and PersistableDatasetFlags instance, adds a
    persistable version of the the dataset to the TF graph-based
    restore registry.
    """
    # TFNode constructor implicitly saves to registry
    _PersistableDataset(dataset, flags)
