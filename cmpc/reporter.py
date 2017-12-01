"""
A thread-local summary reporter. Automatically logs info to a TensorFlow
summary file, which can be viewed with TensorBoard.
"""

from contextlib import contextmanager
import threading

import numpy as np
import tensorflow as tf
from terminaltables import SingleTable

from utils import create_tf_session

@contextmanager
def report_hook(hook):
    """Add an additional reporting hook"""
    _hooks().append(hook)
    yield
    _hooks().pop()

@contextmanager
def create(logdir, verbose):
    """
    Create a reporter for a certain context. The reporter will log
    summaries to TensorFlow tf.Summary files.

    If verbose, the reporter will print summary information to the terminal.
    """
    reporter = _Reporter(logdir, verbose)
    with reporter.as_default():
        yield
    reporter.close()

def add_summary(name, value):
    """Add a known floating-point value summary to the current reporter"""
    _reporter().add_summary(name, value)

def add_summary_statistics(name, values):
    """
    Add a list of floating values, whose statistics are summarized to the
    current reporter.
    """
    _reporter().add_summary_statistics(name, values)

def advance_iteration():
    """
    Advance the iteration and print statistics as recorded by the current
    reported. The statistics are only printed if the reporter was specified
    as verbose.
    """
    _reporter().advance_iteration()

_thread_local = threading.local()

def _hooks():
    if not hasattr(_thread_local, 'hooks'):
        _thread_local.hooks = []
    return _thread_local.hooks

def _reporters():
    if not hasattr(_thread_local, 'reporters'):
        _thread_local.reporters = []
    return _thread_local.reporters

def _reporter():
    """Returns the default summary reporter within the current context."""
    return _reporters()[-1]

class _Summarize:
    _lock = threading.Lock()
    _live_session = None
    _graph = None
    _hist_ph = None
    _hist_op = None

    @staticmethod
    def _session_unlocked():
        if _Summarize._live_session is not None:
            return _Summarize._live_session
        _Summarize._graph = tf.Graph()
        with _Summarize._graph.as_default():
            _Summarize._live_session = create_tf_session(gpu=False)
            _Summarize._hist_ph = tf.placeholder(tf.float32)
            _Summarize._hist_op = tf.summary.histogram(
                'hist', _Summarize._hist_ph)
        _Summarize._graph.finalize()
        return _Summarize._live_session

    @staticmethod
    def hist(name, values):
        """Generate a TF histogram with the specified name and values"""
        with _Summarize._lock:
            sess = _Summarize._session_unlocked()
        summary_pb_bytes = sess.run(
            _Summarize._hist_op, feed_dict={_Summarize._hist_ph: values})
        summary = tf.Summary()
        summary.ParseFromString(summary_pb_bytes)
        values = summary.value
        assert len(values) == 1, values
        values[0].tag = name
        return summary

    @staticmethod
    def value(name, value):
        """Generate a TF simple float value summary"""
        return tf.Summary(value=[
            tf.Summary.Value(
                tag=name, simple_value=value)])

def _floatprint(f):
    return '{:.4g}'.format(f)

class _Reporter:
    """
    Manages logging diagnostic information about the application.
    Keeps track of the current iteration.
    """

    def __init__(self, logdir, verbose):
        self._writer = tf.summary.FileWriter(logdir)
        self._verbose = verbose
        self._global_step = 1
        self._latest_summaries = {}
        self._latest_statistics = {}

    def add_summary(self, name, value):
        """Add a known floating-point value summary"""
        self._latest_summaries[name] = value

    def add_summary_statistics(self, name, values):
        """Add a list of floating values, whose statistics are summarized"""
        self._latest_statistics[name] = values

    def advance_iteration(self):
        """Increment the global step and print the previous step statistics"""
        self._write_summaries()
        if self._verbose:
            self._print_table()

        for hook in _hooks():
            hook(self._latest_summaries, self._latest_statistics)

        self._global_step += 1
        self._latest_summaries = {}
        self._latest_statistics = {}

    def _print_table(self):
        data = [['summary', 'value', 'min', 'mean', 'max', 'std']]
        data.append(['iteration', self._global_step, '', '', '', ''])
        for name, value in self._latest_summaries.items():
            data.append([name, _floatprint(value), '', '', '', ''])
        for name, values in self._latest_statistics.items():
            values = [statistic(values) for statistic in
                      [np.min, np.mean, np.max, np.std]]
            values = [_floatprint(value) for value in values]
            data.append([name, ''] + values)
        table = SingleTable(data)
        table.inner_column_border = False
        print(table.table)

    def _write_summaries(self):
        for name, value in self._latest_summaries.items():
            self._writer.add_summary(
                _Summarize.value(name, value),
                self._global_step)

        for name, values in self._latest_statistics.items():
            self._writer.add_summary(
                _Summarize.hist(name + ' hist', values),
                self._global_step)
            self._writer.add_summary(
                _Summarize.value(name + ' mean', np.mean(values)),
                self._global_step)

    @contextmanager
    def as_default(self):
        """Specify a report to log summaries to within a thread context"""
        _reporters().append(self)
        yield
        _reporters().pop()

    def close(self):
        """Close the internal file writer"""
        self._writer.close()
