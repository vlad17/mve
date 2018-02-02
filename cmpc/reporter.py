"""
A thread-local summary reporter. Automatically logs info to a TensorFlow
summary file, which can be viewed with TensorBoard.
"""

from contextlib import contextmanager
import threading

import numpy as np
import tensorflow as tf

from context import context
from utils import create_tf_config, print_table, timesteps


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
    assert context().reporter is None, 'reporter already exists'
    context().reporter = _Reporter(logdir, verbose)
    yield
    context().reporter.close()
    context().reporter = None


def add_summary(name, value, hide=False):
    """
    Add a known floating-point value summary to the current reporter.
    If hide is set to true, this summary is not printed during
    advance().
    """
    context().reporter.add_summary(name, value, hide)


def add_summary_statistics(name, values, hide=False):
    """
    Add a list of floating values, whose statistics are summarized to the
    current reporter.
    """
    context().reporter.add_summary_statistics(name, values, hide)


def advance_with_paths(paths):
    """
    Advance the iteration and print statistics as recorded by the current
    reported. The statistics are only printed if the reporter was specified
    as verbose.

    We advance by the number of timesteps and episodes taken in the paths
    list that was passed in.
    """
    ts = timesteps(paths)
    episodes = len(paths)
    context().reporter.advance(ts, episodes)


def advance_with_steps(steps):
    """
    We advance by the number of timesteps and episodes taken in the timesteps
    list that was passed in.
    """
    ts = sum(steps)
    episodes = len(steps)
    context().reporter.advance(ts, episodes)


def logging_directory():
    """
    Return the directory where the reporter is logging to, which
    may be useful for saving other files in.
    """
    return context().reporter.logdir


_thread_local = threading.local()


def _hooks():
    if not hasattr(_thread_local, 'hooks'):
        _thread_local.hooks = []
    return _thread_local.hooks


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
            _Summarize._live_session = tf.Session(
                config=create_tf_config(gpu=False))
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
        try:
            summary_pb_bytes = sess.run(
                _Summarize._hist_op, feed_dict={_Summarize._hist_ph: values})
        except tf.errors.InvalidArgumentError as e:
            msg = 'error during histogram generation for {}'.format(name)
            raise ValueError(msg) from e
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
    Keeps track of the current number of timesteps as the global step,
    and the total number of episodes so far.
    """

    def __init__(self, logdir, verbose):
        self._writer = tf.summary.FileWriter(logdir)
        self._verbose = verbose
        self._global_step = 0
        self._num_episodes = 0
        self._latest_summaries = {}
        self._latest_statistics = {}
        self.logdir = logdir
        self._hidden = set()

    def add_summary(self, name, value, hide):
        """
        Add a known floating-point value summary. Optionally, choose to hide
        the summary, preventing it from being printed during
        advance().
        """
        self._latest_summaries[name] = value
        if hide:
            self._hidden.add(name)

    def add_summary_statistics(self, name, values, hide):
        """
        Add a list of floating values, whose statistics are summarized.
        """
        self._latest_statistics[name] = values
        if hide:
            self._hidden.add(name)

    def advance(self, ts, episodes):
        """Increment the global step and print the previous step statistics"""
        self._global_step += ts
        self._num_episodes += episodes
        self.add_summary('total episodes', self._num_episodes, False)
        self._write_summaries()
        if self._verbose:
            self._print_table()

        for hook in _hooks():
            hook(self._latest_summaries, self._latest_statistics)

        self._latest_summaries = {}
        self._latest_statistics = {}
        self._hidden = set()

    def _print_table(self):
        data = [['summary', 'value', 'min', 'mean', 'max', 'std']]
        data.append(['timesteps', self._global_step, '', '', '', ''])
        for name, value in sorted(self._latest_summaries.items()):
            if name in self._hidden:
                continue
            data.append([name, _floatprint(value), '', '', '', ''])
        for name, values in sorted(self._latest_statistics.items()):
            if name in self._hidden:
                continue
            values = [statistic(values) for statistic in
                      [np.min, np.mean, np.max, np.std]]
            values = [_floatprint(value) for value in values]
            data.append([name, ''] + values)
        print_table(data)

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

    def close(self):
        """Close the internal file writer"""
        self._writer.close()
