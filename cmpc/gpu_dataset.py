"""
A ringbuffer that resides on-GPU. Analogous to the dataset.Dataset class
but only stores transition information.
"""

import numpy as np
from tensorflow.python.client import device_lib
import tensorflow as tf

from utils import get_ac_dim, get_ob_dim


def _hasgpu():
    local_device_protos = device_lib.list_local_devices()
    ngpu = sum(device.device_type == 'GPU' for device in local_device_protos)
    return ngpu > 0


class _GPURingBuffer:
    """
    An on-GPU ring buffer with bookkeeping state kept in the python process
    memory.
    """

    def __init__(self, maxlen, shape, dtype='float32'):
        self._maxlen = maxlen
        self._start = 0
        self.length = 0
        device = '/gpu:0' if _hasgpu() else '/cpu:0'
        with tf.device(device):
            self._data = tf.Variable(
                initial_value=tf.zeros((maxlen,) + shape, dtype=dtype),
                trainable=False)
        self._tf_len = tf.Variable(initial_value=0, dtype=tf.int32)

        self._lo_ph1 = tf.placeholder(tf.int32, [])
        self._hi_ph1 = tf.placeholder(tf.int32, [])
        self._source1 = tf.placeholder(dtype, (None,) + shape)

        self._lo_ph2 = tf.placeholder(tf.int32, [])
        self._hi_ph2 = tf.placeholder(tf.int32, [])
        self._source2 = tf.placeholder(dtype, (None,) + shape)

        self._tf_len_update_ph = tf.placeholder(tf.int32, [])

        self._null = np.empty((0,) + shape, dtype)
        self.update = tf.group(
            self._data[self._lo_ph1:self._hi_ph1].assign(self._source1),
            self._data[self._lo_ph2:self._hi_ph2].assign(self._source2),
            tf.assign(self._tf_len, self._tf_len_update_ph))

    def get(self, batch_ix):
        """Get a TF index given a mini-batch index for it"""
        return tf.gather(self._data, batch_ix)

    def append_fd(self, vs):
        """
        Create a feed_dict which feeds self.update to add
        the vs array to the in-GPU ringbuffer.

        This method updates internal counters as if the update had succeeded.
        """
        assert len(vs) <= self._maxlen, (len(vs), self._maxlen)
        fd = {
            self._lo_ph1: 0,
            self._lo_ph2: 0,
            self._hi_ph1: 0,
            self._hi_ph2: 0,
            self._source1: self._null,
            self._source2: self._null,
            self._tf_len_update_ph: self.length
        }

        if self.length < self._maxlen:
            assert self._start == 0, self._start
            room = self._maxlen - self.length
            to_fill = min(len(vs), room)
            fd[self._lo_ph1] = self.length
            fd[self._hi_ph1] = self.length + to_fill
            fd[self._source1] = vs[:to_fill]
            self.length += to_fill
            fd[self._tf_len_update_ph] = self.length
            if to_fill == len(vs):
                return fd
            vs = vs[to_fill:]

        assert self.length == self._maxlen, (self.length, self._maxlen)
        to_fill = min(self._maxlen - self._start, len(vs))
        fd[self._lo_ph2] = self._start
        fd[self._hi_ph2] = self._start + to_fill
        fd[self._source2] = vs[:to_fill]

        self._start = (self._start + to_fill) % self._maxlen
        if to_fill == len(vs):
            return fd

        assert self._start == 0, self._start
        assert fd[self._lo_ph1] == 0, fd[self._lo_ph1]
        assert fd[self._hi_ph1] == 0, fd[self._hi_ph1]
        assert len(vs) < self._maxlen, (len(vs), self._maxlen)

        vs = vs[to_fill:]
        fd[self._lo_ph1] = 0
        fd[self._hi_ph1] = len(vs)
        fd[self._source1] = vs
        return fd

    def tf_sample_uniform(self, n):
        """
        Return a set of randomly-sampled uniform numbers as a TF tensor
        in the range [0, self.length) of the given batch size
        """
        return tf.random_uniform([n], 0, self._tf_len, dtype=tf.int32)


class GPUDataset:
    """
    Stores all data for transitions across several rollouts in
    gpu0, if it is available.
    """

    def __init__(self, env, maxlen):
        ac_dim, ob_dim = get_ac_dim(env), get_ob_dim(env)
        self.maxlen = maxlen
        self._obs = _GPURingBuffer(maxlen, (ob_dim,))
        self._next_obs = _GPURingBuffer(maxlen, (ob_dim,))
        self._rewards = _GPURingBuffer(maxlen, tuple())
        self._acs = _GPURingBuffer(maxlen, (ac_dim,))
        self._terminals = _GPURingBuffer(maxlen, tuple())
        self._update_all = tf.group(
            self._obs.update,
            self._next_obs.update,
            self._rewards.update,
            self._acs.update,
            self._terminals.update)

    def add_paths(self, paths):
        """Aggregate data from a list of paths"""
        for path in paths:
            assert len(path.obs) <= self.maxlen, (len(path.obs), self.maxlen)
            fd = self._obs.append_fd(path.obs)
            fd.update(self._next_obs.append_fd(path.next_obs))
            fd.update(self._rewards.append_fd(path.rewards))
            fd.update(self._acs.append_fd(path.acs))
            fd.update(self._terminals.append_fd(path.terminals))
            tf.get_default_session().run(self._update_all, fd)

    def set_state(self, dictionary):
        """
        Set the internal ringbuffers to contain the values in the dictionary
        (which should be keyed to the corresponding property).
        """
        permissable_keys = ['obs', 'next_obs', 'rewards', 'acs', 'terminals']
        dictionary = {k: dictionary[k] for k in permissable_keys}
        fd = {}
        for attr, arr in dictionary.items():
            rb = getattr(self, '_' + attr)
            assert rb.length == 0, len(rb)
            fd.update(rb.append_fd(arr))
        tf.get_default_session().run(self._update_all, fd)

    def get_batch(self, tf_ix):
        """
        Get a sample of TF tensors from the buffers given an indexing tensor.
        Returns, in the following order:

        observations, next observations, rewards, actions, terminals
        """
        transitions = [self._obs, self._next_obs, self._rewards,
                       self._acs, self._terminals]
        return [x.get(tf_ix) for x in transitions]

    def tf_sample_uniform(self, n):
        """
        Return a set of randomly-sampled uniform numbers as a TF tensor
        in the range [0, self.length) of the given batch size, where
        length is the total number of items in the dataset.
        """
        return self._obs.tf_sample_uniform(n)
