"""Implement a replay buffer storing transitions"""

import numpy as np


class RingBuffer(object):
    """
    A RingBuffer is a finite-size cylic buffer that supports
    fast insertion and access.
    """

    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        """return items at requested indices"""
        return self.data[(self.start + idxs) % self.maxlen]

    def all_data(self):
        """return all data in the buffer"""
        if self.length == self.maxlen:
            return self.data
        assert self.start == 0, self.start
        return self.data[:self.length]

    def append(self, v):
        """Add v to ring buffer, evicting old data if necessary"""
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def append_all(self, vs):
        """Append an array of values"""
        if len(vs) > self.maxlen:
            raise ValueError('cannot add array of size {} to ringbuf size {}'
                             .format(len(vs), self.length))
        if self.length < self.maxlen:
            assert self.start == 0, self.start
            room = self.maxlen - self.length
            to_fill = min(len(vs), room)
            self.data[self.length:self.length + to_fill] = vs[:to_fill]
            self.length += to_fill
            if to_fill == len(vs):
                return
            vs = vs[to_fill:]

        assert self.length == self.maxlen, (self.length, self.maxlen)
        to_fill = min(self.maxlen - self.start, len(vs))
        self.data[self.start:self.start + to_fill] = vs[:to_fill]

        self.start = (self.start + to_fill) % self.maxlen
        if to_fill == len(vs):
            return
        assert self.start == 0, self.start
        assert len(vs) < self.maxlen, (len(vs), self.maxlen)
        vs = vs[to_fill:]
        self.data[:len(vs)] = vs
        self.start = len(self.vs)
