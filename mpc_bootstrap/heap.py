"""
A numpy-backed finite-size max-heap with keys.

Also, a numpy-backed sum-tree
"""

import numpy as np


def _isleaf(heap, idx):
    return (idx + 1) >= len(heap) // 2


def _siblings(idx):
    child1 = idx - 1 + (idx & 1)
    child2 = child1 + 1
    return child1, child2


def _parent(child1):
    return child1 // 2


def _children(parent):
    child1 = parent * 2 + 1
    child2 = child1 + 1
    return child1, child2


class MaxHeap:
    """
    Generate a max heap where the the value being prioritized is a numpy float.

    The size of the heap is a constant specified at creation, and the
    keys are 0..size-1. The value within the heap for a given key may change
    its location in the heap, but key-access will consistently refer to the
    same value, modulo its updates.

    All the values are assumed to be non-negative.
    """

    def __init__(self, size):
        self.size = size
        # round up to the nearest power of 2
        rounded_up = 1 << (size - 1).bit_length()
        self._heap = np.zeros(rounded_up)
        self._key_idxs = np.arange(size, dtype=int)
        self._idx_keys = np.arange(rounded_up, dtype=int)
        self._idx_keys[size:] = -1  # these indices have no key

    def modify_values(self, keys, new_values):
        """Update the priorities for the corresponding keys"""
        # TODO: consider cython
        for key, val in zip(keys, new_values):
            idx = self._key_idxs[key]
            self._heap[idx] = val
            self._sink(val, idx)
            self._swim(val, idx)

    def _sink(self, val, idx):
        while not _isleaf(self._heap, idx):
            child1, child2 = _children(idx)
            val1, val2 = self._heap[child1], self._heap[child2]
            childmax, valmax = child1, val1
            if val2 > val1:
                childmax, valmax = child2, val2
            if valmax > val:
                self._swap(idx, childmax)
                idx = childmax
            else:
                break

    def _swim(self, val, idx):
        while idx > 0:
            parent = _parent(idx)
            pval = self._heap[parent]
            if val > pval:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _swap(self, i, j):
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        keyi, keyj = self._idx_keys[i], self._idx_keys[j]
        self._idx_keys[i], self._idx_keys[j] = keyj, keyi
        self._key_idxs[keyi], self._key_idxs[keyj] = j, i

    def get_values(self, keys):
        """Return the current values for the specified keys"""
        idxs = self._key_idxs[keys]
        return self._heap[idxs]

    def max(self):
        """Returns the current maximum value"""
        return self._heap[0]


class SumTree:
    """
    Given indexed priorities (from 0 to size-1), this class offers log-time
    sampling of the indices [0..size-1] scaled by their priorities.

    Priorities can be arbitrarily scaled (by a fixed constant) probabilities.
    """

    def __init__(self, size):
        self.size = size
        assert size > 0, size
        # round up to nearest power of 2, then double
        rounded_up = 1 << ((size - 1).bit_length() + 1)
        self._tree = np.zeros(rounded_up)
        self._prio_sum = 0

    def sample(self, nsamples):
        """
        Sample according to the priorities (i.e., probability of
        sampling index i is proportional to its priority).

        At least one non-zero priority must be added in.
        """
        assert self._prio_sum > 0, self._prio_sum
        samples = np.zeros(nsamples, dtype=int)
        self._sample_aux(0, samples)
        return samples

    def _sample_aux(self, parent, dest):
        if not dest.size:
            # prune this node from sampling
            return

        if _isleaf(self._tree, parent):
            dest[:] = parent - self._first_leaf()
            return

        child1, child2 = _children(parent)
        prio1, prio2, sum_prio = (
            self._tree[ix] for ix in (child1, child2, parent))
        assert sum_prio == prio1 + prio2, (prio1, prio2, sum_prio)
        assert sum_prio > 0, sum_prio  # could not have gotten to this node

        samp1 = np.random.binomial(dest.size, prio1 / sum_prio)
        self._sample_aux(child1, dest[:samp1])
        self._sample_aux(child2, dest[samp1:])

    def probabilities(self, idxs):
        """
        Return the sample probabilities for the corresponding indices.
        """
        return self._tree[self._first_leaf() + idxs] / self._prio_sum

    def _first_leaf(self):
        tot_len = len(self._tree)
        bottom = tot_len // 2 - 1
        return bottom

    def update_priorities(self, idxs, new_priorities):
        """
        Update the priorities for the corresponding indices.
        idxs should be unique.
        """
        # TODO: simultaneous sum-tree invariant-updating is possible
        # (i.e., dedup parent updates across idxs)
        # TODO: cython would do well here...
        for idx, prio in zip(idxs, new_priorities):
            idx += self._first_leaf()
            self._prio_sum += prio - self._tree[idx]
            self._tree[idx] = prio

            while idx > 0:
                child1, child2 = _siblings(idx)
                idx = _parent(child1)
                self._tree[idx] = self._tree[child1] + self._tree[child2]

    def max(self):
        """
        Returns the current maximum priority.
        """
        # TODO: can use a max-heap here for efficiency instead
        return self._tree[self._first_leaf():].max()
