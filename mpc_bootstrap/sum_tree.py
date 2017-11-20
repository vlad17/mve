"""A numpy-backed sum-tree"""

import numpy as np


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

        if parent >= len(self._priorities):
            # hit the leaf with this many samples
            dest[:] = parent - len(self._priorities)
            return

        child1 = parent * 2 + 1
        child2 = child1 + 1
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
        return self._priorities[idxs] / self._prio_sum

    @property
    def _priorities(self):
        tot_len = len(self._tree)
        bottom = tot_len // 2
        return self._tree[bottom:]

    def update_priorities(self, idxs, new_priorities):
        """
        Update the priorities for the corresponding indices.
        idxs should be unique.
        """
        # TODO: simultaneous sum-tree invariant-updating is possible
        # (i.e., dedup parent updates across idxs)
        # TODO: cython would do well here...
        for idx, prio in zip(idxs, new_priorities):
            self._prio_sum += prio - self._priorities[idx]
            idx += len(self._priorities)  # get leaf idx
            self._tree[idx] = prio
            while idx > 0:
                child1 = idx - 1 + (idx & 1)
                child2 = child1 + 1
                idx = child1 // 2
                self._tree[idx] = self._tree[child1] + self._tree[child2]

    def max(self):
        """
        Returns the current maximum priority.
        """
        # TODO: can use a max-heap here for efficiency instead
        return self._priorities.max()
