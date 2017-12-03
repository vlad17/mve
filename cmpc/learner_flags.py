"""
Learner flags create a learner as specified by the command line, without
us having to do a big switch statement over which learner the caller
desired by relying on inheritance.
"""

from flags import Flags


class LearnerFlags(Flags):
    """Learner flags."""

    def make_learner(self, env):
        """
        Makes a learner.

        Parameters
        ----------
        env: environment in which we will be learning

        Returns
        -------
        learner.Learner
        """
        raise NotImplementedError()


class NoLearnerFlags(LearnerFlags):
    """Used to indicate no learner in use"""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        pass

    @staticmethod
    def name():
        return 'none'

    def __init__(self, _args):
        pass

    def make_learner(self, env):
        return None
