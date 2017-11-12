"""Zero learner flags."""

from learner_flags import LearnerFlags
from zero_learner import ZeroLearner


class ZeroLearnerFlags(LearnerFlags):
    """ZeroLearner flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        # ZeroLearner doesn't have any flags!
        pass

    @staticmethod
    def name():
        return 'zero'

    def __init__(self, _args):
        pass

    def make_learner(self, venv, _sess, _data):
        return ZeroLearner(venv)
